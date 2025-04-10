import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os

class MarketEnv:
    def __init__(self, agents=None, model_path=None):
        self.agents = agents if agents else []
        self.current_date = datetime(2010, 1, 1)  # Starting date
        self.current_week = 1
        self.current_year = 2010
        self.is_weekend = False
        self.is_holiday_season = False
        
        # Load demand prediction model
        model_path = model_path or os.path.join(os.path.dirname(__file__), "../models/lgbm_model.pkl")
        self.demand_model = joblib.load(model_path)
        
        # History tracking
        self.history = []
        
    def add_agent(self, agent):
        self.agents.append(agent)
    
    def update_time(self):
        """Move time forward by one week"""
        self.current_date += timedelta(days=7)
        self.current_week = self.current_date.isocalendar()[1]
        self.current_year = self.current_date.year
        self.is_holiday_season = 40 <= self.current_week <= 52  # Set holiday season for last quarter
    
    def find_competing_products(self, product, agent_id):
        """Find competing products in the same category"""
        competing_products = []
        
        for other_agent in self.agents:
            # Skip the same agent
            if other_agent.agent_id == agent_id:
                continue
                
            # Find products in the same category
            for other_product_name, other_product in other_agent.products.items():
                if other_product.category_cluster == product.category_cluster:
                    competing_products.append({
                        'agent_id': other_agent.agent_id,
                        'product_name': other_product_name,
                        'price': other_product.price,
                        'last_demand': other_product.quantity_history[-1] if other_product.quantity_history else 0
                    })
        
        return competing_products

    def predict_demand(self, features_dict, competing_products=None):
        """Predict demand based on features and competitive factors"""
        # Create a DataFrame with the expected features
        df = pd.DataFrame([features_dict])
        
        # Ensure all required features are present
        required_features = ['Price', 'IsWeekend', 'Year', 'Week', 
                             'IsHolidaySeason', 'CountryCode', 'CategoryCluster',
                             'Quantity_lag_1', 'Quantity_roll_mean_2', 'Quantity_roll_mean_4']
        
        for feature in required_features:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Predict base demand with the ML model
        log_quantity = self.demand_model.predict(df[required_features])
        base_quantity = np.expm1(log_quantity)[0]
        
        # Debugging output
        # print(f"Base quantity from model: {base_quantity}, features: {features_dict}")
        
        # Apply competitive price effects
        if competing_products:
            # Calculate average competitor price (weighted by demand)
            total_weight = sum(p['last_demand'] for p in competing_products) or 1  # Avoid division by zero
            avg_competitor_price = sum(p['price'] * p['last_demand'] for p in competing_products) / total_weight
            
            # Calculate price ratio (our price / competitor price)
            if avg_competitor_price > 0:
                price_ratio = features_dict['Price'] / avg_competitor_price
                
                # Price elasticity effect - when our price is lower, demand increases
                if price_ratio < 1.0:  # Our price is lower
                    # Increase demand when our price is lower (up to 2x boost at significant discount)
                    boost_factor = 1.0 + max(0, (1.0 - price_ratio) * 2)  # Linear boost up to 2x
                    base_quantity *= boost_factor
                else:  # Our price is higher
                    # Decrease demand when our price is higher (down to 0.3x at significant premium)
                    reduction_factor = max(0.3, 1.0 - min(0.7, (price_ratio - 1.0)))  # Linear reduction down to 0.3x
                    base_quantity *= reduction_factor
        
        # Ensure quantity is non-negative integer
        return max(0, int(round(base_quantity)))
    
    def step(self):
        """Execute one step of the market simulation"""
        # print(f"MarketEnv.step(): Week {self.current_week}, Year {self.current_year}")
        
        try:
            # Store demand predictions for all products from all agents
            all_demands = {}
            all_profits = {}
            
            # Update market conditions
            self.update_time()
            
            # First, get all market observations to share with agents
            market_observations = self.get_market_observations()
            
            # For each agent, predict demand and calculate profit
            for agent in self.agents:
                agent_demands = {}
                agent_profits = 0
                
                # For each product the agent sells
                for product_name, product in agent.products.items():
                    # Find competing products in the same category
                    competing_products = self.find_competing_products(product, agent.agent_id)
                    
                    # Generate features for the product
                    features = {
                        'Price': product.price,
                        'IsWeekend': int(self.is_weekend),
                        'Year': self.current_year,
                        'Week': self.current_week,
                        'IsHolidaySeason': int(self.is_holiday_season),
                        'CountryCode': product.country_code,
                        'CategoryCluster': product.category_cluster,
                        'Quantity_lag_1': product.quantity_history[-1] if product.quantity_history else 0,
                        'Quantity_roll_mean_2': np.mean(product.quantity_history[-2:]) if len(product.quantity_history) >= 2 else 0,
                        'Quantity_roll_mean_4': np.mean(product.quantity_history[-4:]) if len(product.quantity_history) >= 4 else 0
                    }
                    
                    # Predict demand with competitive price effects
                    predicted_demand = self.predict_demand(features, competing_products)
                    agent_demands[product_name] = predicted_demand
                    
                    # Calculate profit
                    profit = (product.price - product.cost) * predicted_demand
                    agent_profits += profit
                    
                    # Update product history
                    product.quantity_history.append(predicted_demand)
                    if len(product.quantity_history) > 10:  # Keep only recent history
                        product.quantity_history = product.quantity_history[-10:]
                
                all_demands[agent.agent_id] = agent_demands
                all_profits[agent.agent_id] = agent_profits
                agent.profit_history.append(agent_profits)
                
                # Add this line to store revenue in agent history
                agent.revenue_history.append(agent_profits)
            
            # Record history
            self.history.append({
                'week': self.current_week,
                'year': self.current_year,
                'demands': all_demands,
                'profits': all_profits
            })
            
            # Print demands and revenues for each agent
            for agent_id, agent_demands in all_demands.items():
                agent_revenues = all_profits[agent_id]
                # print(f"Week {self.current_week}: {agent_id} demands: {agent_demands}, revenue: {agent_revenues}")
            
            # Let agents take actions for next step
            for agent in self.agents:
                agent.act(self.current_week, self.current_year, self.is_holiday_season, market_observations)
            
            # Before returning
            return all_demands, all_profits
        except Exception as e:
            print(f"ERROR in MarketEnv.step(): {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to be caught by the outer try/except
    
    def get_market_observations(self):
        """Collect market observations including all prices and demands"""
        market_data = {}
        
        # Group products by category for easy comparison
        category_products = {}
        
        # Collect all products by category cluster
        for agent in self.agents:
            for product_name, product in agent.products.items():
                category = product.category_cluster
                if category not in category_products:
                    category_products[category] = []
                    
                category_products[category].append({
                    'agent_id': agent.agent_id,
                    'product_name': product_name,
                    'price': product.price,
                    'last_demand': product.quantity_history[-1] if product.quantity_history else 0,
                    'avg_demand_4w': np.mean(product.quantity_history[-4:]) if product.quantity_history else 0
                })
        
        # Format the data for agents to consume
        market_data['category_products'] = category_products
        market_data['week'] = self.current_week
        market_data['year'] = self.current_year
        market_data['is_holiday'] = self.is_holiday_season
        
        return market_data