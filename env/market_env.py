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
        
        # Add the same engineered features as in demand.py
        df['Price_log'] = np.log1p(df['Price'])
        df['Price_squared'] = df['Price'] ** 2
        
        # Seasonal decomposition
        df['SinWeek'] = np.sin(2 * np.pi * df['Week'] / 52)
        df['CosWeek'] = np.cos(2 * np.pi * df['Week'] / 52)
        
        # Monthly seasonality approximation
        df['MonthApprox'] = ((df['Week'] - 1) // 4) + 1
        df['SinMonth'] = np.sin(2 * np.pi * df['MonthApprox'] / 12)
        df['CosMonth'] = np.cos(2 * np.pi * df['MonthApprox'] / 12)
        
        # Interaction terms
        df['Price_Holiday'] = df['Price'] * df['IsHolidaySeason']
        df['Price_Category'] = df['Price'] * df['CategoryCluster']
        
        # Trend indicators - use existing data if available
        df['Trend'] = df['Quantity_roll_mean_4'] - df['Quantity_roll_mean_2']
        df['Acceleration'] = df['Quantity_lag_1'] - df['Quantity_roll_mean_2']
        
        # Volatility - simplified version since we don't have category-wide data
        df['Volatility'] = 0  # Default to 0 volatility
        
        # Price position metrics - use market data if available
        df['Price_vs_Category_Avg'] = 1.0  # Default to neutral position
        
        # If we have competing products, calculate a better price ratio
        if competing_products and len(competing_products) > 0:
            avg_category_price = sum(p['price'] for p in competing_products) / len(competing_products)
            if avg_category_price > 0:
                df['Price_vs_Category_Avg'] = df['Price'] / avg_category_price
        
        # Extend required features with new engineered features
        extended_features = required_features + [
            'Price_log', 'Price_squared', 'SinWeek', 'CosWeek', 
            'SinMonth', 'CosMonth', 'Price_Holiday', 'Price_Category',
            'Trend', 'Acceleration', 'Volatility', 'Price_vs_Category_Avg'
        ]
        
        # Predict with the ML model using all features
        log_quantity = self.demand_model.predict(df[extended_features], predict_disable_shape_check=True)[0]
        base_quantity = np.expm1(log_quantity)
        
        # Apply competitive effects (optional, your model may already handle this)
        if competing_products:
            # Calculate average competitor price
            total_weight = sum(p['last_demand'] for p in competing_products) or 1
            avg_competitor_price = sum(p['price'] * p['last_demand'] for p in competing_products) / total_weight
            
            # Calculate price ratio (our price / competitor price)
            if avg_competitor_price > 0:
                price_ratio = features_dict['Price'] / avg_competitor_price
                
                # Apply moderate competition effects (reduced from original implementation)
                # since the model already learned these patterns
                if price_ratio < 1.0:  # Our price is lower
                    boost_factor = 1.0 + max(0, (1.0 - price_ratio) * 0.5)
                    base_quantity *= boost_factor
                else:  # Our price is higher
                    reduction_factor = max(0.7, 1.0 - min(0.3, (price_ratio - 1.0)))
                    base_quantity *= reduction_factor
        
        # Ensure quantity is non-negative integer
        return max(0, int(round(base_quantity)))
    
    def step(self):
        """Execute one step of the market simulation"""
        try:
            # Store demand predictions for all products from all agents
            all_demands = {}
            all_profits = {}
            
            # Update market conditions
            self.update_time()
            
            # First, get all market observations to share with agents
            market_observations = self.get_market_observations()
            
            # Check if current week is a shock week
            shock_multiplier = 1.0
            if hasattr(self, 'enable_shocks') and self.enable_shocks:
                shock_weeks = [26, 52, 78]  # Shock at weeks 26, 52, and 78
                if self.current_week in shock_weeks:
                    print(f"🌊 MARKET SHOCK at Week {self.current_week}: Demand decreased by 30%")
                    shock_multiplier = 0.7  # 30% decrease
            
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
                    predicted_demand = self.predict_demand(features, competing_products) * shock_multiplier
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