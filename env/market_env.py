"""
MarketEnv:

Classes:
- MarketEnv: Simulates a competitive market environment with product demand prediction

MarketEnv:
    - add_agent(): Adds an agent to the market simulation
    - update_time(): Advances the simulation by one week
    - find_competing_products(): Finds products competing in the same category
    - predict_demand(): Predicts product demand based on features and competition
    - step(): Executes one step of the market simulation
    - get_market_observations(): Collects market data for agent observations
"""
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os

class MarketEnv:
    def __init__(self, agents=None, model_path=None):
        self.agents = agents if agents else []
        self.current_date = datetime(2010, 1, 1)
        self.current_week = 1
        self.current_year = 2010
        self.is_weekend = False
        self.is_holiday_season = False
        
        model_path = model_path or os.path.join(os.path.dirname(__file__), "../models/lgbm_model.pkl")
        self.demand_model = joblib.load(model_path)
        
        self.history = []
        
    def add_agent(self, agent):
        self.agents.append(agent)
    
    def update_time(self):
        self.current_date += timedelta(days=7)
        self.current_week = self.current_date.isocalendar()[1]
        self.current_year = self.current_date.year
        self.is_holiday_season = 40 <= self.current_week <= 52
    
    def find_competing_products(self, product, agent_id):
        competing_products = []
        
        for other_agent in self.agents:
            if other_agent.agent_id == agent_id:
                continue
                
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
        df = pd.DataFrame([features_dict])
        
        required_features = ['Price', 'IsWeekend', 'Year', 'Week', 
                             'IsHolidaySeason', 'CountryCode', 'CategoryCluster',
                             'Quantity_lag_1', 'Quantity_roll_mean_2', 'Quantity_roll_mean_4']
        
        for feature in required_features:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        df['Price_log'] = np.log1p(df['Price'])
        df['Price_squared'] = df['Price'] ** 2
        
        df['SinWeek'] = np.sin(2 * np.pi * df['Week'] / 52)
        df['CosWeek'] = np.cos(2 * np.pi * df['Week'] / 52)
        
        df['MonthApprox'] = ((df['Week'] - 1) // 4) + 1
        df['SinMonth'] = np.sin(2 * np.pi * df['MonthApprox'] / 12)
        df['CosMonth'] = np.cos(2 * np.pi * df['MonthApprox'] / 12)
        
        df['Price_Holiday'] = df['Price'] * df['IsHolidaySeason']
        df['Price_Category'] = df['Price'] * df['CategoryCluster']
        
        df['Trend'] = df['Quantity_roll_mean_4'] - df['Quantity_roll_mean_2']
        df['Acceleration'] = df['Quantity_lag_1'] - df['Quantity_roll_mean_2']
        
        df['Volatility'] = 0
        
        df['Price_vs_Category_Avg'] = 1.0
        
        if competing_products and len(competing_products) > 0:
            avg_category_price = sum(p['price'] for p in competing_products) / len(competing_products)
            if avg_category_price > 0:
                df['Price_vs_Category_Avg'] = df['Price'] / avg_category_price
        
        extended_features = required_features + [
            'Price_log', 'Price_squared', 'SinWeek', 'CosWeek', 
            'SinMonth', 'CosMonth', 'Price_Holiday', 'Price_Category',
            'Trend', 'Acceleration', 'Volatility', 'Price_vs_Category_Avg'
        ]
        
        log_quantity = self.demand_model.predict(df[extended_features], predict_disable_shape_check=True)[0]
        base_quantity = np.expm1(log_quantity)
        
        if competing_products:
            total_weight = sum(p['last_demand'] for p in competing_products) or 1
            avg_competitor_price = sum(p['price'] * p['last_demand'] for p in competing_products) / total_weight
            
            if avg_competitor_price > 0:
                price_ratio = features_dict['Price'] / avg_competitor_price
                
                if price_ratio < 1.0:
                    boost_factor = 1.0 + max(0, (1.0 - price_ratio) * 0.5)
                    base_quantity *= boost_factor
                else:
                    reduction_factor = max(0.7, 1.0 - min(0.3, (price_ratio - 1.0)))
                    base_quantity *= reduction_factor
        
        return max(0, int(round(base_quantity)))
    
    def step(self):
        try:
            all_demands = {}
            all_profits = {}
            
            self.update_time()
            
            market_observations = self.get_market_observations()
            
            shock_multiplier = 1.0
            if hasattr(self, 'enable_shocks') and self.enable_shocks:
                shock_weeks = [26, 52, 78]
                if self.current_week in shock_weeks:
                    print(f"🌊 MARKET SHOCK at Week {self.current_week}: Demand decreased by 30%")
                    shock_multiplier = 0.7
            
            for agent in self.agents:
                agent_demands = {}
                agent_profits = 0
                
                for product_name, product in agent.products.items():
                    competing_products = self.find_competing_products(product, agent.agent_id)
                    
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
                    
                    predicted_demand = self.predict_demand(features, competing_products) * shock_multiplier
                    agent_demands[product_name] = predicted_demand
                    
                    profit = (product.price - product.cost) * predicted_demand
                    agent_profits += profit
                    
                    product.quantity_history.append(predicted_demand)
                    if len(product.quantity_history) > 10:
                        product.quantity_history = product.quantity_history[-10:]
                
                all_demands[agent.agent_id] = agent_demands
                all_profits[agent.agent_id] = agent_profits
                agent.profit_history.append(agent_profits)
                
                agent.revenue_history.append(agent_profits)
            
            self.history.append({
                'week': self.current_week,
                'year': self.current_year,
                'demands': all_demands,
                'profits': all_profits
            })
            
            for agent in self.agents:
                agent.act(self.current_week, self.current_year, self.is_holiday_season, market_observations)
            
            return all_demands, all_profits
        except Exception as e:
            print(f"ERROR in MarketEnv.step(): {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_market_observations(self):
        market_data = {}
        
        category_products = {}
        
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
        
        market_data['category_products'] = category_products
        market_data['week'] = self.current_week
        market_data['year'] = self.current_year
        market_data['is_holiday'] = self.is_holiday_season
        
        return market_data