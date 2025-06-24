"""
RuleBasedAgent:

Classes:
- RuleBasedAgent: Agent that implements simple rule-based pricing strategies

RuleBasedAgent:
    - act(): Determines pricing actions based on selected strategy
    - _static_markup_strategy(): Applies fixed markup over cost
    - _competitor_match_strategy(): Matches or undercuts competitor prices
    - _historical_anchor_strategy(): Maintains price unless demand drops
    - _demand_responsive_strategy(): Adjusts price based on demand levels
    - _seasonal_pricing_strategy(): Applies higher markups during holidays
    - set_price(): Sets product price with validation
"""
import numpy as np
from env.pricing_agent import PricingAgent

class RuleBasedAgent(PricingAgent):
    
    VALID_STRATEGIES = [
        "static_markup",
        "competitor_match",
        "historical_anchor",
        "demand_responsive",
        "seasonal_pricing"
    ]
    
    def __init__(self, agent_id, products, strategy="static_markup", markup_pct=0.20, 
                 undercut_pct=0.05, demand_threshold=0.10, seasonal_boost=0.10):
        super().__init__(agent_id, products)
        
        self.revenue_history = []
        self.profit_history = []
        
        if strategy not in self.VALID_STRATEGIES:
            print(f"Warning: Invalid strategy '{strategy}'. Defaulting to 'static_markup'.")
            self.strategy = "static_markup"
        else:
            self.strategy = strategy
            
        self.markup_pct = markup_pct
        self.undercut_pct = undercut_pct
        self.demand_threshold = demand_threshold
        self.seasonal_boost = seasonal_boost
        
        self.historical_demand = {}
        self.historical_prices = {}
        for product_name in self.products.keys():
            self.historical_demand[product_name] = []
            self.historical_prices[product_name] = []
    
    def act(self, week, year, is_holiday, market_observations=None):
        for product_name, product in self.products.items():
            current_price = product.price
            current_demand = product.quantity_history[-1] if product.quantity_history else 0
            
            self.historical_demand[product_name].append(current_demand)
            self.historical_prices[product_name].append(current_price)
            
            if self.strategy == "static_markup":
                new_price = self._static_markup_strategy(product)
            elif self.strategy == "competitor_match":
                new_price = self._competitor_match_strategy(product, market_observations)
            elif self.strategy == "historical_anchor":
                new_price = self._historical_anchor_strategy(product, product_name)
            elif self.strategy == "demand_responsive":
                new_price = self._demand_responsive_strategy(product, product_name)
            elif self.strategy == "seasonal_pricing":
                new_price = self._seasonal_pricing_strategy(product, is_holiday)
            else:
                new_price = current_price
            
            self.set_price(product_name, new_price)
    
    def _static_markup_strategy(self, product):
        return product.cost * (1 + self.markup_pct)
    
    def _competitor_match_strategy(self, product, market_observations):
        category = product.category_cluster
        competitor_prices = []
        
        if market_observations and 'category_products' in market_observations:
            if category in market_observations['category_products']:
                for market_product in market_observations['category_products'][category]:
                    if market_product['agent_id'] != self.agent_id:
                        competitor_prices.append(market_product['price'])
        
        if competitor_prices:
            avg_competitor_price = np.mean(competitor_prices)
            return avg_competitor_price * (1 - self.undercut_pct)
        else:
            return self._static_markup_strategy(product)
    
    def _historical_anchor_strategy(self, product, product_name):
        history_len = len(self.historical_demand[product_name])
        
        if history_len >= 2:
            current_demand = self.historical_demand[product_name][-1]
            previous_demand = self.historical_demand[product_name][-2]
            previous_price = self.historical_prices[product_name][-1]
            
            demand_change = (current_demand / max(1, previous_demand)) - 1.0
            
            if demand_change < -self.demand_threshold:
                return previous_price * 0.95
            elif demand_change > self.demand_threshold:
                return previous_price * 1.02
            else:
                return previous_price
        else:
            return self._static_markup_strategy(product)
    
    def _demand_responsive_strategy(self, product, product_name):
        history_len = len(self.historical_demand[product_name])
        
        if history_len >= 3:
            recent_demands = self.historical_demand[product_name][-3:]
            avg_demand = np.mean(recent_demands)
            
            if len(recent_demands) >= 2:
                trend = recent_demands[-1] - recent_demands[-2]
            else:
                trend = 0
            
            current_price = product.price
            
            if avg_demand > 10 and trend >= 0:
                return current_price * (1 + 0.03)
            elif avg_demand < 5 or trend < 0:
                return current_price * (1 - 0.05)
            else:
                return current_price
        else:
            return self._static_markup_strategy(product)
    
    def _seasonal_pricing_strategy(self, product, is_holiday):
        base_price = self._static_markup_strategy(product)
        
        if is_holiday:
            return base_price * (1 + self.seasonal_boost)
        else:
            return base_price
    
    def set_price(self, product_name, new_price):
        if product_name in self.products:
            min_price = self.products[product_name].cost * 1.02
            final_price = max(min_price, new_price)
            
            self.products[product_name].price = final_price
            self.products[product_name].price_history.append(final_price)
            return True
        return False