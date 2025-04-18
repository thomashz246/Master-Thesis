import numpy as np
from env.pricing_agent import PricingAgent

class RuleBasedAgent(PricingAgent):
    """
    Agent that implements simple rule-based pricing strategies.
    These strategies simulate common business heuristics used in retail pricing.
    """
    
    VALID_STRATEGIES = [
        "static_markup",       # Fixed markup over cost
        "competitor_match",    # Match or slightly undercut competitors
        "historical_anchor",   # Maintain price unless demand drops
        "demand_responsive",   # Increase price when demand is high, decrease when low
        "seasonal_pricing"     # Apply higher markups during holiday periods
    ]
    
    def __init__(self, agent_id, products, strategy="static_markup", markup_pct=0.20, 
                 undercut_pct=0.05, demand_threshold=0.10, seasonal_boost=0.10):
        """
        Initialize the Rule-Based Pricing Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            products: Dictionary of products this agent manages
            strategy: Pricing strategy to use (from VALID_STRATEGIES)
            markup_pct: Percentage markup over cost (for static_markup strategy)
            undercut_pct: Percentage to undercut competitors (for competitor_match)
            demand_threshold: Threshold for demand change to trigger price adjustment
            seasonal_boost: Extra markup during holiday season
        """
        super().__init__(agent_id, products)
        
        # Initialize history
        self.revenue_history = []
        self.profit_history = []
        
        # Set strategy
        if strategy not in self.VALID_STRATEGIES:
            print(f"Warning: Invalid strategy '{strategy}'. Defaulting to 'static_markup'.")
            self.strategy = "static_markup"
        else:
            self.strategy = strategy
            
        # Strategy parameters
        self.markup_pct = markup_pct
        self.undercut_pct = undercut_pct
        self.demand_threshold = demand_threshold
        self.seasonal_boost = seasonal_boost
        
        # Track historical demand and prices
        self.historical_demand = {}
        self.historical_prices = {}
        for product_name in self.products.keys():
            self.historical_demand[product_name] = []
            self.historical_prices[product_name] = []
    
    def act(self, week, year, is_holiday, market_observations=None):
        """
        Apply the selected pricing strategy to each product.
        
        Args:
            week: Current week number
            year: Current year
            is_holiday: Boolean indicating if current period is a holiday season
            market_observations: Observations about the market and competitors
        """
        for product_name, product in self.products.items():
            # Get current values
            current_price = product.price
            current_demand = product.quantity_history[-1] if product.quantity_history else 0
            
            # Store historical data
            self.historical_demand[product_name].append(current_demand)
            self.historical_prices[product_name].append(current_price)
            
            # Apply the selected strategy
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
                # Fallback strategy
                new_price = current_price
            
            # Apply the new price
            self.set_price(product_name, new_price)
    
    def _static_markup_strategy(self, product):
        """Simple fixed markup over cost"""
        return product.cost * (1 + self.markup_pct)
    
    def _competitor_match_strategy(self, product, market_observations):
        """Match or slightly undercut competitor prices in the same category"""
        category = product.category_cluster
        competitor_prices = []
        
        # Collect competitors' prices in the same category
        if market_observations and 'category_products' in market_observations:
            if category in market_observations['category_products']:
                for market_product in market_observations['category_products'][category]:
                    if market_product['agent_id'] != self.agent_id:
                        competitor_prices.append(market_product['price'])
        
        if competitor_prices:
            # Calculate average competitor price and undercut it
            avg_competitor_price = np.mean(competitor_prices)
            return avg_competitor_price * (1 - self.undercut_pct)
        else:
            # No competitors found, use static markup
            return self._static_markup_strategy(product)
    
    def _historical_anchor_strategy(self, product, product_name):
        """Maintain last week's price unless demand has dropped significantly"""
        history_len = len(self.historical_demand[product_name])
        
        if history_len >= 2:
            current_demand = self.historical_demand[product_name][-1]
            previous_demand = self.historical_demand[product_name][-2]
            previous_price = self.historical_prices[product_name][-1]
            
            # Check if demand has dropped significantly
            demand_change = (current_demand / max(1, previous_demand)) - 1.0
            
            if demand_change < -self.demand_threshold:
                # Demand dropped, reduce price slightly
                return previous_price * 0.95
            elif demand_change > self.demand_threshold:
                # Demand increased, try a small price increase
                return previous_price * 1.02
            else:
                # Demand stable, keep the same price
                return previous_price
        else:
            # Not enough history, use static markup
            return self._static_markup_strategy(product)
    
    def _demand_responsive_strategy(self, product, product_name):
        """Increase price when demand is high, decrease when low"""
        history_len = len(self.historical_demand[product_name])
        
        if history_len >= 3:
            # Calculate average demand over recent periods
            recent_demands = self.historical_demand[product_name][-3:]
            avg_demand = np.mean(recent_demands)
            
            # Get trend direction
            if len(recent_demands) >= 2:
                trend = recent_demands[-1] - recent_demands[-2]
            else:
                trend = 0
            
            current_price = product.price
            
            # Adjust price based on demand level and trend
            if avg_demand > 10 and trend >= 0:
                # High demand and stable/increasing trend - increase price
                return current_price * (1 + 0.03)  # 3% increase
            elif avg_demand < 5 or trend < 0:
                # Low demand or declining trend - decrease price
                return current_price * (1 - 0.05)  # 5% decrease
            else:
                # Moderate demand - maintain price
                return current_price
        else:
            # Not enough history, use static markup
            return self._static_markup_strategy(product)
    
    def _seasonal_pricing_strategy(self, product, is_holiday):
        """Apply higher markup during holiday periods"""
        base_price = self._static_markup_strategy(product)
        
        if is_holiday:
            # Apply holiday season boost
            return base_price * (1 + self.seasonal_boost)
        else:
            return base_price
    
    def set_price(self, product_name, new_price):
        """Set the price of a product with some basic validation"""
        if product_name in self.products:
            # Ensure price doesn't go below cost + minimum margin
            min_price = self.products[product_name].cost * 1.02  # 2% minimum margin
            final_price = max(min_price, new_price)
            
            self.products[product_name].price = final_price
            self.products[product_name].price_history.append(final_price)
            return True
        return False