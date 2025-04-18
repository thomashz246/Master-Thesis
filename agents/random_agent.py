import numpy as np
import random
from env.pricing_agent import PricingAgent

class RandomPricingAgent(PricingAgent):
    """
    A simple agent that makes random price adjustments.
    Serves as a lower-bound baseline for comparison with more sophisticated agents.
    """
    
    def __init__(self, agent_id, products, min_adjustment=-0.10, max_adjustment=0.10):
        """
        Initialize the Random Pricing Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            products: Dictionary of products this agent manages
            min_adjustment: Minimum price adjustment percentage (default: -10%)
            max_adjustment: Maximum price adjustment percentage (default: +10%)
        """
        super().__init__(agent_id, products)
        
        # Initialize history
        self.revenue_history = []
        self.profit_history = []
        
        # Define possible price adjustments
        self.price_adjustments = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
        
        # Set adjustment bounds
        self.min_adjustment = min_adjustment
        self.max_adjustment = max_adjustment
    
    def act(self, week, year, is_holiday, market_observations=None):
        """
        Take action by randomly adjusting prices of all products.
        
        Args:
            week: Current week number
            year: Current year
            is_holiday: Boolean indicating if current period is a holiday
            market_observations: Observations about the market and competitors (not used)
        """
        for product_name, product in self.products.items():
            # Select a random price adjustment
            price_adjustment = random.choice(self.price_adjustments)
            
            # Calculate new price
            current_price = product.price
            new_price = current_price * (1 + price_adjustment)
            
            # Ensure price doesn't go below cost + minimum margin
            min_price = product.cost * 1.05  # 5% minimum margin
            new_price = max(min_price, new_price)
            
            # Set the new price
            self.set_price(product_name, new_price)
    
    def set_price(self, product_name, new_price):
        """Set the price of a product with some basic validation"""
        if product_name in self.products:
            self.products[product_name].price = new_price
            self.products[product_name].price_history.append(new_price)
            return True
        return False