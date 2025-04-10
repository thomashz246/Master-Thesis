from env.pricing_agent import PricingAgent
import numpy as np
import random

class LearningAgent(PricingAgent):
    def __init__(self, agent_id, products, strategy="competitive", learning_rate=0.1):
        super().__init__(agent_id, products)
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.price_history = {}  # Track price changes and outcomes
        
    def act(self, week, year, is_holiday, market_observations=None):
        """Implement pricing strategy that learns from competitors"""
        if self.strategy == "competitive":
            competing_products = self.find_competing_products(market_observations)
            
            # For each product, adjust price based on competitors
            for product_name, product in self.products.items():
                current_price = product.price
                competitors = competing_products.get(product_name, [])
                
                # Calculate the weighted average competitor price (if any)
                if competitors:
                    # Get competitor prices and their last demands
                    competitor_prices = [c['price'] for c in competitors]
                    competitor_demands = [c['last_demand'] for c in competitors]
                    
                    # Calculate demand-weighted average price
                    if sum(competitor_demands) > 0:
                        weighted_avg_price = sum(p * d for p, d in zip(competitor_prices, competitor_demands)) / sum(competitor_demands)
                    else:
                        weighted_avg_price = np.mean(competitor_prices)
                    
                    # Adjust our price towards the weighted average with some noise
                    # Lower learning rate = slower adaptation
                    price_diff = weighted_avg_price - current_price
                    adjustment = price_diff * self.learning_rate
                    
                    # Add some exploration noise
                    noise = random.uniform(-0.05, 0.05) * current_price
                    
                    new_price = current_price + adjustment + noise
                    
                    # Make sure price remains reasonable
                    new_price = max(current_price * 0.8, min(new_price, current_price * 1.2))
                    
                    # Set the new price
                    self.set_price(product_name, new_price)
                else:
                    # No competitors, adjust based on holiday season
                    if is_holiday:
                        new_price = current_price * random.uniform(1.02, 1.08)
                    else:
                        new_price = current_price * random.uniform(0.97, 1.03)
                    self.set_price(product_name, new_price)
                    
        elif self.strategy == "follower":
            # A simpler strategy that just tracks the lowest competitor price
            competing_products = self.find_competing_products(market_observations)
            
            for product_name, product in self.products.items():
                current_price = product.price
                competitors = competing_products.get(product_name, [])
                
                if competitors:
                    # Find the most successful competitor (highest demand)
                    most_successful = max(competitors, key=lambda x: x['last_demand'])
                    target_price = most_successful['price']
                    
                    # Match their price with a small discount
                    discount = random.uniform(0.95, 0.99)
                    new_price = target_price * discount
                    self.set_price(product_name, new_price)
                else:
                    # Default behavior when no competitors
                    self.set_price(product_name, current_price * random.uniform(0.97, 1.03))