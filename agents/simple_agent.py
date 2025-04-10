from env.pricing_agent import PricingAgent
import random

class SimpleAgent(PricingAgent):
    def __init__(self, agent_id, products, strategy="random"):
        super().__init__(agent_id, products)
        self.strategy = strategy
    
    def act(self, week, year, is_holiday):
        """Implement pricing strategy"""
        if self.strategy == "random":
            # Randomly adjust prices within ±10%
            for product_name, product in self.products.items():
                current_price = product.price
                adjustment = random.uniform(-0.1, 0.1)  # ±10%
                new_price = current_price * (1 + adjustment)
                self.set_price(product_name, new_price)
                
        elif self.strategy == "seasonal":
            # Increase prices during holiday season
            for product_name, product in self.products.items():
                current_price = product.price
                if is_holiday:
                    # Increase prices by 5-15% during holidays
                    new_price = current_price * random.uniform(1.05, 1.15)
                else:
                    # Decrease prices by 0-5% outside holidays
                    new_price = current_price * random.uniform(0.95, 1.0)
                self.set_price(product_name, new_price)
        
        # Could add more sophisticated strategies here