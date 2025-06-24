"""
PricingAgent:

Classes:
- PricingAgent: Base class for all pricing agent implementations

PricingAgent:
    - set_price(): Updates the price of a product
    - act(): Determines pricing strategy based on market observations
    - fulfill_demands(): Processes product demand and calculates revenue
    - find_competing_products(): Identifies competing products in the same category
"""
class PricingAgent:
    def __init__(self, agent_id, products):
        self.agent_id = agent_id
        self.products = {p.name: p for p in products}
        self.revenue_history = []
    
    def set_price(self, product_name, new_price):
        if product_name in self.products:
            self.products[product_name].price = max(0.01, new_price)
            self.products[product_name].price_history.append(new_price)
            return True
        return False
    
    def act(self, week, year, is_holiday, market_observations=None):
        pass
    
    def fulfill_demands(self, demands):
        fulfilled = {}
        total_revenue = 0
        
        for product_name, demand in demands.items():
            if product_name in self.products:
                product = self.products[product_name]
                quantity_sold = demand
                revenue = quantity_sold * product.price
                
                fulfilled[product_name] = {
                    'demand': demand,
                    'sold': quantity_sold,
                    'price': product.price,
                    'revenue': revenue
                }
                
                total_revenue += revenue
        
        self.revenue_history.append(total_revenue)
        return fulfilled
    
    def find_competing_products(self, market_observations):
        competing_products = {}
        
        if market_observations is None:
            return competing_products
            
        for product_name, product in self.products.items():
            category = product.category_cluster
            competing_products[product_name] = []
            
            if category in market_observations.get('category_products', {}):
                for market_product in market_observations['category_products'][category]:
                    if market_product['agent_id'] != self.agent_id:
                        competing_products[product_name].append(market_product)
        
        return