class PricingAgent:
    def __init__(self, agent_id, products):
        self.agent_id = agent_id
        self.products = {p.name: p for p in products}
        self.revenue_history = []  # Initialize empty list
    
    def set_price(self, product_name, new_price):
        """Update the price of a product"""
        if product_name in self.products:
            self.products[product_name].price = max(0.01, new_price)  # Ensure price is positive
            self.products[product_name].price_history.append(new_price)
            return True
        return False
    
    def act(self, week, year, is_holiday, market_observations=None):
        """Define the agent's pricing strategy based on market observations"""
        # This is a placeholder to be overridden by specific agent implementations
        pass
    
    def fulfill_demands(self, demands):
        """Process demand for each product and return quantities sold and revenue"""
        fulfilled = {}
        total_revenue = 0
        
        for product_name, demand in demands.items():
            if product_name in self.products:
                product = self.products[product_name]
                quantity_sold = demand  # In a simple model, we fulfill all demand
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
        """Find competing products in the same category as this agent's products"""
        competing_products = {}
        
        if market_observations is None:
            return competing_products
            
        # For each of our products, find competitors in the same category
        for product_name, product in self.products.items():
            category = product.category_cluster
            competing_products[product_name] = []
            
            if category in market_observations.get('category_products', {}):
                # Find all products from other agents in the same category
                for market_product in market_observations['category_products'][category]:
                    if market_product['agent_id'] != self.agent_id:  # Only include other agents' products
                        competing_products[product_name].append(market_product)
        
        return competing_products