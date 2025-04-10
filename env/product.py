class Product:
    def __init__(self, name, price, category_cluster, country_code=1, cost=None):
        self.name = name
        self.price = price
        self.category_cluster = category_cluster
        self.country_code = country_code
        self.cost = cost or price * 0.5  # Default cost is 70% of initial price
        self.quantity_history = [0, 0, 0, 0]  # Initialize with zeros for lag features
        self.price_history = [price]
        
        # Add this line to fix the AttributeError when fulfill_demand is called
        self.stock = 10000  # Initialize with sufficient stock

    def update_price(self, new_price: float):
        self.price = new_price

    def fulfill_demand(self, demand: int) -> int:
        fulfilled = min(demand, self.stock)
        self.stock -= fulfilled
        return fulfilled

    def restock(self, amount: int):
        self.stock += amount