"""
Product:

Classes:
- Product: Represents a product with pricing and inventory attributes

Product:
    - update_price(): Changes the product price
    - fulfill_demand(): Processes demand and reduces inventory
"""
class Product:
    def __init__(self, name, price, category_cluster, country_code=1, cost=None):
        self.name = name
        self.price = price
        self.category_cluster = category_cluster
        self.country_code = country_code
        self.cost = cost or price * 0.5
        self.quantity_history = [0, 0, 0, 0]
        self.price_history = [price]

    def update_price(self, new_price: float):
        self.price = new_price

    def fulfill_demand(self, demand: int) -> int:
        fulfilled = min(demand, self.stock)
        self.stock -= fulfilled
        return fulfilled