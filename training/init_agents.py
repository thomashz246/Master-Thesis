import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.pricing_agent import PricingAgent
from env.product import Product


def generate_random_products(product_names):
    products = {}
    for name in product_names:
        stock = random.randint(50, 200)         # random stock between 50–200 units
        price = round(random.uniform(5.0, 20.0), 2)  # random price between €5.00–€20.00
        products[name] = Product(name=name, initial_stock=stock, initial_price=price)
    return products

def init_agents():
    product_names = ["A", "B", "C"]

    agent1_products = generate_random_products(product_names)
    agent2_products = generate_random_products(product_names)

    # Pass the values (Product objects) from the dictionaries instead of the dictionaries themselves
    agent1 = PricingAgent(agent_id="Agent1", products=list(agent1_products.values()))
    agent2 = PricingAgent(agent_id="Agent2", products=list(agent2_products.values()))

    return agent1, agent2

if __name__ == "__main__":
    agent1, agent2 = init_agents()

    print("🔹 Agent 1 Inventory:")
    for p in agent1.products.values():
        print(f"  • {p.name}: {p.stock} units at €{p.price}")

    print("\n🔹 Agent 2 Inventory:")
    for p in agent2.products.values():
        print(f"  • {p.name}: {p.stock} units at €{p.price}")