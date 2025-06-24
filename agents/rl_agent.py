"""
RLAgent:

Classes:
- RLAgent: Tabular Q-learning agent that inherits from PricingAgent

RLAgent:
    - discretize_state(): Transforms continuous state to discrete for Q-table lookup
    - get_q_value(): Retrieves Q-values from the table
    - update_q_value(): Updates Q-values using Q-learning update rule
    - choose_action(): Selects actions using epsilon-greedy policy
    - remember(): Stores experiences in replay memory
    - replay(): Learns from batches of past experiences
    - get_state(): Converts market data into state representation
    - act(): Determines pricing actions based on market observations
    - set_price(): Applies minimum margin constraints to product prices
"""
import numpy as np
import random
from collections import deque
from env.pricing_agent import PricingAgent

class RLAgent(PricingAgent):
    def __init__(self, agent_id, products, learning_rate=0.01, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        super().__init__(agent_id, products)
        
        self.revenue_history = []
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.profit_history = []
        
        self.price_change_options = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
        self.num_actions = len(self.price_change_options)
        self.state_size = 7
        
        self.q_table = {}
        
        self.memory = deque(maxlen=2000)
        
        self.last_revenue = 0
        self.last_state = {}
        self.last_actions = {}
        
    def discretize_state(self, state):
        own_price = state['own_price']
        competitor_price = state['competitor_price']
        own_demand = state['own_demand']
        competitor_demand = state['competitor_demand']
        week = state['week']
        revenue_trend = state['revenue_trend']
        is_holiday = state['is_holiday']
        
        price_ratio_bin = int(np.clip((own_price / competitor_price) * 10, 5, 15)) if competitor_price > 0 else 10
        demand_ratio_bin = int(np.clip((own_demand / competitor_demand) * 5, 0, 10)) if competitor_demand > 0 else 5
        week_bin = int(week / 13)
        trend_bin = int(np.clip(revenue_trend + 2, 0, 4))
        
        discrete_state = (price_ratio_bin, demand_ratio_bin, week_bin, int(is_holiday), trend_bin)
        
        return discrete_state
        
    def get_q_value(self, state, action):
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.num_actions)
        return self.q_table[discrete_state][action]
    
    def update_q_value(self, state, action, reward, next_state):
        discrete_state = self.discretize_state(state)
        next_discrete_state = self.discretize_state(next_state)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.num_actions)
        if next_discrete_state not in self.q_table:
            self.q_table[next_discrete_state] = np.zeros(self.num_actions)
        
        current_q = self.q_table[discrete_state][action]
        next_max_q = np.max(self.q_table[next_discrete_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q)
        
        self.q_table[discrete_state][action] = new_q
    
    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.num_actions)
        
        return np.argmax(self.q_table[discrete_state])
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state in batch:
            self.update_q_value(state, action, reward, next_state)
    
    def get_state(self, product_name, market_observations):
        product = self.products[product_name]
        own_price = product.price
        own_demand = product.quantity_history[-1] if product.quantity_history else 0
        
        category = product.category_cluster
        competitor_price = 0
        competitor_demand = 0
        
        if market_observations and 'category_products' in market_observations:
            if category in market_observations['category_products']:
                for market_product in market_observations['category_products'][category]:
                    if market_product['agent_id'] != self.agent_id:
                        competitor_price = market_product['price']
                        competitor_demand = market_product['last_demand']
                        break
        
        if len(self.revenue_history) >= 2:
            revenue_trend = (self.revenue_history[-1] - self.revenue_history[-2]) / max(1.0, self.revenue_history[-2])
        else:
            revenue_trend = 0
            
        state = {
            'own_price': own_price,
            'competitor_price': competitor_price,
            'own_demand': own_demand, 
            'competitor_demand': competitor_demand,
            'week': market_observations.get('week', 1),
            'revenue_trend': revenue_trend,
            'is_holiday': market_observations.get('is_holiday', False)
        }
        
        return state
    
    def act(self, week, year, is_holiday, market_observations=None):
        total_revenue = sum(self.revenue_history[-1:] or [0])
        reward = total_revenue - self.last_revenue
        self.last_revenue = total_revenue
        
        for product_name, product in self.products.items():
            current_state = self.get_state(product_name, market_observations)
            
            if product_name in self.last_state and product_name in self.last_actions:
                self.remember(
                    self.last_state[product_name], 
                    self.last_actions[product_name],
                    reward,
                    current_state
                )
            
            self.last_state[product_name] = current_state
            
            action_idx = self.choose_action(current_state)
            self.last_actions[product_name] = action_idx
            
            price_change = self.price_change_options[action_idx]
            new_price = product.price * (1 + price_change)
            
            self.set_price(product_name, new_price)
        
        self.replay(32)
        
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay
    
    def set_price(self, product_name, new_price):
        if product_name in self.products:
            min_price = self.products[product_name].cost * 1.05
            new_price = max(min_price, new_price)
            self.products[product_name].price = new_price
            return True
        return