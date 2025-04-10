import numpy as np
import random
from collections import deque
from env.pricing_agent import PricingAgent

class RLAgent(PricingAgent):
    def __init__(self, agent_id, products, learning_rate=0.01, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        super().__init__(agent_id, products)
        
        # Initialize revenue history
        self.revenue_history = []
        
        # RL parameters
        self.learning_rate = learning_rate          # Alpha: learning rate
        self.discount_factor = discount_factor      # Gamma: future reward discount
        self.exploration_rate = exploration_rate    # Epsilon: exploration vs exploitation
        self.exploration_decay = exploration_decay  # Epsilon decay rate
        self.min_exploration = min_exploration      # Minimum exploration probability
        self.profit_history = []                    # Track profit history
        
        # State and action spaces
        self.price_change_options = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])  # Possible price changes
        self.num_actions = len(self.price_change_options)
        self.state_size = 7  # [own_price, competitor_price, own_demand, competitor_demand, week, revenue_trend, is_holiday]
        
        # Initialize Q-table (state -> action values)
        # We'll use a dictionary for sparse representation as state space is continuous
        self.q_table = {}
        
        # Experience replay buffer
        self.memory = deque(maxlen=2000)
        
        # Track performance for learning
        self.last_revenue = 0
        self.last_state = {}
        self.last_actions = {}
        
    def discretize_state(self, state):
        """Convert continuous state to discrete bins for Q-table lookup"""
        # Extract state components
        own_price = state['own_price']
        competitor_price = state['competitor_price']
        own_demand = state['own_demand']
        competitor_demand = state['competitor_demand']
        week = state['week']
        revenue_trend = state['revenue_trend']
        is_holiday = state['is_holiday']
        
        # Discretize each dimension
        price_ratio_bin = int(np.clip((own_price / competitor_price) * 10, 5, 15)) if competitor_price > 0 else 10
        demand_ratio_bin = int(np.clip((own_demand / competitor_demand) * 5, 0, 10)) if competitor_demand > 0 else 5
        week_bin = int(week / 13)  # Quarter of the year
        trend_bin = int(np.clip(revenue_trend + 2, 0, 4))  # Trend from -2 (declining) to +2 (growing)
        
        # Create a hashable state representation
        discrete_state = (price_ratio_bin, demand_ratio_bin, week_bin, int(is_holiday), trend_bin)
        
        return discrete_state
        
    def get_q_value(self, state, action):
        """Get Q-value from table with default 0"""
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.num_actions)
        return self.q_table[discrete_state][action]
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        discrete_state = self.discretize_state(state)
        next_discrete_state = self.discretize_state(next_state)
        
        # Create state entries if they don't exist
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.num_actions)
        if next_discrete_state not in self.q_table:
            self.q_table[next_discrete_state] = np.zeros(self.num_actions)
        
        # Q-learning update rule
        current_q = self.q_table[discrete_state][action]
        next_max_q = np.max(self.q_table[next_discrete_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q)
        
        self.q_table[discrete_state][action] = new_q
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        
        # Exploitation: best known action
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.num_actions)
        
        return np.argmax(self.q_table[discrete_state])
    
    def remember(self, state, action, reward, next_state):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state))
    
    def replay(self, batch_size=32):
        """Learn from random batch of experiences"""
        if len(self.memory) < batch_size:
            return
            
        # Sample random experiences
        batch = random.sample(self.memory, batch_size)
        
        # Learn from each experience
        for state, action, reward, next_state in batch:
            self.update_q_value(state, action, reward, next_state)
    
    def get_state(self, product_name, market_observations):
        """Create state representation from market observations"""
        product = self.products[product_name]
        own_price = product.price
        own_demand = product.quantity_history[-1] if product.quantity_history else 0
        
        # Find competing products
        category = product.category_cluster
        competitor_price = 0
        competitor_demand = 0
        
        # Get data about competitors in the same category
        if market_observations and 'category_products' in market_observations:
            if category in market_observations['category_products']:
                for market_product in market_observations['category_products'][category]:
                    # Skip our own products
                    if market_product['agent_id'] != self.agent_id:
                        competitor_price = market_product['price']
                        competitor_demand = market_product['last_demand']
                        # Just use the first competitor found for simplicity
                        # Could be extended to average multiple competitors
                        break
        
        # Calculate revenue trend (comparing to previous weeks)
        if len(self.revenue_history) >= 2:
            revenue_trend = (self.revenue_history[-1] - self.revenue_history[-2]) / max(1.0, self.revenue_history[-2])
        else:
            revenue_trend = 0
            
        # Create the state
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
        """Take action based on market observations using RL"""
        # Calculate reward from last action (total revenue change)
        total_revenue = sum(self.revenue_history[-1:] or [0])
        reward = total_revenue - self.last_revenue
        self.last_revenue = total_revenue
        
        # Process each product
        for product_name, product in self.products.items():
            # Get current state
            current_state = self.get_state(product_name, market_observations)
            
            # Learn from previous action if we have one
            if product_name in self.last_state and product_name in self.last_actions:
                self.remember(
                    self.last_state[product_name], 
                    self.last_actions[product_name],
                    reward,
                    current_state
                )
            
            # Store current state
            self.last_state[product_name] = current_state
            
            # Choose action based on current state
            action_idx = self.choose_action(current_state)
            self.last_actions[product_name] = action_idx
            
            # Apply the chosen price change
            price_change = self.price_change_options[action_idx]
            new_price = product.price * (1 + price_change)
            
            # Apply action (set new price)
            self.set_price(product_name, new_price)
        
        # Learn from batch of previous experiences
        self.replay(32)
        
        # Decay exploration rate
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay
    
    def set_price(self, product_name, new_price):
        if product_name in self.products:
            # Ensure price doesn't go below cost + minimum margin
            min_price = self.products[product_name].cost * 1.05  # 5% minimum margin
            # print(f"Setting price for {product_name}: target={new_price}, min={min_price}, final={max(min_price, new_price)}")
            new_price = max(min_price, new_price)
            self.products[product_name].price = new_price
            return True
        return False