"""
MADQNAgent:

Classes:
- ReplayBuffer: Experience memory with prioritized recent experience sampling
- DQNetwork: Neural network for Q-value estimation
- MADQNAgent: Main agent class that inherits from PricingAgent

MADQNAgent:
    - get_state_representation(): Converts market data into state vectors
    - choose_action(): Selects actions using epsilon-greedy policy
    - act(): Determines pricing actions based on market observations
    - set_price(): Applies smoothed price changes to products
    - learn_from_experiences(): Updates network using DQN algorithm
    - save()/load(): Persistence methods for trained models
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
import os
from env.pricing_agent import PricingAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        if buffer_size <= batch_size:
            return map(np.array, zip(*self.buffer))
            
        recent_count = min(batch_size // 2, buffer_size // 4)
        recent_samples = list(self.buffer)[-recent_count:]
        
        old_samples = random.sample(list(self.buffer), batch_size - recent_count)
        
        combined_samples = recent_samples + old_samples
        return map(np.array, zip(*combined_samples))
    
    def size(self):
        return len(self.buffer)

class DQNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, name='dqn'):
        super(DQNetwork, self).__init__(name=name)
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(32, activation='relu')
        self.out = layers.Dense(action_dim)
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out(x)

class MADQNAgent(PricingAgent):
    def __init__(self, agent_id, products, state_dim=10, 
                 learning_rate=0.001, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.05,
                 batch_size=64, update_target_every=5):
        super().__init__(agent_id, products)
        
        self.revenue_history = []
        self.profit_history = []
        
        self.state_dim = state_dim
        self.price_change_options = np.array([-0.10, -0.07, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.07, 0.10])
        self.action_dim = len(self.price_change_options)
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.update_counter = 0
        
        self.q_network = DQNetwork(state_dim, self.action_dim)
        self.target_q_network = DQNetwork(state_dim, self.action_dim, name='target_dqn')
        
        self._initialize_networks()
        self.update_target_network(tau=1.0)
        
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        self.buffer = ReplayBuffer()
        
        self.last_state = {}
        self.last_action = {}
        self.last_revenue = 0
        
    def _initialize_networks(self):
        dummy_state = np.zeros((1, self.state_dim), dtype=np.float32)
        
        self.q_network(dummy_state)
        self.target_q_network(dummy_state)
        
    def update_target_network(self, tau=1.0):
        target_weights = []
        for tw, w in zip(self.target_q_network.get_weights(), self.q_network.get_weights()):
            target_weights.append(tau * w + (1 - tau) * tw)
        self.target_q_network.set_weights(target_weights)
        
    def get_state_representation(self, product_name, market_observations):
        product = self.products[product_name]
        own_price = product.price
        own_demand = product.quantity_history[-1] if product.quantity_history else 0
        
        category = product.category_cluster
        
        competitor_prices = []
        competitor_demands = []
        if market_observations and 'category_products' in market_observations:
            if category in market_observations['category_products']:
                for market_product in market_observations['category_products'][category]:
                    if market_product['agent_id'] != self.agent_id:
                        competitor_prices.append(market_product['price'])
                        competitor_demands.append(market_product['last_demand'])
        
        avg_competitor_price = np.mean(competitor_prices) if competitor_prices else own_price
        avg_competitor_demand = np.mean(competitor_demands) if competitor_demands else 0
        
        price_ratio = own_price / max(0.1, avg_competitor_price)
        demand_ratio = own_demand / max(0.1, avg_competitor_demand) if avg_competitor_demand > 0 else 1.0
        
        if len(self.revenue_history) >= 2:
            revenue_trend = (self.revenue_history[-1] - self.revenue_history[-2]) / max(1.0, self.revenue_history[-2])
        else:
            revenue_trend = 0
        
        price_trends = []
        if len(product.price_history) >= 3:
            price_change_1 = product.price_history[-1] / product.price_history[-2] - 1.0 if product.price_history[-2] > 0 else 0
            price_change_2 = product.price_history[-2] / product.price_history[-3] - 1.0 if product.price_history[-3] > 0 else 0
            price_trends = [price_change_1, price_change_2]
        else:
            price_trends = [0, 0]
        
        demand_trends = []
        if len(product.quantity_history) >= 3:
            demand_change_1 = (product.quantity_history[-1] / max(0.1, product.quantity_history[-2]) - 1.0) if product.quantity_history[-2] > 0 else 0
            demand_change_2 = (product.quantity_history[-2] / max(0.1, product.quantity_history[-3]) - 1.0) if product.quantity_history[-3] > 0 else 0
            demand_trends = [demand_change_1, demand_change_2]
        else:
            demand_trends = [0, 0]
        
        week = market_observations.get('week', 1) / 52.0
        
        total_category_demand = own_demand
        for demand in competitor_demands:
            total_category_demand += demand
        
        market_share = own_demand / max(1.0, total_category_demand)
        
        state = np.array([
            price_ratio - 1.0,
            demand_ratio - 1.0,
            revenue_trend,
            week,
            int(market_observations.get('is_holiday', False)),
            price_trends[0],
            price_trends[1],
            demand_trends[0],
            demand_trends[1],
            market_share,
        ], dtype=np.float32)
        
        assert len(state) == self.state_dim, f"State dimension mismatch: {len(state)} vs {self.state_dim}"
        
        return state
        
    def choose_action(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_dim)
        else:
            state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
            q_values = self.q_network(state_tensor).numpy()[0]
            return np.argmax(q_values)
    
    def act(self, week, year, is_holiday, market_observations=None):
        try:
            total_revenue = sum(self.revenue_history[-1:] or [0])
            base_reward = total_revenue - self.last_revenue
            
            stability_reward = 0
            for product_name, product in self.products.items():
                if len(product.price_history) >= 2:
                    last_price = product.price_history[-1]
                    prev_price = product.price_history[-2]
                    price_change_pct = abs(last_price / prev_price - 1.0)
                    
                    stability_reward -= min(2.0, price_change_pct * price_change_pct * 20)
            
            reward = base_reward + stability_reward
            self.last_revenue = total_revenue
            
            for product_name, product in self.products.items():
                current_state = self.get_state_representation(product_name, market_observations)
                
                if product_name in self.last_state and product_name in self.last_action:
                    self.buffer.add(
                        self.last_state[product_name], 
                        self.last_action[product_name],
                        reward,  
                        current_state,
                        False
                    )
                
                self.last_state[product_name] = current_state
                
                action_idx = self.choose_action(current_state)
                self.last_action[product_name] = action_idx
                
                price_change = self.price_change_options[action_idx]
                new_price = product.price * (1 + price_change)
                
                self.set_price(product_name, new_price)
            
            self.learn_from_experiences()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay
            
    def set_price(self, product_name, new_price):
        if product_name in self.products:
            current_price = self.products[product_name].price
            
            smoothing_factor = 0.3
            smoothed_price = current_price + smoothing_factor * (new_price - current_price)
            
            min_price = self.products[product_name].cost * 1.05
            final_price = max(min_price, smoothed_price)
            
            self.products[product_name].price = final_price
            self.products[product_name].price_history.append(final_price)
            return True
        return False
            
    def learn_from_experiences(self):
        if self.buffer.size() < self.batch_size:
            return
        
        try:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            target_q_values = self.target_q_network(next_states)
            max_target_q = tf.reduce_max(target_q_values, axis=1)
            targets = rewards + (1 - dones) * self.discount_factor * max_target_q
            
            with tf.GradientTape() as tape:
                q_values = self.q_network(states)
                
                action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
                predicted_q_values = tf.gather_nd(q_values, action_indices)
                
                loss = tf.reduce_mean(tf.square(targets - predicted_q_values))
            
            gradients = tape.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
            
            self.update_counter += 1
            if self.update_counter % self.update_target_every == 0:
                self.update_target_network(tau=0.1)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def save(self, path='models/madqn_models'):
        os.makedirs(path, exist_ok=True)
        self.q_network.save_weights(f"{path}/qnetwork_{self.agent_id}.weights.h5")
        
    def load(self, path='models/madqn_models'):
        try:
            self.q_network.load_weights(f"{path}/qnetwork_{self.agent_id}.weights.h5")
            self.update_target_network(tau=1.0)
        except:
            print(f"Failed to load model for {self.agent_id}, using untrained