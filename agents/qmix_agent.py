import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
import os
from env.pricing_agent import PricingAgent

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ReplayBuffer:
    """Experience replay buffer with recency bias"""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state_dict, action_dict, reward, next_state_dict, done):
        self.buffer.append((state_dict, action_dict, reward, next_state_dict, done))
    
    def sample(self, batch_size):
        # Add recency bias by sampling more from recent experiences
        buffer_size = len(self.buffer)
        if buffer_size <= batch_size:
            return list(self.buffer)
            
        # More weight to recent experiences
        recent_count = min(batch_size // 2, buffer_size // 4)
        recent_samples = list(self.buffer)[-recent_count:]
        
        # Rest randomly from entire buffer
        old_samples = random.sample(list(self.buffer), batch_size - recent_count)
        
        # Combine samples
        combined_samples = recent_samples + old_samples
        return combined_samples
    
    def size(self):
        return len(self.buffer)

class AgentNetwork(tf.keras.Model):
    """Individual agent DQN network"""
    def __init__(self, state_dim, action_dim, name='agent_net'):
        super(AgentNetwork, self).__init__(name=name)
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(32, activation='relu')
        self.out = layers.Dense(action_dim)  # Q-values for each action
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.out(x)

class MixingNetwork(tf.keras.Model):
    """QMIX mixing network that combines individual Q-values"""
    def __init__(self, num_agents, name='mixing_net'):
        super(MixingNetwork, self).__init__(name=name)
        self.num_agents = num_agents
        
        # Hypernetwork layers
        self.hyper_w1 = layers.Dense(32, activation='relu')
        self.hyper_w2 = layers.Dense(32, activation='relu')  # Changed from num_agents to fixed size
        self.hyper_b1 = layers.Dense(32)
        
    def call(self, agent_qs, state):
        """Generate weights from global state using hypernetwork"""
        batch_size = tf.shape(agent_qs)[0]
        num_agents = tf.shape(agent_qs)[1]
        
        # Generate weights from global state
        w1 = tf.abs(self.hyper_w1(state))  # Ensure positive weights
        b1 = self.hyper_b1(state)
        w2 = tf.abs(self.hyper_w2(state))  # Ensure positive weights
        
        # First layer - reshape w1 properly for matrix multiplication
        w1_reshaped = tf.reshape(w1, [batch_size, 32])  # Shape: [batch_size, 32]
        
        # Create a matrix of appropriate shape
        w1_matrix = tf.zeros([batch_size, num_agents, 32])
        # Fill the matrix with reshaped w1 values, dividing evenly across agents
        w1_matrix = tf.fill([batch_size, num_agents, 32], 1.0) * tf.reshape(w1_reshaped, [batch_size, 1, 32]) / tf.cast(num_agents, tf.float32)
        
        # Reshape agent_qs for batch matrix multiplication
        agent_qs_reshaped = tf.reshape(agent_qs, [batch_size, num_agents, 1])  # Shape: [batch_size, num_agents, 1]
        
        # Batch matrix multiplication for first layer
        # Each agent's Q-value multiplied by its portion of w1
        hidden = tf.matmul(agent_qs_reshaped, w1_matrix, transpose_a=True)  # [batch_size, 1, 32]
        hidden = tf.reshape(hidden, [batch_size, 32])  # Shape: [batch_size, 32]
        
        # Add bias and apply ReLU
        hidden = tf.nn.relu(hidden + b1)  # Shape: [batch_size, 32]
        
        # Second layer
        w2_reshaped = tf.reshape(w2, [batch_size, 32])  # Shape: [batch_size, 32]
        
        # Final output
        q_total = tf.reduce_sum(hidden * w2_reshaped, axis=1, keepdims=True)  # Shape: [batch_size, 1]
        
        return q_total

class QMIXAgent(PricingAgent):
    def __init__(self, agent_id, products, state_dim=10, 
                 learning_rate=0.001, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.05,
                 batch_size=64, update_target_every=5, num_agents=4):
        super().__init__(agent_id, products)
        
        # Initialize history
        self.revenue_history = []
        self.profit_history = []
        
        # State dimensions and price change options
        self.state_dim = state_dim
        self.price_change_options = np.array([-0.10, -0.07, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.07, 0.10])
        self.action_dim = len(self.price_change_options)
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.update_counter = 0
        self.num_agents = num_agents
        
        # Store product networks by name
        self.product_networks = {}
        self.target_product_networks = {}
        
        # Initialize product networks
        for product_name in self.products.keys():
            self.product_networks[product_name] = AgentNetwork(state_dim, self.action_dim, name=f'agent_net_{product_name}')
            self.target_product_networks[product_name] = AgentNetwork(state_dim, self.action_dim, name=f'target_agent_net_{product_name}')
            
            # Initialize networks with dummy data
            dummy_state = np.zeros((1, self.state_dim), dtype=np.float32)
            self.product_networks[product_name](dummy_state)
            self.target_product_networks[product_name](dummy_state)
            
            # Copy weights to target network
            self.target_product_networks[product_name].set_weights(
                self.product_networks[product_name].get_weights())
        
        # Create mixing networks
        self.mixer = MixingNetwork(len(self.products))
        self.target_mixer = MixingNetwork(len(self.products))
        
        # Initialize mixer with dummy data
        num_products = len(self.products)
        dummy_qs = np.zeros((1, num_products), dtype=np.float32)
        # Make sure global state has the right size
        dummy_state = np.zeros((1, self.state_dim * num_products), dtype=np.float32)

        # Initialize both networks with dummy forward passes
        self.mixer(dummy_qs, dummy_state)
        self.target_mixer(dummy_qs, dummy_state)
        
        # Copy weights to target mixer
        self.target_mixer.set_weights(self.mixer.get_weights())
        
        # Initialize optimizer
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        # Experience replay buffer
        self.buffer = ReplayBuffer()
        
        # Current and previous states, actions, rewards
        self.last_state = {}
        self.last_action = {}
        self.last_revenue = 0
        
        # Coordinator ID - only the coordinator agent will perform learning updates
        self.is_coordinator = (agent_id == "Agent1")  # Assuming Agent1 is the coordinator
        
        # Shared buffer for coordination (in a real system, this would be a database or message queue)
        self.shared_experiences = []
        
    def update_target_networks(self, tau=1.0):
        """Update target network weights"""
        # Update product networks
        for product_name in self.products.keys():
            target_weights = []
            source_weights = self.product_networks[product_name].get_weights()
            target_w = self.target_product_networks[product_name].get_weights()
            
            for tw, w in zip(target_w, source_weights):
                target_weights.append(tau * w + (1 - tau) * tw)
            
            self.target_product_networks[product_name].set_weights(target_weights)
        
        # Update mixer network
        mixer_target_weights = []
        for tw, w in zip(self.target_mixer.get_weights(), self.mixer.get_weights()):
            mixer_target_weights.append(tau * w + (1 - tau) * tw)
        self.target_mixer.set_weights(mixer_target_weights)
    
    def get_state_representation(self, product_name, market_observations):
        """Create normalized state vector from market observations"""
        product = self.products[product_name]
        own_price = product.price
        own_demand = product.quantity_history[-1] if product.quantity_history else 0
        
        # Find competing products
        category = product.category_cluster
        
        # Get data about ALL competitors in the same category
        competitor_prices = []
        competitor_demands = []
        if market_observations and 'category_products' in market_observations:
            if category in market_observations['category_products']:
                for market_product in market_observations['category_products'][category]:
                    # Skip our own products
                    if market_product['agent_id'] != self.agent_id:
                        competitor_prices.append(market_product['price'])
                        competitor_demands.append(market_product['last_demand'])
        
        # Use average competitor price and demand
        avg_competitor_price = np.mean(competitor_prices) if competitor_prices else own_price
        avg_competitor_demand = np.mean(competitor_demands) if competitor_demands else 0
        
        # Calculate price ratio compared to market average
        price_ratio = own_price / max(0.1, avg_competitor_price)
        demand_ratio = own_demand / max(0.1, avg_competitor_demand) if avg_competitor_demand > 0 else 1.0
        
        # Calculate revenue trend
        if len(self.revenue_history) >= 2:
            revenue_trend = (self.revenue_history[-1] - self.revenue_history[-2]) / max(1.0, self.revenue_history[-2])
        else:
            revenue_trend = 0
        
        # Price history features
        price_trends = []
        if len(product.price_history) >= 3:
            # Last 2 price changes as percentages
            price_change_1 = product.price_history[-1] / product.price_history[-2] - 1.0 if product.price_history[-2] > 0 else 0
            price_change_2 = product.price_history[-2] / product.price_history[-3] - 1.0 if product.price_history[-3] > 0 else 0
            price_trends = [price_change_1, price_change_2]
        else:
            price_trends = [0, 0]  # Default if not enough history
        
        # Demand history features
        demand_trends = []
        if len(product.quantity_history) >= 3:
            # Last 2 demand changes as percentages
            demand_change_1 = (product.quantity_history[-1] / max(0.1, product.quantity_history[-2]) - 1.0) if product.quantity_history[-2] > 0 else 0
            demand_change_2 = (product.quantity_history[-2] / max(0.1, product.quantity_history[-3]) - 1.0) if product.quantity_history[-3] > 0 else 0
            demand_trends = [demand_change_1, demand_change_2]
        else:
            demand_trends = [0, 0]  # Default if not enough history
        
        # Normalize week number to [0,1]
        week = market_observations.get('week', 1) / 52.0  # Assuming 52 weeks per year
        
        # Calculate market share for this product
        total_category_demand = own_demand
        for demand in competitor_demands:
            total_category_demand += demand
        
        market_share = own_demand / max(1.0, total_category_demand)
        
        # Create the state vector
        state = np.array([
            price_ratio - 1.0,  # Normalized around 0
            demand_ratio - 1.0,  # Normalized around 0
            revenue_trend,      # Already normalized around 0
            week,               # Normalized to [0,1]
            int(market_observations.get('is_holiday', False)),
            price_trends[0],    # Already normalized
            price_trends[1],    # Already normalized
            demand_trends[0],   # Already normalized
            demand_trends[1],   # Already normalized
            market_share,       # Market share component
        ], dtype=np.float32)
        
        assert len(state) == self.state_dim, f"State dimension mismatch: {len(state)} vs {self.state_dim}"
        
        return state
    
    def get_global_state(self, product_states):
        """Combine individual product states into a global state for the mixer"""
        # Flatten all product states into one vector
        states = []
        for product_name in sorted(product_states.keys()):  # Sort to ensure consistent order
            states.append(product_states[product_name])
        
        if not states:
            return np.zeros((1, self.state_dim * len(self.products)), dtype=np.float32)
        
        # Concatenate and ensure correct shape
        global_state = np.concatenate(states)
        expected_size = self.state_dim * len(self.products)
        
        # Make sure we have the right size
        if len(global_state) < expected_size:
            # Pad with zeros if needed
            padding = np.zeros(expected_size - len(global_state), dtype=np.float32)
            global_state = np.concatenate([global_state, padding])
        elif len(global_state) > expected_size:
            # Truncate if too large
            global_state = global_state[:expected_size]
        
        return global_state
        
    def choose_action(self, state, product_name):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.exploration_rate:
            # Exploration: choose random action
            return np.random.randint(0, self.action_dim)
        else:
            # Exploitation: choose best action from Q-values
            state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
            q_values = self.product_networks[product_name](state_tensor).numpy()[0]
            return np.argmax(q_values)
        
    def act(self, week, year, is_holiday, market_observations=None):
        """Take action based on market observations using QMIX"""
        try:
            # Calculate base reward from revenue change
            total_revenue = sum(self.revenue_history[-1:] or [0])
            base_reward = total_revenue - self.last_revenue
            
            # Add price stability component to reward
            stability_reward = 0
            for product_name, product in self.products.items():
                if len(product.price_history) >= 2:
                    # Calculate percentage price change
                    last_price = product.price_history[-1]
                    prev_price = product.price_history[-2]
                    price_change_pct = abs(last_price / prev_price - 1.0)
                    
                    # Quadratically penalize price changes (more penalty for larger changes)
                    stability_reward -= min(2.0, price_change_pct * price_change_pct * 20)
            
            # Combined reward 
            reward = base_reward + stability_reward
            self.last_revenue = total_revenue
            
            # Get current state for each product
            current_states = {}
            for product_name, product in self.products.items():
                current_states[product_name] = self.get_state_representation(product_name, market_observations)
            
            # Learn from previous action if we have one
            if self.last_state and self.last_action:
                # Add experience to replay buffer
                self.buffer.add(
                    self.last_state,
                    self.last_action,
                    reward,
                    current_states,
                    False  # Not terminal state
                )
                
                # Share experience with coordinator if not the coordinator
                if not self.is_coordinator:
                    # In a real implementation, this would be a network or message queue call
                    # For simulation purposes, we just store it (the coordinator would access this)
                    self.shared_experiences.append((self.last_state, self.last_action, reward, current_states, False))
            
            # Store current state
            self.last_state = current_states
            
            # Choose action for each product
            actions = {}
            for product_name, state in current_states.items():
                action_idx = self.choose_action(state, product_name)
                actions[product_name] = action_idx
                
                # Apply the chosen price change
                price_change = self.price_change_options[action_idx]
                new_price = self.products[product_name].price * (1 + price_change)
                
                # Apply action (set new price)
                self.set_price(product_name, new_price)
            
            # Store actions
            self.last_action = actions
            
            # Only the coordinator performs learning
            if self.is_coordinator:
                self.learn_from_experiences()
            
            # Decay exploration rate
            if self.exploration_rate > self.min_exploration:
                self.exploration_rate *= self.exploration_decay
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            
    def set_price(self, product_name, new_price):
        """Set price with temporal smoothing"""
        if product_name in self.products:
            current_price = self.products[product_name].price
            
            # Apply temporal smoothing - only move partially toward target price
            smoothing_factor = 0.3  # Only move 30% toward target price
            smoothed_price = current_price + smoothing_factor * (new_price - current_price)
            
            # Ensure price doesn't go below cost + minimum margin
            min_price = self.products[product_name].cost * 1.05  # 5% minimum margin
            final_price = max(min_price, smoothed_price)
            
            self.products[product_name].price = final_price
            self.products[product_name].price_history.append(final_price)
            return True
        return False
            
    def learn_from_experiences(self):
        """Perform QMIX learning using experience replay"""
        # Only learn if we have enough samples
        if self.buffer.size() < self.batch_size:
            return
        
        try:
            # Sample batch of experiences
            batch = self.buffer.sample(self.batch_size)
            
            # Calculate joint loss across all experiences
            with tf.GradientTape() as tape:
                total_loss = 0
                
                for state_dict, action_dict, reward, next_state_dict, done in batch:
                    # Convert states to list to maintain order
                    product_names = list(state_dict.keys())
                    
                    # Get current Q values for each product
                    current_qs = []
                    for product_name in product_names:
                        state = state_dict[product_name]
                        action = action_dict[product_name]
                        
                        # Get Q value for taken action
                        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
                        q_values = self.product_networks[product_name](state_tensor)[0]
                        q_value = q_values[action]
                        current_qs.append(q_value)
                    
                    current_qs_tensor = tf.stack(current_qs)
                    
                    # Get global state for mixer
                    global_state = self.get_global_state(state_dict)
                    global_state_tensor = tf.convert_to_tensor(np.expand_dims(global_state, axis=0), dtype=np.float32)
                    
                    # Mix Q values
                    current_q_total = self.mixer(tf.expand_dims(current_qs_tensor, axis=0), global_state_tensor)
                    
                    # Get max Q values for next state
                    next_max_qs = []
                    for product_name in product_names:
                        next_state = next_state_dict[product_name]
                        next_state_tensor = tf.convert_to_tensor(np.expand_dims(next_state, axis=0), dtype=tf.float32)
                        next_q_values = self.target_product_networks[product_name](next_state_tensor)[0]
                        next_max_q = tf.reduce_max(next_q_values)
                        next_max_qs.append(next_max_q)
                    
                    next_max_qs_tensor = tf.stack(next_max_qs)
                    
                    # Get global next state for target mixer
                    next_global_state = self.get_global_state(next_state_dict)
                    next_global_state_tensor = tf.convert_to_tensor(np.expand_dims(next_global_state, axis=0), dtype=tf.float32)
                    
                    # Mix next Q values
                    next_q_total = self.target_mixer(tf.expand_dims(next_max_qs_tensor, axis=0), next_global_state_tensor)
                    
                    # Calculate target (Bellman equation)
                    target = reward + (1 - float(done)) * self.discount_factor * next_q_total
                    
                    # Calculate loss for this experience
                    loss = tf.square(target - current_q_total)
                    total_loss += loss
                
                # Average loss across batch
                avg_loss = total_loss / self.batch_size
            
            # Get all trainable variables
            trainable_vars = []
            for product_name in self.products.keys():
                trainable_vars.extend(self.product_networks[product_name].trainable_variables)
            trainable_vars.extend(self.mixer.trainable_variables)
            
            # Compute gradients and update networks
            grads = tape.gradient(avg_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(grads, trainable_vars))
            
            # Periodically update target networks
            self.update_counter += 1
            if self.update_counter % self.update_target_every == 0:
                self.update_target_networks(tau=0.1)  # Soft update
                
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def save(self, path='qmix_models'):
        """Save the trained model"""
        os.makedirs(path, exist_ok=True)
        
        # Save individual product networks
        for product_name in self.products.keys():
            self.product_networks[product_name].save_weights(
                f"{path}/{self.agent_id}_{product_name}_net.weights.h5")
        
        # Save mixer network
        self.mixer.save_weights(f"{path}/{self.agent_id}_mixer.weights.h5")
        
    def load(self, path='qmix_models'):
        """Load trained model"""
        try:
            # Load individual product networks
            for product_name in self.products.keys():
                self.product_networks[product_name].load_weights(
                    f"{path}/{self.agent_id}_{product_name}_net.weights.h5")
            
            # Load mixer network
            self.mixer.load_weights(f"{path}/{self.agent_id}_mixer.weights.h5")
            
            # Update target networks
            self.update_target_networks(tau=1.0)  # Hard update
            print(f"Successfully loaded models for {self.agent_id}")
        except Exception as e:
            print(f"Failed to load model for {self.agent_id}, using untrained network: {str(e)}")