import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses
from collections import deque
import random
import sys, os
from env.pricing_agent import PricingAgent
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode

class ActorNetwork(tf.keras.Model):
    """Actor (Policy) Network for MADDPG"""
    def __init__(self, state_dim, action_dim, name='actor'):
        super(ActorNetwork, self).__init__(name=name)
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(64, activation='relu')
        # Final layer with tanh activation for bounded actions [-1,1]
        self.out = layers.Dense(action_dim, activation='tanh')
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out(x)

class CriticNetwork(tf.keras.Model):
    """Centralized Critic for MADDPG"""
    def __init__(self, state_dim, action_dim, num_agents=4, name='critic'):
        super(CriticNetwork, self).__init__(name=name)
        self.state_fc = layers.Dense(128, activation='relu')
        self.action_fc = layers.Dense(64, activation='relu')
        self.concat = layers.Concatenate()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(1)
        
    def call(self, all_states, all_actions):
        # all_states shape: [batch_size, num_agents * state_dim]
        # all_actions shape: [batch_size, num_agents * action_dim]
        s = self.state_fc(all_states)
        a = self.action_fc(all_actions)
        x = self.concat([s, a])
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

class ReplayBuffer:
    """Experience replay buffer with recency bias"""
    def __init__(self, max_size=10000):  # Smaller buffer size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # Add recency bias by sampling more from recent experiences
        buffer_size = len(self.buffer)
        if (buffer_size <= batch_size):
            return map(np.array, zip(*self.buffer))
            
        # More weight to recent experiences
        recent_count = min(batch_size // 2, buffer_size // 4)
        recent_samples = list(self.buffer)[-recent_count:]
        
        # Rest randomly from entire buffer
        old_samples = random.sample(list(self.buffer), batch_size - recent_count)
        
        # Combine samples
        combined_samples = recent_samples + old_samples
        states, actions, rewards, next_states, dones = map(np.array, zip(*combined_samples))
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)

class MADDPGAgent(PricingAgent):
    def __init__(self, agent_id, products, state_dim=10, action_dim=1, 
                 actor_lr=0.001, critic_lr=0.002, discount_factor=0.95,
                 exploration_noise=0.3, noise_decay=0.9995, min_noise=0.05,
                 batch_size=64, tau=0.01):
        super().__init__(agent_id, products)
        
        # Initialize revenue history
        self.revenue_history = []
        self.profit_history = []
        
        # State and action dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Learning parameters
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.exploration_noise = exploration_noise
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self.batch_size = batch_size
        self.tau = tau  # Target network update rate
        
        # Action bounds for continuous pricing (percentage change limits)
        self.max_action = 0.25  # +25% price change
        self.min_action = -0.25  # -25% price change
        
        # Create the actor and critic networks
        self._build_networks()
        
        # Initialize optimizers
        self.actor_optimizer = optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=critic_lr)
        
        # Experience replay buffer
        self.buffer = ReplayBuffer()
        
        # Current and previous states, actions, rewards
        self.last_state = {}
        self.last_action = {}
        self.last_revenue = 0
        
        # Ensure TensorFlow operations are initialized
        self._initialize_networks()
        
        # Debug tracking
        self.debug_mode = False
        self.actor_losses = []
        self.critic_losses = []
        self.actions_before_noise = []
        self.actions_after_noise = []
    
    def _build_networks(self):
        """Build the actor and critic networks (main and target)"""
        # Main networks
        self.actor = ActorNetwork(self.state_dim, self.action_dim)
        self.critic = CriticNetwork(self.state_dim, self.action_dim)
        
        # Target networks (for stable learning)
        self.target_actor = ActorNetwork(self.state_dim, self.action_dim, name='target_actor')
        self.target_critic = CriticNetwork(self.state_dim, self.action_dim, name='target_critic')
        
        # Initialize target networks with the same weights
        self.update_target_networks(tau=1.0)  # Hard copy
    
    def _initialize_networks(self):
        """Ensure all networks are properly initialized by passing dummy data"""
        dummy_state = np.zeros((1, self.state_dim), dtype=np.float32)
        dummy_action = np.zeros((1, self.action_dim), dtype=np.float32)
        
        # Initialize forward pass
        self.actor(dummy_state)
        self.critic(dummy_state, dummy_action)
        self.target_actor(dummy_state)
        self.target_critic(dummy_state, dummy_action)
    
    def update_target_networks(self, tau=None):
        tau = tau if tau is not None else self.tau
        
        # Update target actor
        target_weights = []
        for tw, w in zip(self.target_actor.get_weights(), self.actor.get_weights()):
            target_weights.append(tau * w + (1 - tau) * tw)
        self.target_actor.set_weights(target_weights)
        
        # Update target critic (this part is already correct)
        target_weights = []
        for tw, w in zip(self.target_critic.get_weights(), self.critic.get_weights()):
            target_weights.append(tau * w + (1 - tau) * tw)
        self.target_critic.set_weights(target_weights)
    
    def get_state_representation(self, product_name, market_observations):
        """Create normalized state vector from market observations"""
        product = self.products[product_name]
        own_price = product.price
        own_demand = product.quantity_history[-1] if product.quantity_history else 0
        
        # Find competing products
        category = product.category_cluster
        competitor_price = 0
        competitor_demand = 0
        
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
            market_share,       # New: market share component
        ], dtype=np.float32)
        
        # Debug state dimensions
        # print(f"State shape: {state.shape}, expected: {self.state_dim}")
        assert len(state) == self.state_dim, f"State dimension mismatch: {len(state)} vs {self.state_dim}"
        
        return state
    
    def act(self, week, year, is_holiday, market_observations=None):
        # Store last market observations for reference
        self.last_market_observations = market_observations if market_observations else {}
        
        # Extract state from market observations if available
        if market_observations:
            # Create state representation
            # FIXING THE METHOD NAME HERE - was create_state_representation
            state = self.get_state_representation(list(self.products.keys())[0], market_observations)
            
            # Get action from neural network
            action = self.get_action(state)
            
            # Apply action to product pricing
            for product_name, product in self.products.items():
                self.set_price(product_name, self.scale_action_to_price(action[0], product))
            
            # Store experience in replay buffer if we have previous state
            if hasattr(self, 'previous_state') and self.previous_state is not None:
                # Calculate reward as revenue increase
                previous_revenue = sum(self.revenue_history[-2:-1]) if len(self.revenue_history) > 1 else 0
                current_revenue = self.revenue_history[-1] if self.revenue_history else 0
                reward = current_revenue - previous_revenue
                
                # Store experience: (state, action, reward, next_state, done)
                self.buffer.add(self.previous_state, self.previous_action, reward, state, False)
                
                # Now learn from experiences
                self.learn_from_experiences()
            
            # Update previous state/action for next step
            self.previous_state = state
            self.previous_action = action
        else:
            # Default behavior if no market observations
            for product_name, product in self.products.items():
                current_price = product.price
                # Default pricing logic
    
    def get_action(self, state):
        """Get action from actor network and add exploration noise"""
        # Reshape state to batch format
        state_batch = np.expand_dims(state, axis=0).astype(np.float32)
        
        # Get deterministic action from policy
        action = self.actor(state_batch).numpy()[0]
        
        # Track actions before noise if in debug mode
        if self.debug_mode:
            self.actions_before_noise.append(float(action[0]))
        
        # Add exploration noise
        noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
        noisy_action = action + noise
        
        # Clip action to [-1, 1]
        clipped_action = np.clip(noisy_action, -1.0, 1.0)
        
        # Track actions after noise if in debug mode
        if self.debug_mode:
            self.actions_after_noise.append(float(clipped_action[0]))
        
        return clipped_action

    def learn_from_experiences(self):
        """Learn from batch of experiences using MADDPG algorithm"""
        # Only learn if we have enough samples
        if self.buffer.size() < self.batch_size:
            return
        
        try:
            # Sample batch of experiences
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            
            # Convert to tensors
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            # Since we don't have access to other agents' states/actions in this implementation,
            # We'll use a simplified version that works with the local observations
            
            # Update critic network
            with tf.GradientTape() as tape:
                # Get target actions from target actor network
                target_actions = self.target_actor(next_states)
                
                # Get target Q value using target networks
                target_q_values = self.target_critic(next_states, target_actions)
                
                # Compute TD targets
                targets = rewards + self.discount_factor * (1 - dones) * target_q_values
                
                # Compute current Q values
                q_values = self.critic(states, actions)
                
                # Mean squared TD error
                critic_loss = tf.reduce_mean(tf.square(targets - q_values))
            
            # Update critic
            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
            
            # Update actor using policy gradient
            with tf.GradientTape() as tape:
                # Get actions from current actor network
                predicted_actions = self.actor(states)
                
                # Calculate loss (maximize Q value)
                actor_loss = -tf.reduce_mean(self.critic(states, predicted_actions))
            
            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
            
            # Update target networks
            self.update_target_networks()
            
            # Record losses if in debug mode
            if self.debug_mode:
                self.actor_losses.append(float(actor_loss))
                self.critic_losses.append(float(critic_loss))
                
                # Print learning progress periodically
                if len(self.actor_losses) % 10 == 0:  # Every 10 updates
                    print(f"\nTraining step {len(self.actor_losses)}:")
                    print(f"  Actor Loss: {float(actor_loss):.6f}")
                    print(f"  Critic Loss: {float(critic_loss):.6f}")
                    print(f"  Current exploration noise: {self.exploration_noise:.3f}")
                    if len(self.actions_before_noise) > 0 and len(self.actions_after_noise) > 0:
                        print(f"  Recent action (before noise): {self.actions_before_noise[-1]:.3f}")
                        print(f"  Recent action (after noise): {self.actions_after_noise[-1]:.3f}")
        
        except Exception as e:
            print(f"ERROR in learn_from_experiences(): {e}")
            import traceback
            traceback.print_exc()
    
    def set_price(self, product_name, new_price):
        if (product_name in self.products):
            current_price = self.products[product_name].price
            
            # Apply temporal smoothing - only move partially toward target price
            # Fix: Use a fixed smoothing factor since episode isn't tracked in this class
            smoothing_factor = 0.3  # Fixed smoothing factor (30% of the way to target price)
            
            # Alternative: Get episode from market observations if available
            # if hasattr(self, 'last_market_observations') and 'episode' in self.last_market_observations:
            #     episode = self.last_market_observations.get('episode', 0)
            #     smoothing_factor = max(0.1, 0.3 * np.exp(-episode / 20))
            
            smoothed_price = current_price + smoothing_factor * (new_price - current_price)
            
            # Ensure price doesn't go below cost + minimum margin
            min_price = self.products[product_name].cost * 1.05  # 5% minimum margin
            final_price = max(min_price, smoothed_price)
            
            self.products[product_name].price = final_price
            self.products[product_name].price_history.append(final_price)
            return True
        return False
    
    def scale_action_to_price(self, action_value, product):
        """
        Convert a continuous action value [-1, 1] to a price change,
        then apply it to get the new price.
        """
        # Convert action from [-1, 1] to price change percentage
        # Mapping: -1 -> -25%, 0 -> 0%, 1 -> +25%
        price_change_pct = action_value * 0.25  # -0.25 to +0.25
        
        # Get current price
        current_price = product.price
        
        # Calculate new price
        new_price = current_price * (1 + price_change_pct)
        
        # Ensure price stays within reasonable bounds (e.g., at least 5% above cost)
        min_price = product.cost * 1.05
        max_price = product.cost * 3.0  # Maximum 200% markup as a safeguard
        
        # Clip to range
        new_price = max(min_price, min(new_price, max_price))
        
        return new_price
    
    def save(self, path='maddpg_models'):
        """Save the trained models"""
        os.makedirs(path, exist_ok=True)
        # Fix: Add .weights.h5 suffix as required by Keras
        self.actor.save_weights(f"{path}/actor_{self.agent_id}.weights.h5")
        self.critic.save_weights(f"{path}/critic_{self.agent_id}.weights.h5")
        
    def load(self, path='maddpg_models'):
        """Load trained models"""
        try:
            # Fix: Update load paths to match the new format
            self.actor.load_weights(f"{path}/actor_{self.agent_id}.weights.h5")
            self.critic.load_weights(f"{path}/critic_{self.agent_id}.weights.h5")
            self.update_target_networks(tau=1.0)  # Hard update
            # print(f"Successfully loaded models for {self.agent_id}")
        except:
            print(f"Failed to load models for {self.agent_id}, using untrained networks")