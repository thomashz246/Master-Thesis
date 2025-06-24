import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses
from collections import deque
import random
import sys, os
from env.pricing_agent import PricingAgent
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ActorNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, name='actor'):
        super(ActorNetwork, self).__init__(name=name)
        self.fc1 = layers.Dense(64, activation='relu', 
                           kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.fc2 = layers.Dense(64, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.out = layers.Dense(action_dim, activation='tanh')
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.out(x)

class CriticNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, num_agents=4, name='critic'):
        super(CriticNetwork, self).__init__(name=name)
        self.state_fc = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.action_fc = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.concat = layers.Concatenate()
        self.norm1 = layers.LayerNormalization()
        self.fc1 = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.norm2 = layers.LayerNormalization()
        self.fc2 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.out = layers.Dense(1)
        
    def call(self, all_states, all_actions):
        s = self.state_fc(all_states)
        a = self.action_fc(all_actions)
        x = self.concat([s, a])
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return self.out(x)

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        if (buffer_size <= batch_size):
            return map(np.array, zip(*self.buffer))
            
        recent_count = min(batch_size // 2, buffer_size // 4)
        recent_samples = list(self.buffer)[-recent_count:]
        
        old_samples = random.sample(list(self.buffer), batch_size - recent_count)
        
        combined_samples = recent_samples + old_samples
        states, actions, rewards, next_states, dones = map(np.array, zip(*combined_samples))
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)

class MADDPGAgent(PricingAgent):
    def __init__(self, agent_id, products, state_dim=10, action_dim=1, 
                 actor_lr=0.0001,
                 critic_lr=0.00001,
                 buffer_size=100000,
                 batch_size=64,
                 discount_factor=0.95,
                 tau=0.001,
                 exploration_noise=0.2):
        super().__init__(agent_id, products)
        
        self.revenue_history = []
        self.profit_history = []
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.exploration_noise = exploration_noise
        self.noise_decay = 0.9995
        self.min_noise = 0.05
        self.batch_size = batch_size
        self.tau = tau
        
        self.max_action = 0.25
        self.min_action = -0.25
        
        self._build_networks()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr, clipnorm=1.0)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, clipnorm=1.0)
        
        self.buffer = ReplayBuffer()
        
        self.last_state = {}
        self.last_action = {}
        self.last_revenue = 0
        
        self._initialize_networks()
        
        self.debug_mode = True
        self.actor_losses = []
        self.critic_losses = []
        self.actions_before_noise = []
        self.actions_after_noise = []
    
    def _build_networks(self):
        self.actor = ActorNetwork(self.state_dim, self.action_dim)
        self.critic = CriticNetwork(self.state_dim, self.action_dim)
        
        self.target_actor = ActorNetwork(self.state_dim, self.action_dim, name='target_actor')
        self.target_critic = CriticNetwork(self.state_dim, self.action_dim, name='target_critic')
        
        self.update_target_networks(tau=1.0)
    
    def _initialize_networks(self):
        dummy_state = np.zeros((1, self.state_dim), dtype=np.float32)
        dummy_action = np.zeros((1, self.action_dim), dtype=np.float32)
        
        self.actor(dummy_state)
        self.critic(dummy_state, dummy_action)
        self.target_actor(dummy_state)
        self.target_critic(dummy_state, dummy_action)
    
    def update_target_networks(self, tau=None):
        tau = tau if tau is not None else self.tau
        
        target_weights = []
        for tw, w in zip(self.target_actor.get_weights(), self.actor.get_weights()):
            target_weights.append(tau * w + (1 - tau) * tw)
        self.target_actor.set_weights(target_weights)
        
        target_weights = []
        for tw, w in zip(self.target_critic.get_weights(), self.critic.get_weights()):
            target_weights.append(tau * w + (1 - tau) * tw)
        self.target_critic.set_weights(target_weights)
    
    def get_state_representation(self, product_name, market_observations):
        product = self.products[product_name]
        own_price = product.price
        own_demand = product.quantity_history[-1] if product.quantity_history else 0
        
        category = product.category_cluster
        competitor_price = 0
        competitor_demand = 0
        
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
    
    def act(self, week, year, is_holiday, market_observations=None):
        self.last_market_observations = market_observations if market_observations else {}
        
        if market_observations:
            state = self.get_state_representation(list(self.products.keys())[0], market_observations)
            
            action = self.get_action(state)
            
            for product_name, product in self.products.items():
                self.set_price(product_name, self.scale_action_to_price(action[0], product))
            
            self.previous_state = state
            self.previous_action = action
        else:
            for product_name, product in self.products.items():
                current_price = product.price
    
    def get_action(self, state):
        state_batch = np.expand_dims(state, axis=0).astype(np.float32)
        
        action_deterministic = self.actor(state_batch).numpy()[0]
        
        if self.debug_mode:
            self.actions_before_noise.append(float(action_deterministic[0]))
        
        noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
        noisy_action = action_deterministic + noise
        
        clipped_action = np.clip(noisy_action, -1.0, 1.0)
        
        if self.debug_mode:
            self.actions_after_noise.append(float(clipped_action[0]))
        
        return clipped_action

    def learn_from_experiences(self):
        if self.buffer.size() < self.batch_size:
            return
    
    def learn_from_joint_experience(self, joint_states, joint_actions, own_reward, joint_next_states, dones, all_agents):
        try:
            batch_size = joint_states.shape[0]
            
            joint_states_flat = np.reshape(joint_states, (batch_size, -1))
            joint_actions_flat = np.reshape(joint_actions, (batch_size, -1))
            joint_next_states_flat = np.reshape(joint_next_states, (batch_size, -1))
            
            joint_states_tensor = tf.convert_to_tensor(joint_states_flat, dtype=tf.float32)
            joint_actions_tensor = tf.convert_to_tensor(joint_actions_flat, dtype=tf.float32)
            own_reward_tensor = tf.convert_to_tensor(own_reward, dtype=tf.float32)
            joint_next_states_tensor = tf.convert_to_tensor(joint_next_states_flat, dtype=tf.float32)
            dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            next_actions = []
            for i, target_agent in enumerate(all_agents):
                agent_next_states = joint_next_states[:, i]
                agent_next_action = target_agent.target_actor(tf.convert_to_tensor(agent_next_states, dtype=tf.float32))
                next_actions.append(agent_next_action)
            
            joint_next_actions_flat = tf.concat(next_actions, axis=1)
            
            with tf.GradientTape() as tape:
                target_q = self.target_critic(joint_next_states_tensor, joint_next_actions_flat)
                
                y = own_reward_tensor + self.discount_factor * (1 - dones_tensor) * target_q
                
                q_value = self.critic(joint_states_tensor, joint_actions_tensor)
                critic_loss = tf.reduce_mean(tf.square(y - q_value))
            
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            
            if None in critic_grads or any(tf.reduce_all(tf.math.is_nan(g)) for g in critic_grads if g is not None):
                print("Warning: NaN or None gradients in critic update - skipping update")
                
                print(f"Target Q range: {tf.reduce_min(target_q):.4f} to {tf.reduce_max(target_q):.4f}")
                print(f"Current Q range: {tf.reduce_min(q_value):.4f} to {tf.reduce_max(q_value):.4f}")
            else:
                critic_grads, _ = tf.clip_by_global_norm(critic_grads, 1.0)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            
            with tf.GradientTape() as tape:
                my_idx = all_agents.index(self)
                my_state = joint_states[:, my_idx]
                my_actions = self.actor(tf.convert_to_tensor(my_state, dtype=tf.float32))
                
                actions_updated = joint_actions.copy()
                actions_updated[:, my_idx] = my_actions.numpy()
                actions_updated_flat = np.reshape(actions_updated, (batch_size, -1))
                
                actor_loss = -tf.reduce_mean(
                    self.critic(joint_states_tensor, tf.convert_to_tensor(actions_updated_flat, dtype=tf.float32))
                )
            
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            
            if None in actor_grads or any(tf.reduce_all(tf.math.is_nan(g)) for g in actor_grads if g is not None):
                print("Warning: NaN or None gradients in actor update - skipping update")
            else:
                actor_grads, _ = tf.clip_by_global_norm(actor_grads, 1.0)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            
            self.update_target_networks()
            
            if self.debug_mode:
                self.actor_losses.append(float(actor_loss))
                self.critic_losses.append(float(critic_loss))
        
            self.exploration_noise = max(self.min_noise, self.exploration_noise * self.noise_decay)
        
        except Exception as e:
            print(f"Error in learn_from_joint_experience for {self.agent_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def set_price(self, product_name, new_price):
        if (product_name in self.products):
            current_price = self.products[product_name].price
            
            smoothing_factor = 0.3
            
            smoothed_price = current_price + smoothing_factor * (new_price - current_price)
            
            min_price = self.products[product_name].cost * 1.05
            final_price = max(min_price, smoothed_price)
            
            self.products[product_name].price = final_price
            self.products[product_name].price_history.append(final_price)
            return True
        return False
    
    def scale_action_to_price(self, action_value, product):
        price_change_pct = action_value * 0.25
        
        current_price = product.price
        
        new_price = current_price * (1 + price_change_pct)
        
        min_price = product.cost * 1.05
        max_price = product.cost * 3.0
        
        new_price = max(min_price, min(new_price, max_price))
        
        return new_price
    
    def save(self, path='models/maddpg_models'):
        os.makedirs(path, exist_ok=True)
        self.actor.save_weights(f"{path}/actor_{self.agent_id}.weights.h5")
        self.critic.save_weights(f"{path}/critic_{self.agent_id}.weights.h5")
        
    def load(self, path='models/maddpg_models'):
        try:
            self.actor.load_weights(f"{path}/actor_{self.agent_id}.weights.h5")
            self.critic.load_weights(f"{path}/critic_{self.agent_id}.weights.h5")
            self.update_target_networks(tau=1.0)
        except:
            print(f"Failed to load models for {self.agent_id}, using untrained networks")