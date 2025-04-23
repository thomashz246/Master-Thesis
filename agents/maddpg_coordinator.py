import numpy as np
import tensorflow as tf
from collections import deque
import random

class JointReplayBuffer:
    """Buffer to store joint experiences from all agents"""
    
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, joint_states, joint_actions, rewards, joint_next_states, dones):
        """
        Store joint experience
        joint_states: array of shape [num_agents, state_dim]
        joint_actions: array of shape [num_agents, action_dim]
        rewards: array of shape [num_agents]
        joint_next_states: array of shape [num_agents, state_dim]
        dones: array of shape [num_agents] or single value
        """
        self.buffer.append((joint_states, joint_actions, rewards, joint_next_states, dones))
    
    def sample(self, batch_size):
        """Sample batch of joint experiences"""
        buffer_size = len(self.buffer)
        if buffer_size <= batch_size:
            return map(np.array, zip(*self.buffer))
            
        # Add recency bias - more weight to recent experiences
        recent_count = min(batch_size // 2, buffer_size // 4)
        recent_indices = range(buffer_size - recent_count, buffer_size)
        old_indices = random.sample(range(buffer_size - recent_count), batch_size - recent_count)
        indices = list(recent_indices) + old_indices
        
        joint_states = []
        joint_actions = []
        rewards = []
        joint_next_states = []
        dones = []
        
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            joint_states.append(s)
            joint_actions.append(a)
            rewards.append(r)
            joint_next_states.append(ns)
            dones.append(d)
        
        return (np.array(joint_states), np.array(joint_actions), 
                np.array(rewards), np.array(joint_next_states), 
                np.array(dones))
    
    def size(self):
        return len(self.buffer)

class MADDPGCoordinator:
    """Coordinates training between multiple MADDPG agents"""
    
    def __init__(self, maddpg_agents, state_dim=10, action_dim=1, batch_size=64):
        self.maddpg_agents = maddpg_agents
        self.num_agents = len(maddpg_agents)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.buffer = JointReplayBuffer()
        self.training_step = 0
    
    def store_transition(self, joint_states, joint_actions, joint_rewards, joint_next_states, dones=False):
        """Store joint transition in buffer"""
        self.buffer.add(joint_states, joint_actions, joint_rewards, joint_next_states, dones)
    
    def learn(self):
        """Centralized training function"""
        if self.buffer.size() < self.batch_size:
            return
        
        # Only train every 4 steps to stabilize learning
        if self.training_step % 4 != 0:
            self.training_step += 1
            return
        
        try:
            # Sample batch of joint experiences
            joint_states_batch, joint_actions_batch, rewards_batch, joint_next_states_batch, dones_batch = self.buffer.sample(self.batch_size)
            
            # Train each agent using joint information
            for agent_idx, agent in enumerate(self.maddpg_agents):
                agent.learn_from_joint_experience(
                    joint_states_batch, 
                    joint_actions_batch, 
                    rewards_batch[:, agent_idx:agent_idx+1], 
                    joint_next_states_batch, 
                    dones_batch,
                    self.maddpg_agents
                )
            
            self.training_step += 1
            
            # Print training progress periodically
            if self.training_step % 10 == 0:
                print(f"\nJoint training step {self.training_step}")
                for idx, agent in enumerate(self.maddpg_agents):
                    if hasattr(agent, 'actor_losses') and len(agent.actor_losses) > 0:
                        print(f"  Agent {agent.agent_id} - Actor Loss: {agent.actor_losses[-1]:.6f}, "
                              f"Critic Loss: {agent.critic_losses[-1]:.6f}")
        
        except Exception as e:
            print(f"Error in MADDPG coordinator training: {e}")
            import traceback
            traceback.print_exc()