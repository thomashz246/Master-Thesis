"""
MADDPGCoordinator:

Classes:
- JointReplayBuffer: Memory buffer for storing multi-agent experiences
- MADDPGCoordinator: Manages centralized training across multiple MADDPG agents

MADDPGCoordinator:
    - store_transition(): Records joint experiences from all agents
    - learn(): Coordinates training across agents using shared experiences
    - sample(): Retrieves batches of experiences with emphasis on recent data
"""
import numpy as np
import tensorflow as tf
from collections import deque
import random

class JointReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, joint_states, joint_actions, rewards, joint_next_states, dones):
        self.buffer.append((joint_states, joint_actions, rewards, joint_next_states, dones))
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        if buffer_size <= batch_size:
            return map(np.array, zip(*self.buffer))
            
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
    def __init__(self, maddpg_agents, state_dim=10, action_dim=1, batch_size=64):
        self.maddpg_agents = maddpg_agents
        self.num_agents = len(maddpg_agents)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.buffer = JointReplayBuffer()
        self.training_step = 0
    
    def store_transition(self, joint_states, joint_actions, joint_rewards, joint_next_states, dones=False):
        self.buffer.add(joint_states, joint_actions, joint_rewards, joint_next_states, dones)
    
    def learn(self):
        if self.buffer.size() < self.batch_size:
            return
        
        if self.training_step % 4 != 0:
            self.training_step += 1
            return
        
        try:
            joint_states_batch, joint_actions_batch, rewards_batch, joint_next_states_batch, dones_batch = self.buffer.sample(self.batch_size)
            
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