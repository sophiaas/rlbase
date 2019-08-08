import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent
from networks.actor_critic import ActorCritic
from core.replay_buffer import Memory
from envs import Lightbot


"""
Advantage Actor-Critic Proximal Policy Optimization
"""

class PPO(BaseAgent):
    
    def __init__(self, config):
        super(PPO, self).__init__(config)
        
        self.config.network.body.indim = self.config.env.obs_dim
        self.config.network.heads['actor'].outdim = self.config.env.action_dim
        
        self.policy = ActorCritic(config).to(self.device)
        self.policy_old = ActorCritic(config).to(self.device)

        self.optimizer = config.training.optim(self.policy.parameters(),
                                          lr=self.config.training.lr, 
                                          betas=self.config.training.betas)

        self.MSELoss = nn.MSELoss()
        
    def discount(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for i, reward in enumerate(reversed(self.memory.reward)):
            discounted_reward = reward \
                                + (self.config.algorithm.gamma * discounted_reward * self.memory.mask[i])
            rewards.insert(0, discounted_reward)
        return rewards
        
    def update(self):   
        # Discount and normalize the rewards:
        rewards = self.discount()
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        
        # Convert list to tensor
        old_states = torch.stack(self.memory.state).to(self.device).detach()
        old_actions = torch.stack(self.memory.action).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprob).to(self.device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.config.algorithm.optim_epochs):
            
            # Evaluate old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Find the ratio (policy / old policy):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Find surrogate loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) \
                                * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MSELoss(state_values, rewards) - 0.01 \
                                * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def step(self, state):
        # Run old policy:
        action, start_state, log_prob = self.policy_old.act(state)
        state, reward, done, _ = self.env.step(action.item())
        
        step_data = {
            'reward': reward, 
            'mask': bool(not done),
            'state': start_state,
            'action': action,
            'logprob': log_prob
        }
        
        # Push to memory:
        self.memory.push(step_data)
        
        return step_data, state, done
