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
                                          betas=self.config.training.betas,
                                          weight_decay=self.config.training.weight_decay)

        self.lr_scheduler = self.config.training.lr_scheduler(self.optimizer, 
                                                              step_size=1, 
                                                              gamma=config.training.lr_gamma)
        
    def discount(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
#         d = 0
#         v = 0
#         a = 0
        for i, reward in enumerate(reversed(self.memory.reward)):
            discounted_reward = reward \
                                + (self.config.algorithm.gamma * discounted_reward * self.memory.mask[i])
            rewards.insert(0, discounted_reward)
#             d = reward + self.config.algorithm.gamma * v * self.memory.mask[i] - self.memory.value[i]
#             a = d + self.config.algorithm.gamma * self.config.algorithm.tau * a * self.memory.mask[i]
#             v = self.memory.value[i]
#             advantages.insert(0, a)
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
            permutation = torch.randperm(old_states.shape[0]).to(self.device)
            for m in range(0, old_states.shape[0], self.config.training.minibatch_size):
                idxs = permutation[m:m+self.config.training.minibatch_size]

#                 # Evaluate old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[idxs], old_actions[idxs])

#                 # Find the ratio (policy / old policy):
                ratios = torch.exp(logprobs - old_logprobs.detach()[idxs])

#                 # Find surrogate loss:
#                 #TODO: this is different than the advantage calculation in old code. Find out if it matters
                advantages = rewards[idxs] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) \
                                    * advantages
                actor_loss = -torch.min(surr1, surr2) 
                critic_loss = (state_values - rewards[idxs]) ** 2
                entropy_penalty = -0.01 * dist_entropy
                loss = actor_loss + critic_loss + entropy_penalty

#                 # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.optimizer.step()
        
        # Step learning rate
        self.lr_scheduler.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def step(self, state):
        # Run old policy:
        action, start_state, log_prob = self.policy_old.act(state)
        state, reward, done, env_data = self.env.step(action.item())
        
        step_data = {
            'reward': reward, 
            'mask': bool(not done),
            'state': start_state,
            'action': action,
            'logprob': log_prob,
            'env_data': env_data
        }
        
        # Push to memory:
        self.memory.push(step_data)
        
        return step_data, state, done
