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

        self.optimizer = config.training.optim(self.policy.parameters(),
                                          lr=self.config.training.lr) 
# weight_decay=self.config.training.weight_decay)
        #TODO: Add back in weight decay

        self.lr_scheduler = self.config.training.lr_scheduler(
                                        self.optimizer, 
                                        step_size=config.training.lr_step_interval, 
                                        gamma=config.training.lr_gamma)
    
    def discounted_advantages(self, rewards, masks, values, gamma, tau):
        shape = values.shape
        returns = torch.zeros(shape).to(self.device)
        deltas = torch.zeros(shape).to(self.device)
        advantages = torch.zeros(shape).to(self.device)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        
        # Compute discounted returns and advantages
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
        
        return advantages, returns
        
    def update(self):   

        # Convert list to tensor
        # TODO: torch.no_grad when acting. No need to detach
        # TODO: check if eveyrthing works out using old_logprobs stored in memory
        old_states = torch.stack(self.memory.state).to(self.device).detach()
        old_actions = torch.stack(self.memory.action).to(self.device).detach()
#         old_logprobs = torch.stack(self.memory.logprob).to(self.device).detach()
        old_masks = torch.tensor(self.memory.mask).to(self.device).detach()
        old_rewards = torch.tensor(self.memory.reward).to(self.device)
        
        with torch.no_grad():
            old_logprobs, values, _ = self.policy.evaluate(old_states, old_actions)
            advantages, returns = self.discounted_advantages(old_rewards, 
                                                             old_masks,
                                                             values,
                                                        self.config.training.gamma,
                                                        self.config.training.tau) #0.95
                
        # Optimize policy for K epochs:
        for _ in range(self.config.algorithm.optim_epochs):
            permutation = torch.randperm(old_states.shape[0]).to(self.device)
            for m in range(0, old_states.shape[0], self.config.training.minibatch_size):
                idxs = permutation[m:m+self.config.training.minibatch_size]

                # Evaluate old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    old_states[idxs].detach().requires_grad_(), 
                    old_actions[idxs].detach().float().requires_grad_())
                # TODO: do we need grads on states and actions?

                # Find the ratio (policy / old policy):
                ratios = torch.exp(logprobs - old_logprobs[idxs].detach())
        
                # Compute surrogate loss
                surr1 = ratios * advantages[idxs]
                surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 
                                    1+self.config.algorithm.clip) * advantages[idxs]
                
                actor_loss = -torch.min(surr1, surr2)       
                critic_loss = (state_values - returns[idxs]) ** 2
#                 entropy_penalty = -0.01 * dist_entropy
                loss = actor_loss + critic_loss
#                      + entropy_penalty
                #TODO: add back in the entropy penalty

                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.optimizer.step()
        
        # Step learning rate
        self.lr_scheduler.step()
        
    def step(self, state):
        # Run old policy:
        env_data = self.env.raw_state
        action, start_state, log_prob = self.policy.act(state)
        state, reward, done, _ = self.env.step(action.item())
        if self.episode_steps == self.config.training.max_episode_length:
            done = True

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
