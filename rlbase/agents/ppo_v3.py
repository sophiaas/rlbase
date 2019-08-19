import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent
from networks.actor_critic import ActorCritic
from core.replay_buffer import Memory
from envs import Lightbot

#PPO EPSILON CLIP ANNEALING!!!


"""
Advantage Actor-Critic Proximal Policy Optimization
"""

class PPO(BaseAgent):
    
    def __init__(self, config):
        super(PPO, self).__init__(config)
        
        self.config.network.body.indim = self.config.env.obs_dim
        self.config.network.heads['actor'].outdim = self.config.env.action_dim
        
        self.policy = ActorCritic(config).to(self.device)
#         self.policy_old = ActorCritic(config).to(self.device)

#         self.actor_optimizer = config.training.optim(self.policy.actor.parameters(),
#                                           lr=self.config.training.lr) 
    
#         self.critic_optimizer = config.training.optim(self.policy.critic.parameters(),
#                                           lr=self.config.training.lr) 

        self.optimizer = config.training.optim(self.policy.parameters(),
                                          lr=self.config.training.lr) 
#                                           betas=self.config.training.betas,
#                                           weight_decay=self.config.training.weight_decay)

        self.lr_scheduler = self.config.training.lr_scheduler(self.optimizer, 
                                                              step_size=config.training.lr_step_interval, 
                                                              gamma=config.training.lr_gamma)
    
    def discounted_advantages(self, rewards, masks, values, gamma, tau):
        #TODO: why is this taking so long to compute?
        returns = torch.zeros(values.shape).to(self.device)
        deltas = torch.zeros(values.shape).to(self.device)
        advantages = torch.zeros(values.shape).to(self.device)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
            
        advantages = (advantages - advantages.mean()) / advantages.std()
        
        return advantages, returns
        
    def update(self):   
        
        #V1: compute advantages and values before optim loop
        #V2: add requires grad
        #V3: compute old log probs before optim loop
        
        # Discount and normalize the rewards:
#         rewards = self.discount()
#         rewards = torch.tensor(rewards).to(self.device)
#         rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        
        # Convert list to tensor
        old_states = torch.stack(self.memory.state).to(self.device).detach()
        old_actions = torch.stack(self.memory.action).to(self.device).detach()
#         old_logprobs = torch.stack(self.memory.logprob).to(self.device).detach()
        old_masks = torch.tensor(self.memory.mask).to(self.device).detach()
        old_rewards = torch.tensor(self.memory.reward).to(self.device)
        
        with torch.no_grad():
            old_logprobs, values, _ = self.policy.evaluate(old_states, old_actions)
            advantages, returns = self.discounted_advantages(old_rewards, old_masks, 
                                                                 values, self.config.training.gamma, 0.95)
                
        # Optimize policy for K epochs:
        for _ in range(self.config.algorithm.optim_epochs):
            permutation = torch.randperm(old_states.shape[0]).to(self.device)
            for m in range(0, old_states.shape[0], self.config.training.minibatch_size):
                idxs = permutation[m:m+self.config.training.minibatch_size]

#                 # Evaluate old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[idxs].detach().requires_grad_(), old_actions[idxs].detach().float().requires_grad_())

#                 # Find the ratio (policy / old policy):
#                 ratios = torch.exp(logprobs - old_logprobs.detach().requires_grad_())
                ratios = torch.exp(logprobs - old_logprobs[idxs].detach())
    
    

#                 ratios = torch.exp(logprobs - old_logprobs.detach()[idxs].requires_grad_())

#                 # Find surrogate loss:
#                 #TODO: this is different than the advantage calculation in old code. Find out if it matters

#                 advantages = rewards[idxs] - state_values.detach()
        
                surr1 = ratios * advantages[idxs]
                surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) * advantages[idxs]
                
                actor_loss = -torch.min(surr1, surr2)       
                critic_loss = (state_values - returns[idxs]) ** 2
#                 entropy_penalty = -0.01 * dist_entropy
                loss = actor_loss + critic_loss
#                      + entropy_penalty
#                 print('actor loss: {}'.format(actor_loss.mean()))
#                 print('critic loss: {}'.format(critic_loss.mean()))

#                 # Take gradient step

#                 self.actor_optimizer.zero_grad()
#                 actor_loss.mean().backward()
# #                 nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 40)
#                 self.actor_optimizer.step()
                
#                 self.critic_optimizer.zero_grad()
#                 critic_loss.mean().backward()
# #                 nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 40)
#                 self.critic_optimizer.step()
                
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.optimizer.step()
        
        # Step learning rate
#         self.actor_lr_scheduler.step()
#         self.critic_lr_scheduler.step()

        self.lr_scheduler.step()

        
#         # Copy new weights into old policy:
#         self.policy_old.load_state_dict(self.policy.state_dict())
        
    def step(self, state):
        # Run old policy:
#         env_data = self.env.raw_state
        action, start_state, log_prob = self.policy.act(state)

#         action, start_state, log_prob = self.policy_old.act(state)
        state, reward, done, env_data = self.env.step(action.item())
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


# import numpy as np
# import torch
# import torch.nn as nn

# from .base import BaseAgent
# from networks.actor_critic import ActorCritic
# from core.replay_buffer import Memory
# from envs import Lightbot

# from torch.utils.tensorboard import SummaryWriter



# """
# Advantage Actor-Critic Proximal Policy Optimization
# """

# class PPO(BaseAgent):
    
#     def __init__(self, config):
#         super(PPO, self).__init__(config)
        
#         self.config.network.body.indim = self.config.env.obs_dim
#         self.config.network.heads['actor'].outdim = self.config.env.action_dim
        
#         self.policy = ActorCritic(config).to(self.device)
#         self.policy_old = ActorCritic(config).to(self.device)

#         self.optimizer = config.training.optim(self.policy.parameters(),
#                                           lr=self.config.training.lr, 
#                                           betas=self.config.training.betas,
#                                           weight_decay=self.config.training.weight_decay)
        
#         self.lr_scheduler = self.config.training.lr_scheduler(self.optimizer, 
#                                                               step_size=1, 
#                                                               gamma=config.training.lr_gamma)
        
#     def discount(self):
#         # Monte Carlo estimate of state rewards:
#         rewards = []
#         discounted_reward = 0
#         for i, reward in enumerate(reversed(self.memory.reward)):
#             discounted_reward = reward \
#                                 + (self.config.algorithm.gamma * discounted_reward * self.memory.mask[i])
#             rewards.insert(0, discounted_reward)
#         return rewards
    
#     def update(self):   
#         # Discount and normalize the rewards:
#         rewards = self.discount()
#         rewards = torch.tensor(rewards).to(self.device)
#         rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        
#         # Convert list to tensor
#         old_states = torch.stack(self.memory.state).to(self.device).detach()
#         old_actions = torch.stack(self.memory.action).to(self.device).detach()
#         old_logprobs = torch.stack(self.memory.logprob).to(self.device).detach()
        
#         # Optimize policy for K epochs:
#         for _ in range(self.config.algorithm.optim_epochs):
#             permutation = torch.randperm(old_states.shape[0]).to(self.device)
#             for m in range(0, old_states.shape[0], self.config.training.minibatch_size):
#                 idxs = permutation[m:m+self.config.training.minibatch_size]
                
# #                 # Evaluate old actions and values :
#                 logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[idxs], old_actions[idxs])
    
# #                 # Find the ratio (policy / old policy):
#                 ratios = torch.exp(logprobs - old_logprobs.detach()[idxs])
    
# #                 # Find surrogate loss:

# #                 #TODO: this is different than the advantage calculation in old code. Find out if it matters
#                 advantages = rewards[idxs] - state_values.detach()
    
#                 surr1 = ratios * advantages
#                 surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) \
#                                     * advantages
            
#                 actor_loss = -torch.min(surr1, surr2) 
#                 critic_loss = (state_values - rewards[idxs]) ** 2
#                 entropy_penalty = -0.01 * dist_entropy
#                 loss = actor_loss + critic_loss + entropy_penalty
                
# #                 # Take gradient step
#                 self.optimizer.zero_grad()
#                 loss.mean().backward()
#                 nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
#                 self.optimizer.step()

#                 self.lr_scheduler.step()
                
#         # Copy new weights into old policy:
#         self.policy_old.load_state_dict(self.policy.state_dict())
        
#     def update(self):   
#         # Discount and normalize the rewards:
#         rewards = self.discount()
#         rewards = torch.tensor(rewards, requires_grad=True).to(self.device)
#         rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        
#         # Convert list to tensor
#         old_states = torch.stack(self.memory.state).to(self.device).clone().detach()
#         old_actions = torch.stack(self.memory.action).to(self.device).clone().detach()
#         old_logprobs = torch.stack(self.memory.logprob).to(self.device).clone().detach()  
                    
#         # Optimize policy for K epochs:
#         for _ in range(self.config.algorithm.optim_epochs):
#             permutation = torch.randperm(old_states.shape[0]).to(self.device)
#             for m in range(0, old_states.shape[0], self.config.training.minibatch_size):
#                 idxs = permutation[m:m+self.config.training.minibatch_size]

#                 # Evaluate old actions and values :
#                 logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[idxs], 
#                                                             old_actions[idxs])

#                 # Find the ratio (policy / old policy):
        
#                 ratios = torch.exp(logprobs - old_logprobs.detach()[idxs])

#                 # Find surrogate loss:
#                 #TODO: this is different than the advantage calculation in old code. Find out if it matters
#                 advantages = rewards[idxs] - state_values.detach()

#                 surr1 = ratios * advantages
#                 surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) \
#                                     * advantages
#                 actor_loss = -torch.min(surr1, surr2) 
#                 critic_loss = (state_values - rewards[idxs]) ** 2
# #                 entropy_penalty = -0.01 * dist_entropy
#                 loss = actor_loss + critic_loss
# #                         + entropy_penalty

#                 # Take gradient step
#                 self.optimizer.zero_grad()
#                 loss.mean().backward(retain_graph=True)
#                 nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
#                 self.optimizer.step()

#         self.lr_scheduler.step()
        
#         # Copy new weights into old policy:
#         self.policy_old.load_state_dict(self.policy.state_dict())
        
#     def step(self, state):
#         # Run old policy:
#         action, start_state, log_prob = self.policy_old.act(state)
#         state, reward, done, env_data = self.env.step(action.item())
        
#         step_data = {
#             'reward': reward, 
#             'mask': bool(not done),
#             'state': start_state,
#             'action': action,
# #             'value': value,
#             'logprob': log_prob,
#             'env_data': env_data
#         }
        
#         # Push to memory:
#         self.memory.push(step_data)
        
#         return step_data, state, done
