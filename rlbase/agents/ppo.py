import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .base import BaseAgent
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
from networks.actor_critic import ActorCritic
from core.replay_buffer import Memory

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# """
# Advantage Actor-Critic Proximal Policy Optimization
# """

# class PPO(BaseAgent):
    
#     def __init__(self, config):
#         super(PPO, self).__init__(config)
#         self.policy = ActorCritic(config.network).to(device)
#         self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.training.lr, 
#                                           betas=config.training.betas)
#         self.policy_old = ActorCritic(config.network).to(device)
#         self.MseLoss = nn.MSELoss()
#         self.timestep = 0
        
# #     def sample_episode(self):
# #         state = self.env.reset()
# #         episode_rewards = []
# #         for t in range(self.config.training.max_episode_length):
# #             self.timestep += 1
            
# #             # Running policy_old:
# #             action = self.policy_old.act(state, self.memory)
# #             state, reward, done, _ = self.env.step(action)
# #             # Saving reward:
# #             self.memory.rewards.append(reward)
# #             episode_rewards.append(reward)

# #             # update if its time
# #             if self.timestep > 0 and self.timestep % self.config.training.update_every == 0:
# # #                 print('mem: {}'.format(self.memory.rewards))
# # #                 print('UPDATING')
# # #                 print('timestep: {}'.format(self.timestep))
# # #                 self.update(self.memory)
# #                 self.update()

# #                 self.memory.clear_memory()
# #                 self.timestep = 0
                
# # #             running_reward += reward
# #             if self.config.experiment.render:
# #                 self.env.render()
# #             if done:
# #                 break
# #         self.update_running_rewards(episode_rewards)
# #         self.update_average_rewards(episode_rewards)
        
        
    
#     def update(self, memory):
#         # Monte Carlo estimate of state rewards:
#         rewards = []
#         discounted_reward = 0
#         for reward in reversed(memory.rewards):
#             discounted_reward = reward + (self.config.algorithm.gamma * discounted_reward)
#             rewards.insert(0, discounted_reward)
        
#         # Normalizing the rewards:
#         rewards = torch.tensor(rewards).to(device)
#         rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
#         # convert list to tensor
#         old_states = torch.stack(memory.states).to(device).detach()
#         old_actions = torch.stack(memory.actions).to(device).detach()
#         old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
#         # Optimize policy for K epochs:
#         for _ in range(self.config.algorithm.optim_epochs):
#             # Evaluating old actions and values :
#             logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
#             # Finding the ratio (pi_theta / pi_theta__old):
#             ratios = torch.exp(logprobs - old_logprobs.detach())
                
#             # Finding Surrogate Loss:
#             advantages = rewards - state_values.detach()
#             surr1 = ratios * advantages
#             surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) * advantages
#             loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
#             # take gradient step
#             self.optimizer.zero_grad()
#             loss.mean().backward()
#             self.optimizer.step()
            
#     def train(self):
#         running_reward = 0
#         avg_length = 0
#         timestep = 0
#         memory = Memory()
#         # training loop
#         for i_episode in range(1, self.config.training.max_episodes+1):
#             state = self.env.reset()
#             for t in range(self.config.training.max_episode_length):
#                 timestep += 1

#                 # Running policy_old:
#                 action = self.policy_old.act(state, memory)
#                 state, reward, done, _ = self.env.step(action)
#                 # Saving reward:
#                 memory.rewards.append(reward)

#                 # update if its time
#                 if timestep % self.config.training.update_every == 0:
#                     self.update(memory)
#                     memory.clear_memory()
#                     timestep = 0

#                 running_reward += reward
# #                 if render:
# #                     self.env.render()
#                 if done:
#                     break

#             avg_length += t

# #             # stop training if avg_reward > solved_reward
# #             if running_reward > (log_interval*solved_reward):
# #                 print("########## Solved! ##########")
# #                 torch.save(self.policy.state_dict(), './PPO_{}.pth'.format(env_name))
# #                 break

#             # logging
#             if i_episode % self.config.experiment.log_interval == 0:
#                 avg_length = int(avg_length/self.config.experiment.log_interval)
#                 running_reward = int((running_reward/self.config.experiment.log_interval))

#                 print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
#                 running_reward = 0
#                 avg_length = 0


class PPO(object):
    
    def __init__(self, config):
        self.config = config

        self.config.network.in_dim = config.env.state_dim
        self.config.network.out_dim = config.env.action_dim
        self.config.network.device = device
        
        self.policy = ActorCritic(config.network).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.config.training.lr, 
                                          betas=self.config.training.betas)

        self.policy_old = ActorCritic(config.network).to(device)
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.config.algorithm.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.config.algorithm.optim_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) \ 
                                * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01 \
                                * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

