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
from envs import Lightbot

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

"""
Advantage Actor-Critic Proximal Policy Optimization
"""

class PPO(BaseAgent):
    
    def __init__(self, config):
        super(PPO, self).__init__(config)
        
        self.policy = ActorCritic(config.network).to(device)
        self.policy_old = ActorCritic(config.network).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.config.training.lr, 
                                          betas=self.config.training.betas)

        self.MSELoss = nn.MSELoss()
        
    def discount(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for i, reward in enumerate(reversed(memory.rewards)):
            discounted_reward = reward \
                                + (self.config.algorithm.gamma * discounted_reward * memory.masks[i])
            rewards.insert(0, discounted_reward)
        return rewards
        
    def update(self, memory):   
        # Normalizing the rewards:
        rewards = self.discount(memory)
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
            loss = -torch.min(surr1, surr2) + 0.5 * self.MSELoss(state_values, rewards) - 0.01 \
                                * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def step(self, state):
        # Running policy_old:
        action, start_state, log_prob = self.policy_old.act(state, self.memory)
        state, reward, done, _ = self.env.step(action.item())
        
        # Push to memory:
        self.memory.push({
            'rewards': reward, 
            'masks': bool(not done),
            'states': start_state,
            'actions': action,
            'logprobs': log_prob
        })
        return reward, done
        


        
        
        
    def train(self):
        # Logging variables
        running_reward = 0
        avg_length = 0
        timestep = 0

        # Training loop
        for i_episode in range(1, self.config.training.max_episodes+1):
            state = self.env.reset()
            episode_data = {}
            for t in range(self.config.training.max_episode_length):
                timestep += 1

                # Running policy_old:
                action, start_state, log_prob = self.policy_old.act(state, self.memory)
                state, reward, done, _ = self.env.step(action.item())
                
                # Push to memory:
                self.memory.push({
                    'rewards': reward, 
                    'masks': bool(not done),
                    'states': start_state,
                    'actions': action,
                    'logprobs': log_prob
                })

                # update if its time
                if timestep % self.config.training.update_every == 0:
                    self.update(self.memory)
                    self.memory.clear()
                    timestep = 0

                running_reward += reward
                if self.config.experiment.render:
                    self.env.render()
                if done:
                    break

            self.episode += 1
            avg_length += t


            # Logging
            if i_episode % self.config.experiment.log_interval == 0:
                avg_length = int(avg_length/self.config.experiment.log_interval)
                running_reward = int((running_reward/self.config.experiment.log_interval))

                self.logger.push(self.get_summary())

                #TODO add episode data
                episode_data = {}

                if self.config.experiment.save_episode_data:
                    self.logger.push_episode_data(episode_data)

                self.logger.plot('running_rewards')
                self.logger.plot('running_moves')

                self.logger.save()
                self.logger.save_checkpoint(self)

                if self.config.experiment.save_episode_data:
                    self.logger.save_episode_data()

                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, 
                                                                          avg_length, 
                                                                          running_reward))
                running_reward = 0
                avg_length = 0
                
        print('Training complete')

