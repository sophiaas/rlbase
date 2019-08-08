import torch
import torch.nn as nn
from torch.distributions import Categorical
from core.replay_buffer import Memory
from core.logger import Logger
import pandas as pd
import numpy as np
from collections import defaultdict
import gym

"""
Base class for Deep RL agents
"""
EPS = np.finfo(np.float32).eps.item()



class BaseAgent(object):
    
    def __init__(self, config):
        self.config = config
        self.env = self.config.env.init_env()
        print('env: {}'.format(self.env))
        self.config.network.in_dim = self.env.observation_space.n
        self.config.network.out_dim = self.env.action_space.n
        self.config.network.device = self.config.training.device
        self.memory = Memory()
        self.logger = Logger(config)
        self.model = None
        self.policy = None
        self.episode = 0
        self.running_rewards = None
        self.running_moves = None
        self.average_rewards = 0
        self.average_moves = 0
        
    def reset(self):
        self.episode = 0
        self.replay_buffer.clear()
        self.env.reset()

    def to_cpu(self, x):
        if type(x) == torch.Tensor:
            return x.data.cpu().tolist()
        else:
            return x

    def convert_data(self, x):
        if type(x) == torch.Tensor:
            return x.data.cpu().tolist()
        elif type(x) == bool:
            return int(x)
        elif type(x) == np.ndarray:
            return list(x)
        else:
            return x

    def get_summary(self):
        summary = {
            'episode': int(self.episode),
            'running_rewards': self.running_rewards,
            'running_moves': self.running_moves
            }
        return summary
            
    def update_running_rewards(self, episode_rewards):
        if self.episode == 1:
            self.running_rewards = episode_rewards            
        else:
            self.running_rewards = self.running_rewards * 0.99 + episode_rewards * 0.01
            
    def update_running_moves(self, episode_moves):        
        if self.episode == 1:
            self.running_moves = episode_moves
        else:
            self.running_moves = self.running_moves * 0.99 + episode_moves * 0.01
            
    def update_average_rewards(self, episode_rewards):
        self.average_rewards += np.sum(episode_rewards)
        self.average_moves += len(episode_rewards)

    def train(self):
        #TODO: ADD HANDLE RESUME
        running_reward = 0
        avg_length = 0
        timestep = 0
            
        for i_episode in range(1, self.config.training.max_episodes+1):
            episode_reward = 0
            state = self.env.reset()
            episode_data = defaultdict(list, {'episode': int(self.episode)})
            for t in range(self.config.training.max_episode_length):
                timestep += 1
                transition, state, done = self.step(state)
                for key, val in transition.items():
                    episode_data[key].append(self.convert_data(val))
                episode_reward += transition['reward']
                
                if timestep % self.config.training.update_every == 0:
                    self.update(self.memory)
                    self.memory.clear()
                    timestep = 0

                running_reward += transition['reward']
                if self.config.experiment.render:
                    self.env.render()
                if done:
                    break
            self.episode += 1
            avg_length += t
            self.update_running_rewards(episode_reward)
            self.update_running_moves(t)
                    
            self.logger.push(self.get_summary())
            if self.config.experiment.save_episode_data:
                self.logger.push_episode_data(episode_data)
            
            if i_episode % self.config.experiment.log_interval == 0:
                self.average_rewards /= self.config.experiment.log_interval
                self.average_moves /= self.config.experiment.log_interval
                avg_length = int(avg_length/self.config.experiment.log_interval)
                running_reward = int((running_reward/self.config.experiment.log_interval))
    
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, 
                                                                          avg_length, 
                                                                          running_reward))
                running_reward = 0
                avg_length = 0
                
                self.logger.save()
                self.logger.save_checkpoint(self)
                
                if self.config.experiment.save_episode_data:
                    self.logger.save_episode_data()
                
                self.logger.plot('running_rewards')
                self.logger.plot('running_moves')
            
        print('Training complete')
        
    def sample_episode(self):
        return NotImplementedError
    
    def evaluate(self):
        return NotImplementedError
            
    def improve(self):
        return NotImplementedError
        