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
        total_episode_rewards = np.sum(episode_rewards)
        episode_moves = len(episode_rewards)
        
        if self.episode == 0:
            self.running_rewards = total_episode_rewards
            self.running_moves = episode_moves
            
        else:
            self.running_rewards = self.running_rewards * 0.99 \
                                        + total_episode_rewards * 0.01
            self.running_moves = self.running_moves * 0.99 + episode_moves * 0.01
            
    def update_average_rewards(self, episode_rewards):
        self.average_rewards += np.sum(episode_rewards)
        self.average_moves += len(episode_rewards)

#     def train(self):
#         #TODO: ADD HANDLE RESUME
#         running_reward = 0
#         avg_length = 0
#         timestep = 0
            
#         while self.episode < self.config.training.max_episodes:
#             episode_data = self.sample_episode()
#             self.logger.push(self.get_summary())
#             if self.config.experiment.save_episode_data:
#                 self.logger.push_episode_data(episode_data)
            
#             if self.episode % self.config.experiment.log_interval == 0 and self.episode > 0:
#                 self.average_rewards /= self.config.experiment.log_interval
#                 self.average_moves /= self.config.experiment.log_interval
                
                
#                 print('---------------------------')
#                 print('episode: {}'.format(self.episode))
# #                 print('running_moves: {}'.format(self.running_moves))
# #                 print('running_rewards: {}'.format(self.running_rewards))
#                 print('average_moves: {}'.format(self.average_moves))
#                 print('average_rewards: {}'.format(self.average_rewards))
#                 print('---------------------------')
                
#                 self.average_rewards = 0
#                 self.average_moves = 0
                
#                 self.logger.save()
#                 self.logger.save_checkpoint(self)
                
#                 if self.config.experiment.save_episode_data:
#                     self.logger.save_episode_data()
                
#                 self.logger.plot('running_rewards')
#                 self.logger.plot('running_moves')
                
# #             if self.episode % self.config.training.update_every == 0 and self.episode > 0:
# #                 print('updating weights...')
# #                 self.improve()
                
#             self.episode += 1
            
#         print('Training complete')


        
    def sample_episode(self):
        return NotImplementedError
    
    def evaluate(self):
        return NotImplementedError
            
    def improve(self):
        return NotImplementedError
        