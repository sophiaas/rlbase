import numpy as np
import torch
import gym
from collections import defaultdict

from core.replay_buffer import Memory
from core.logger import Logger


"""
Base class for Deep RL agents
"""

class BaseAgent(object):
    
    def __init__(self, config):
        self.config = config
        
        self.env = config.env.init_env()
        
        self.eps = np.finfo(np.float32).eps.item()
        self.device = config.training.device
        
        self.memory = Memory()
        self.logger = Logger(config)
        
        self.model = None
        self.policy = None
       
        self.running_rewards = None
        self.running_moves = None
        
        self.episode = 1

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
            
    def update_running(self, episode_rewards, episode_moves):
        if self.episode == 1:
            self.running_rewards = episode_rewards   
            self.running_moves = episode_moves
        else:
            self.running_rewards = self.running_rewards * 0.99 + episode_rewards * 0.01
            self.running_moves = self.running_moves * 0.99 + episode_moves * 0.01

    def train(self):
        # TODO: Add handle resumes
        running_reward = 0
        avg_length = 0
        timestep = 0
            
        # Iterate through episodes
        for i_episode in range(1, self.config.training.max_episodes+1):
            episode_reward = 0
            episode_data = defaultdict(list, {'episode': int(self.episode)})
            
            state = self.env.reset()
            
            # Iterate through steps
            for t in range(self.config.training.max_episode_length):
                timestep += 1
                
                transition, state, done = self.step(state)
                
                for key, val in transition.items():
                    episode_data[key].append(self.convert_data(val))
                    
                episode_reward += transition['reward']
                
                if timestep % self.config.training.update_every == 0:
                    self.update()
                    self.memory.clear()
                    timestep = 0

                running_reward += transition['reward']
                
                if self.config.experiment.render:
                    self.env.render()
                if done:
                    break
                    
            # Update logging variables
            self.episode += 1
            avg_length += t
            
            self.update_running(episode_reward, t)
                    
            self.logger.push(self.get_summary())
            
            if self.config.experiment.save_episode_data:
                self.logger.push_episode_data(episode_data)
            
            # Logging
            if i_episode % self.config.experiment.log_interval == 0:

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
    
    def step(self):
        return NotImplementedError