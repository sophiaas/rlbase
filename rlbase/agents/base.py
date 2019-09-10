import numpy as np
import torch
import gym
from collections import defaultdict

from core.replay_buffer import Memory
from core.logger import Logger

from torch.optim.lr_scheduler import StepLR


"""
Base class for Deep RL agents
"""

class BaseAgent(object):
    
    def __init__(self, config):
        torch.manual_seed(config.experiment.seed)
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
        self.episode_steps = 0

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
            'episode': int(self.episode)-10,
            'running_rewards': self.running_rewards,
            'running_moves': self.running_moves
            }
        return summary
            
    def update_running(self, episode_rewards, episode_moves):
        if self.episode == 10:
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
        self.episode_steps = 0

        # Iterate through episodes
        for i_episode in range(1, self.config.training.max_episodes+1):
            episode_reward = 0
            episode_data = defaultdict(list, {'episode': int(self.episode)})
            state = self.env.reset()
            
            # Iterate through steps
            for t in range(1, self.config.training.max_episode_length+1):
                timestep += 1
                self.episode_steps += 1
                state = torch.from_numpy(state).float().to(self.device)
                
                with torch.no_grad():
                    transition, state, done = self.step(state)
                
                for key, val in transition.items():
                    episode_data[key].append(self.convert_data(val))
                
                episode_reward += transition['reward']
                running_reward += transition['reward']
                
                if self.config.experiment.render:
                    self.env.render()
                    
                if done:
                    break

                if timestep % self.config.training.update_every == 0:
                    self.update()
                    self.memory.clear()
            
            if self.config.experiment.save_episode_data and self.episode % self.config.experiment.every_n_episodes == 0:
                self.logger.push_episode_data(episode_data)
                
                
            # Logging
            if i_episode % self.config.experiment.log_interval == 0:
                
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, 
                                                                    round(self.running_moves, 2), 
                                                                    round(self.running_rewards, 2)))
                self.logger.save()
                self.logger.save_checkpoint(self)
                
                if self.config.experiment.save_episode_data:
                    self.logger.save_episode_data()
                
                self.logger.plot('running_rewards')
                self.logger.plot('running_moves')
                
            avg_length += self.episode_steps
            
#             if i_episode % 10 == 0:
#                 self.update_running(running_reward/10, avg_length/10)
#                 running_reward = 0
#                 avg_length = 0
#                 self.logger.push(self.get_summary())
            if i_episode % 10 == 0:
                self.update_running(running_reward/10, avg_length/10)
                running_reward = 0
                avg_length = 0
                summary = {
                    'steps': timestep
                    'return': episode_reward
                    'moves': self.episode_steps
                }
                self.logger.push(summary)
                
            self.episode_steps = 0
            self.episode += 1
            
        print('Training complete')
        
    def evaluate(self):
        """TODO: Break out redundant functions from train() and consolidate"""
        running_reward = 0
        avg_length = 0
        timestep = 0
        self.episode_steps = 0

        # Iterate through episodes            
        for i_episode in range(1, self.config.training.max_episodes+1):
            episode_reward = 0
            episode_data = defaultdict(list, {'episode': int(self.episode)})
            state = self.env.reset()
            
            # Iterate through steps
            for t in range(1, self.config.training.max_episode_length+1):
                timestep += 1
                self.episode_steps += 1
                state = torch.from_numpy(state).float().to(self.device)
                
                with torch.no_grad():
                    transition, state, done = self.step(state)
                
                for key, val in transition.items():
                    episode_data[key].append(self.convert_data(val))
                
                episode_reward += transition['reward']
                running_reward += transition['reward']
                
                if self.config.experiment.render:
                    self.env.render()
                    
                if done:
                    break

                self.memory.clear()
                
                if timestep == self.config.eval.n_eval_steps:
                    break
            
            self.logger.push_episode_data(episode_data)
                
                
            # Logging
            if i_episode % self.config.experiment.log_interval == 0:
                
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, 
                                                                    round(self.running_moves, 2), 
                                                                    round(self.running_rewards, 2)))
                self.logger.save()
                
                if self.config.experiment.save_episode_data:
                    self.logger.save_episode_data()
                
                self.logger.plot('running_rewards')
                self.logger.plot('running_moves')
                
            avg_length += self.episode_steps
            
            if i_episode % 10 == 0:
                self.update_running(running_reward/10, avg_length/10)
                running_reward = 0
                avg_length = 0
                self.logger.push(self.get_summary())
                
            self.episode_steps = 0
            self.episode += 1
            
            if timestep == self.config.eval.n_eval_steps:
                break
            
        print('Evaluation complete')
    
    def step(self):
        return NotImplementedError