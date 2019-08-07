import torch
import torch.nn as nn
from torch.distributions import Categorical
from core.replay_buffer import ReplayBuffer
from core.logger import Logger
import pandas as pd
import numpy as np
from collections import defaultdict
# from envs.utils import set_env
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
        self.replay_buffer = ReplayBuffer(config)
        self.logger = Logger(config)
        self.model = None
        self.policy = None
        self.episode = 0
        self.running_rewards = None
        self.running_moves = None
        if self.config.training.cuda:
            torch.cuda.set_device(self.config.training.device)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        
    def reset(self):
        self.episode = 0
        self.replay_buffer.clear()
        self.env.reset()
    
    def if_cuda(self, x):
        if self.config.training.cuda:
            return x.cuda()
        else:
            return x

#     def tensor(self, x):
#         if isinstance(x, torch.Tensor):
#             return x
#         x = np.asarray(x, dtype=np.float)
#         x = torch.as_tensor(x)
#         x = torch.tensor(x, device=self.config.training.device, dtype=torch.float32, requires_grad=True)
#         return x
    
    def cuda(self):
        for network in self.model.values():
            network.cuda()
            
    def normalize(self, x):
        return (x - np.mean(x)) / (np.std(x) + EPS)
#         return (x - x.mean()) / (x.std() + EPS)
        
    def policy_forward(self, state):
        prediction = self.policy.forward(state)
        
        if type(self.env.action_space) == gym.spaces.Discrete:
            max_val, max_idx = torch.max(prediction, 1)
            action_scores = prediction - torch.max(prediction[:, max_idx]) # subtract max logit
            probabilities = torch.exp(action_scores)
            probabilities = torch.clamp(probabilities, float(np.finfo(np.float32).eps), 1)  # for numerical instabilities ??
            if len(probabilities.shape) == 3:
                probabilities = torch.squeeze(probabilities)
            distribution = Categorical(probabilities)
            
        elif type(self.env.action_space) == gym.spaces.Box:
            return NotImplementedError
        
        else:
            raise ValueError('env.action_space must be either gym.spaces.Discrete \
                             or gym.spaces.Box')
            
        return distribution
    
    def log_prob(self, state, action):
        # not volatile
        distribution = self.policy_forward(state)
        log_prob = distribution.log_prob(action)
        return log_prob
    
    def sample_action(self, state):
#         state = self.if_cuda(state)
        with torch.no_grad():
            distribution = self.policy_forward(state)
        
        action = distribution.sample().data
        log_prob = self.log_prob(state, action)  # not volatile
        value = self.value_net(state)  # not volatile
        return {'action': action[0], 'log_prob': log_prob, 'value': value}
    
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
        
    def sample_episode(self):
        episode_data = defaultdict(list, {'episode': int(self.episode)})
        state = self.env.reset()
        
        for t in range(self.config.training.max_episode_length):
#             state = torch.from_numpy(np.expand_dims(state, 0))
            state = torch.as_tensor(np.expand_dims(state, 0)).float()
            step_data = {'state': state}
            action_data = self.sample_action(state)
            state_data = self.env.step(action_data['action'])
            step_data.update(action_data)
            step_data.update(state_data)
            for key, val in step_data.items():
                if key != 'next_state' and key != 'state':
                    episode_data[key].append(self.convert_data(val))
            if self.config.experiment.render:
                self.env.render()
#             print('STEP DATA: {}'.format(step_data))
            self.replay_buffer.push(step_data)
            state = step_data['next_state']
            if step_data['done']:
#                 self.reset() ??
                break
        self.update_running_rewards(episode_data['reward'])
        return episode_data

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

    def train(self):
        #TODO: ADD HANDLE RESUME
        if self.config.training.cuda:
            self.cuda()
            
        while self.episode < self.config.training.max_episodes:
            episode_data = self.sample_episode()
            self.logger.push(self.get_summary())
            if self.config.experiment.save_episode_data:
                self.logger.push_episode_data(episode_data)
            
            if self.episode % self.config.experiment.log_interval == 0 and self.episode > 0:
                
                print('---------------------------')
                print('episode: {}'.format(self.episode))
                print('running_moves: {}'.format(self.running_moves))
                print('running_rewards: {}'.format(self.running_rewards))
                print('---------------------------')
                
                self.logger.save()
                self.logger.save_checkpoint(self)
                
                if self.config.experiment.save_episode_data:
                    self.logger.save_episode_data()
                
                self.logger.plot('running_rewards')
                self.logger.plot('running_moves')
                
            if self.episode % self.config.training.update_every == 0 and self.episode > 0:
                print('updating weights...')
                self.improve()
                
            self.episode += 1
            
        print('Training complete')
        
    def evaluate(self):
        return NotImplementedError
            
    def improve(self):
        return NotImplementedError
        