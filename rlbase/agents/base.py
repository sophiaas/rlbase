import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical
from core.replay_buffer import ReplayBuffer
from core.logger import Logger
import pandas as pd
import numpy as np
from collections import defaultdict
# from envs.utils import set_env
import gym


class BaseAgent(object):
    
    def __init__(self, config):
        self.config = config
        self.env = self.config.env.init()
#         self.config.env.action_dim = self.env.action_space.n #FIX THIS
#         self.config.env.obs_dim = self.env.observation_space.n # FIX THIS
        self.replay_buffer = ReplayBuffer(config)
        self.logger = Logger(config)
        self.model = None
        self.policy = None
        self.episode = 0
        self.running_rewards = None
        self.running_moves = None
        
    def reset(self):
        self.episode = 0
        self.replay_buffer.clear()
        self.env.reset()
    
    def cuda_if_needed(self, x):
        if self.config.training.cuda:
            return x.cuda()
        else:
            return x
    
    def cuda(self):
        for network in self.model.values():
            network.cuda()
        
    def policy_forward(self, state):
        prediction = self.policy.forward(state)
        
        if type(self.env.action_space) == gym.spaces.Discrete:
            max_val, max_idx = torch.max(prediction, 1)
            action_scores = prediction - torch.max(prediction[:, max_idx]) # subtract max logit
            probabilities = torch.exp(action_scores)
            probabilities = torch.clamp(probabilities, float(np.finfo(np.float32).eps), 1)  # for numerical instabilities
            distribution = Categorical(probabilities)
            
        elif type(self.env.action_space) == gym.spaces.Box:
            return NotImplementedError
        
        else:
            raise ValueError('env.action_space must be either gym.spaces.Discrete or gym.spaces.Box')
            
        return distribution
    
    def log_prob(self, state, action):
        # not volatile
        distribution = self.policy_forward(state)
        log_prob = distribution.log_prob(action)
        return log_prob
    
    def sample_action(self, state):
        state = self.cuda_if_needed(torch.from_numpy(state).float().unsqueeze(0))
        with torch.no_grad():
            distribution = self.policy_forward(Variable(state))
        action = distribution.sample().data
        log_prob = self.log_prob(Variable(state), Variable(action))  # not volatile
        value = self.value_net(Variable(state))  # not volatile
        return {'action': action[0], 'log_prob': log_prob, 'value': value}

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
            self.replay_buffer.push(step_data)
            state = step_data['next_state'] ###or next state?
            if step_data['done']:
#                 self.reset()
                break
        self.update_running_rewards(episode_data['reward'])
        return episode_data


    def compute_loss(self):
        return NotImplementedError

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
            self.running_rewards = self.running_rewards * 0.99 + total_episode_rewards * 0.01
            self.running_moves = self.running_moves * 0.99 + episode_moves * 0.01

    def improve(self):
        batch = self.replay_buffer.sample()
        
        for sched in self.lr_scheduler:
            sched.step()
        
        for opt in self.optimizer.values():
            opt.zero_grad()
        
        losses = self.compute_loss(batch)
        
        for loss in losses:
            loss.backward()
        
        for opt in self.optimizer.values():
            opt.step()
            
        self.replay_buffer.clear()

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
                #TODO: ADD PPO OPTIM EPOCHS??
                self.improve()
                
            self.episode += 1
            
        print('Training complete')
            
        