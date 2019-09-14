import numpy as np
import pandas as pd
import torch
import pickle
from agents import PPO, PPOC, SSC
from collections import defaultdict
import os


class Evaluator(object):
    
    def __init__(self, config):
        self.config = config
        self.n_eval_steps = config.n_eval_steps
        
        with open(config.model_dir+'/config.p', 'rb') as f:
            self.checkpoint_config = pickle.load(f)
        
        self.checkpoint_config.eval = config
        self.checkpoint_config.training.max_episodes = config.max_episodes
        self.checkpoint_config.training.max_episode_length = config.max_episode_length
        self.checkpoint_config.experiment.render = config.render
        self.checkpoint_config.experiment.save_episode_data = config.save_episode_data
        self.checkpoint_config.experiment.log_interval = config.log_interval
        
        print(self.checkpoint_config.algorithm.name)
            
        if self.checkpoint_config.algorithm.name == 'PPO':
            self.model = PPO(self.checkpoint_config)
        elif self.checkpoint_config.algorithm.name == 'PPOC':
            self.model = PPOC(self.checkpoint_config)
        elif self.checkpoint_config.algorithm.name == 'SSC':
            self.checkpoint_config.load_action_dir = config.model_dir
            self.model = SSC(self.checkpoint_config)
        else:
            return ValueError('Unknown model type')
        
        self.model.device = config.device
        
        checkpoint = torch.load(config.model_dir+'checkpoints/episode_{}'.format(config.episode))
        self.model.policy.load_state_dict(checkpoint['policy'])
        
        self.model.logger.logdir += 'evaluate/'
        self.model.logger.episodedir = self.model.logger.logdir + 'episodes/'

        os.makedirs(self.model.logger.logdir)
        if self.config.save_episode_data:
            os.makedirs(self.model.logger.episodedir)
 
        