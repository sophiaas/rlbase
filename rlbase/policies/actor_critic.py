import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
import gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.config = config
        
        # TODO: Make separate bodies nicer
        self.obs_transform_actor = self.config.network.init_body()
        self.obs_transform_critic = self.config.network.init_body()

        self.actor, self.critic = self.config.network.init_heads()

    def forward(self):
        raise NotImplementedError
    
    def actor_forward(self, state, cutoff=None):
        x = self.obs_transform_actor(state)
        
        if type(self.config.env.action_space) == gym.spaces.Discrete:                
            action_probs = self.actor(x)
            if cutoff:
                action_probs = action_probs[:cutoff]
            dist = Categorical(action_probs)
            
        elif type(self.config.env.action_space) == gym.spaces.Box:
            action_mean = self.actor(x)
            cov_mat = torch.diag(self.config.training.action_var).to(self.config.training.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
        else:
            raise ValueError('env action space must be either gym.spaces.Discrete or gym.spaces.Box')
        return dist
    
    def critic_forward(self, state):
        x = self.obs_transform_critic(state)
        value = self.critic(x)
        value = torch.squeeze(value)
        return value
        
    def act(self, state, cutoff=None):
        dist = self.actor_forward(state, cutoff)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def evaluate(self, state, action):
        dist = self.actor_forward(state)
        value = self.critic_forward(state)
        logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return logprobs, value, dist_entropy
    
    #TODO: make the outputs of act and evaluate more sensible
