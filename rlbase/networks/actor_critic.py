import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
import gym

class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.config = config
        
        # Shared body
        self.obs_transform = self.config.network.init_body()
        self.actor, self.critic = self.config.network.init_heads()

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(self.config.training.device)
        x = self.obs_transform(state)
        
        if self.config.env.action_space == gym.spaces.Discrete:
            action_probs = self.actor(x)
            dist = Categorical(action_probs)
            
        elif self.config.env.action_space == gym.spaces.Box:
            action_mean = self.actor(x)
            cov_mat = torch.diag(self.config.training.action_var).to(self.config.training.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, state, log_prob
    
    
    def evaluate(self, state, action):
        x = self.obs_transform(state)
        
        if self.config.env.action_space == gym.spaces.Discrete:
            action_probs = self.actor(x)
            dist = Categorical(action_probs)
            
        elif self.config.env.action_space == gym.spaces.Box:
            action_mean = self.actor(x)
            cov_mat = torch.diag(self.config.training.action_var).to(self.config.training.device)
            dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(x)
        return action_logprobs, torch.squeeze(state_value), dist_entropy
