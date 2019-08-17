import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
import gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.config = config
        self.device = self.config.training.device
        
        # Shared body
        self.obs_transform = self.config.network.init_body()
        self.actor, self.critic = self.config.network.init_heads()

#     def forward(self, state):
#         action, state, log_prob = self.act(state, forward=True)
# #         raise NotImplementedError
    def forward(self, state):
        raise NotImplementedError

    def actor_forward(self, state):
        # generating data uses volatile states

        x = self.obs_transform(state)

        if type(self.config.env.action_space) == gym.spaces.Discrete:
            scores = self.actor(x)
            normalized_scores = scores - scores.max()
            probs = torch.exp(scores)
            probs = torch.clamp(probs, float(np.finfo(np.float32).eps), 1) # For numerical instabilities
            dist = Categorical(probs)
            
        elif type(self.config.env.action_space) == gym.spaces.Box:
            action_mean = self.actor(x)
            cov_mat = torch.diag(self.config.training.action_var).to(self.config.training.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
        return dist
        
    def select_action(self, state):
        dist = self.actor_forward(state)
        action = dist.sample()
        return action
    
    def log_prob(self, state, action):
        action_dist = self.actor_forward(state)
        log_prob = action_dist.log_prob(action)
        return log_prob
    
    def critic_forward(self, state):
        x = self.obs_transform(state)
        value = self.critic(x)
        return value
            
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        with torch.no_grad():
            action = self.select_action(state)
        log_prob = self.log_prob(state.requires_grad_(), action.float().requires_grad_())
        value = self.critic_forward(state)
        return action, log_prob, value, state
        
#     def act(self, state, forward=False):
#         #TODO: make this forward
# #             state = torch.from_numpy(state).float().to(self.config.training.device)
# #             state = torch.tensor(state, dtype=torch.float).to(self.config.training.device)

# #             state = torch.tensor(state, requires_grad=True, dtype=torch.float).to(self.config.training.device)
# #             print('tensor:', state)
# #             print('requires_grad:', state.requires_grad)

#         x = self.obs_transform(state)
        
#         if type(self.config.env.action_space) == gym.spaces.Discrete:
#             action_probs = self.actor(x)
#             dist = Categorical(action_probs)
            
#         elif type(self.config.env.action_space) == gym.spaces.Box:
#             action_mean = self.actor(x)
#             cov_mat = torch.diag(self.config.training.action_var).to(self.config.training.device)
#             dist = MultivariateNormal(action_mean, cov_mat)
            
#         else:
#             raise ValueError('env action space must be either gym.spaces.Discrete or gym.spaces.Box') 
        
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         value = self.critic(x)
#         return action, state, log_prob, value
    
    
    def evaluate(self, state, action):
        x = self.obs_transform(state)
        
        #TODO: consolidate this and the section in act()
        
        if type(self.config.env.action_space) == gym.spaces.Discrete:
            action_probs = self.actor(x)
            dist = Categorical(action_probs)
            
        elif type(self.config.env.action_space) == gym.spaces.Box:
            action_mean = self.actor(x)
            cov_mat = torch.diag(self.config.training.action_var).to(self.config.training.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
        else:
            raise ValueError('env action space must be either gym.spaces.Discrete or gym.spaces.Box') 
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(x)
        return action_logprobs, torch.squeeze(state_value), dist_entropy
