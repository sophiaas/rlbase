import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
import gym

class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.config = config
        
        # Shared body
#         self.obs_transform_actor = self.config.network.init_body()
#         self.obs_transform_critic = self.config.network.init_body()
        self.obs_transform = self.config.network.init_body()
        
        
        self.actor, self.critic = self.config.network.init_heads()

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(self.config.training.device)
#         x = self.obs_transform_actor(state)
        x = self.obs_transform(state)
        
        if type(self.config.env.action_space) == gym.spaces.Discrete:
            action_probs = self.actor(x)
            dist = Categorical(action_probs)
            
        elif type(self.config.env.action_space) == gym.spaces.Box:
            action_mean = self.actor(x)
            cov_mat = torch.diag(self.config.training.action_var).to(self.config.training.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
        else:
            raise ValueError('env action space must be either gym.spaces.Discrete or gym.spaces.Box') 
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, state, log_prob
    
    
    def evaluate(self, state, action):
#         x = self.obs_transform_actor(state)
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

#         y = self.obs_transform_actor(state)
        state_value = self.critic(x)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

# import torch
# import torch.nn as nn
# from torch.distributions import Categorical, MultivariateNormal
# import gym

# class ActorCritic(nn.Module):
#     def __init__(self, config):
#         super(ActorCritic, self).__init__()
#         self.config = config
        
#         # Shared body
# #         self.actor_obs_transform = self.config.network.init_body()
# #         self.critic_obs_transform = self.config.network.init_body()
#         self.obs_transform = self.config.network.init_body()
        
#         self.actor, self.critic = self.config.network.init_heads()

#     def forward(self, state):
#         raise NotImplementedError
    
#     def act(self, state):
#         #TODO: make this forward
#         state = torch.tensor(state, dtype=torch.float).to(self.config.training.device)
        
#         x = self.obs_transform(state)
        
#         if type(self.config.env.action_space) == gym.spaces.Discrete:
#             action_probs = self.actor(x)
#             dist = Categorical(action_probs)
            
#         elif type(self.config.env.action_space) == gym.spaces.Box:
#             action_mean = self.actor(x1)
#             cov_mat = torch.diag(self.config.training.action_var).to(self.config.training.device)
#             dist = MultivariateNormal(action_mean, cov_mat)
            
#         else:
#             raise ValueError('env action space must be either gym.spaces.Discrete or gym.spaces.Box') 
        
#         action = dist.sample()
#         log_prob = dist.log_prob(action)

#         return action, state, log_prob
    
    
#     def evaluate(self, state, action):
#         x = self.obs_transform(state)
        
#         #TODO: consolidate this and the section in act()
        
#         if type(self.config.env.action_space) == gym.spaces.Discrete:
#             action_probs = self.actor(x)
#             dist = Categorical(action_probs)
            
#         elif type(self.config.env.action_space) == gym.spaces.Box:
#             action_mean = self.actor(x)
#             cov_mat = torch.diag(self.config.training.action_var).to(self.config.training.device)
#             dist = MultivariateNormal(action_mean, cov_mat)
            
#         else:
#             raise ValueError('env action space must be either gym.spaces.Discrete or gym.spaces.Box') 
        
#         action_logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
        
#         state_value = self.critic(x)
#         return action_logprobs, torch.squeeze(state_value), dist_entropy
