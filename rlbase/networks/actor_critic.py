import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.config = config
        
        # Shared body
        self.obs_transform = self.config.network.init_body()
        self.actor, self.critic = self.config.network.init_heads()

    def forward(self):
        raise NotImplementedError
        
#     def act(self, state, memory):
#         state = torch.from_numpy(state).float().to(self.config.device)
#         action_probs = self.actor(state)
#         dist = Categorical(action_probs)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         return action, state, log_prob
    
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(self.config.training.device)
        x = self.obs_transform(state)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, state, log_prob
    
    def evaluate(self, state, action):
        x = self.obs_transform(state)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(x)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
