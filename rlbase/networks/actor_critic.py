import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.config = config
        
        # actor
        self.actor = nn.Sequential(
                nn.Linear(config.in_dim, config.latent_dim),
                config.activation(),
                nn.Linear(config.latent_dim, config.latent_dim),
                config.activation(),
                nn.Linear(config.latent_dim, config.out_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.critic = nn.Sequential(
                nn.Linear(config.in_dim, config.latent_dim),
                config.activation(),
                nn.Linear(config.latent_dim, config.latent_dim),
                config.activation(),
                nn.Linear(config.latent_dim, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(self.config.device) 
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
