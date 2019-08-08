import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init


class OptionHead(nn.Module):
    __constants__ = ['bias']

    def __init__(self, config):
        super(OptionHead, self).__init__()

        self.weight = Parameter(torch.Tensor(config.n_options, config.latent_dim, 
                                             config.out_dim))
        self.bias = Parameter(torch.Tensor(config.n_options, config.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, option):
        if x.shape[0] > 1:
            out = torch.zeros((x.shape[0], self.weight.shape[2]))
            for i in range(x.shape[0]):
                out[i] = torch.matmul(input[i], self.weight[option[i]]) 
                + self.bias[option[i]]
        else:
            W = self.weight[option]   
            b = self.bias[option] 
            out = torch.matmul(input, W) + b         
        return out

    
class OptionCritic(nn.Module):
    
    def __init__(self, config):
        super(OptionHead, self).__init__()
        
        self.obs_transform = self.config.network.init_body()
        self.actor, self.critic = self.config.network.init_heads()
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, option):
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
        