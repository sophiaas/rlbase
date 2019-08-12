import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.distributions import Categorical
import numpy as np

    
class OptionCritic(nn.Module):
    
    def __init__(self, config):
        super(OptionCritic, self).__init__()
        
        self.config = config
        self.n_options = config.algorithm.n_options
        self.device = self.config.training.device
        
        self.obs_transform = self.config.network.init_body()
        # actor = linear3d(hdim, n_options, n_actions)
        # critic = linear(hdim, n_options)
        # termination = linear(hdim, n_options)
        self.actor, self.option_actor, self.critic, self.termination = self.config.network.init_heads()
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, option):        
        state = torch.from_numpy(state).float().to(self.device)
        x = self.obs_transform(state)
        
        option_probs = self.option_actor(x)
        option_dist = Categorical(option_probs)
        if option is None:
            option = option_dist.sample()
        option_logprob = option_dist.log_prob(option)
        
        termination_probability = self.termination(x)[option.data]

#             option_dist = self.critic(x)
#             if np.random.uniform() > self.config.algorithm.option_eps:
#                 option = option_dist.argmax()
#             else:
#                 option = np.random.randint(self.config.algorithm.n_options)
            
        action_probs = self.actor(x, option)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)

        if termination_probability > torch.rand(1).to(self.device):
            terminate = True
            option = option_dist.sample()
        else:
            terminate = False
            
        return state, action, action_logprob, option, option_logprob, termination_probability, terminate
    
    def evaluate(self, state, action, option):
        x = self.obs_transform(state)        
        
        action_probs = self.actor(x, option).to(self.device) #Shouldn't have to put on cuda again
        action_dist = Categorical(action_probs)
        action_logprob = action_dist.log_prob(action)
        action_dist_entropy = action_dist.entropy()
        
        option_probs = self.option_actor(x)
        option_dist = Categorical(option_probs)
        
        option_dist_entropy = option_dist.entropy()
        
        option_value_full = torch.squeeze(self.critic(x))
        option_value = torch.cat([torch.index_select(a, 0, i) for a, i in zip(option_value_full, option)]) # TODO: do this in the linear3d network also

        option_logprob = option_dist.log_prob(option)
#         option_logprob_full = torch.zeros((state.shape[0], self.n_options)).to(self.device)
#         for o in range(self.n_options):
#             vec = torch.ones((state.shape[0]), dtype=torch.int64).to(self.device) * o
#             option_logprob_full[:, o] = option_dist.log_prob(vec)
        
#         termination_probability = self.termination(x)[option]
        termination_probability = torch.cat([torch.index_select(a, 0, i) for a, i in zip(self.termination(x), option)])
        
        return action_logprob, option_value, option_value_full, option_logprob, \
                option_probs, termination_probability, action_dist_entropy, \
                option_dist_entropy
