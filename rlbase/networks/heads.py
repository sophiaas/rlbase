import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from core import ReplayBuffer, Memory
from core.utils import cuda_if_needed


class BaseHead(nn.Module):    
    
    def __init__(self, config):
        head_config = config.network_head_config
        self.hdim = head_config.hdim
        self.action_dim = head_config.action_dim
        self.nlayers = head_config.nlayers
        self.activation = head_config.activation
    
    def define_network(self):
        return NotImplementedError
    

class FullyConnectedHead(BaseHead):
    
    def __init__(self, config, body):
        super().__init__(config)
        self.define_network()
        self.body = body
        
    def define_network(self):
        self.network = nn.ModuleList([])
        for i in range(self.nlayers-1):
            self.network.append(nn.Linear(hdim, hdim))
        self.network.append(nn.Linear(hdim, self.action_dim))
            
    def forward(self, x):
        x = self.body.forward(x)
        for layer in self.network:
            x = self.activation(layer(x))
        return x
    
    
class OptionCriticHead(BaseHead):
    
    def __init__(self, config, body):
        super().__init__(config)
        self.define_network()
        self.body = body
        self.n_options = self.config.n_options
        
    def define_network(self):
        self.network = nn.Linear3D(self.hdim, self.n_options, self.action_dim)
        #TODO: Multilayer
            
    def forward(self, x, option):
        x = self.body.forward(x)
        x = self.activation(self.network(x, option))
        return x
    
    


      