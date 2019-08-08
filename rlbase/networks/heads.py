import torch
import torch.nn as nn
from .architectures import Linear3D

class BaseHead(nn.Module):    
    
    def __init__(self, config):
        super(BaseHead, self).__init__()
#         head_config = config.network.head
        self.config = config
        self.hdim = config.hdim
        self.outdim = config.outdim
        self.nlayers = config.nlayers
        self.activation = config.activation
        self.out_activation = config.out_activation
    
    def define_network(self):
        return NotImplementedError
    

# class FullyConnectedHead(BaseHead):
    
#     def __init__(self, config, body):
#         super(FullyConnectedHead, self).__init__(config)
#         self.define_network()
#         self.body = body
        
#     def define_network(self):
#         self.network = nn.ModuleList([])
#         for i in range(self.nlayers-1):
#             self.network.append(nn.Linear(self.hdim, self.hdim))
#         self.network.append(nn.Linear(self.hdim, self.outdim))
            
#     def forward(self, x):
#         x = self.body.forward(x)
#         for layer in self.network:
#             x = self.activation(layer(x))
#         return x
    

# class FullyConnectedHead(BaseHead):
    
#     def __init__(self, config, body):
#         super(FullyConnectedHead, self).__init__(config)
#         self.define_network()
#         self.body = body
        
#     def define_network(self):
#         self.network = nn.ModuleList([])
#         for i in range(self.nlayers-1):
#             self.network.append(nn.Linear(self.hdim, self.hdim))
#         self.network.append(nn.Linear(self.hdim, self.outdim))
            
#     def forward(self, x):
#         x = self.body(x)
#         for layer in self.network[:len(self.network)-1]:
#             x = self.activation(layer(x))
#         x = self.config.out_activation(self.network[-1](x))
#         return x
    
class FullyConnectedHead(BaseHead):
    
    def __init__(self, config):
        super(FullyConnectedHead, self).__init__(config)
        self.define_network()
        
    def define_network(self):
        self.network = nn.ModuleList([])
        for i in range(self.nlayers-1):
            self.network.append(nn.Linear(self.hdim, self.hdim))
        self.network.append(nn.Linear(self.hdim, self.outdim))
            
    def forward(self, x):
        for layer in self.network[:len(self.network)-1]:
            x = self.activation(layer(x))
        x = self.config.out_activation(self.network[-1](x))
        return x
    
    
class OptionCriticHead(BaseHead):
    
    def __init__(self, config, body):
        super(OptionCriticHead, self).__init__(config)
        self.define_network()
        self.body = body
        self.n_options = self.config.n_options
        
    def define_network(self):
        self.network = Linear3D(self.hdim, self.n_options, self.action_dim)
        #TODO: Multilayer
            
    def forward(self, x, option):
        x = self.body.forward(x)
        x = self.activation(self.network(x, option))
        return x
    