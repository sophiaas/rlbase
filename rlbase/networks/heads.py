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
    
    def __init__(self, config):
        super(OptionCriticHead, self).__init__(config)
        self.n_options = self.config.n_options
        self.outdim = self.config.outdim
        self.define_network()
        
    def define_network(self):
        self.network = Linear3D(self.hdim, self.n_options, self.outdim)
        #TODO: Multilayer
            
    def forward(self, x, option):
        x = self.config.out_activation(self.network(x, option))
        return x
    
# class OptionHead(nn.Module):
#     __constants__ = ['bias']

#     def __init__(self, config):
#         super(OptionHead, self).__init__()

#         self.weight = Parameter(torch.Tensor(config.n_options, config.hdim, 
#                                              config.out_dim))
#         self.bias = Parameter(torch.Tensor(config.n_options, config.outdim))
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#     def forward(self, x, option):
#         if x.shape[0] > 1:
#             out = torch.zeros((x.shape[0], self.weight.shape[2]))
#             for i in range(x.shape[0]):
#                 out[i] = torch.matmul(input[i], self.weight[option[i]]) 
#                 + self.bias[option[i]]
#         else:
#             W = self.weight[option]   
#             b = self.bias[option] 
#             out = torch.matmul(input, W) + b         
#         return out

    