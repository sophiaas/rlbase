import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


class BaseBody(nn.Module):    
    
    def __init__(self, config):
        super(BaseBody, self).__init__()
        self.config = config
        self.indim = config.indim
        self.hdim = config.hdim
        self.nlayers = config.nlayers
        self.activation = config.activation
    
    def define_network(self):
        return NotImplementedError

    
class ConvolutionalBody(BaseBody):
    
    def __init__(self, config):
        super(ConvolutionalBody, self).__init__(config)
        self.define_network()
        
    def define_network(self):
        self.layer_config = self.config.layer_config
        self.network = nn.ModuleList([])
        for i in range(self.nlayers):
            self.network.append(
                nn.Conv2d(
                    in_channels=self.indim, 
                    out_channels=self.layer_config[i]['out_channels'],
                    kernel_size=self.layer_config[i]['kernel_size'],
                    stride=self.layer_config[i]['stride']
                )
            )
    
    def forward(self, x):
        for layer in self.network:
            x = self.activation(layer(x))
        return x
        
                                                      
class FullyConnectedBody(BaseBody):
    
    def __init__(self, config):
        super(FullyConnectedBody, self).__init__(config)
        self.define_network()
        
    def define_network(self):
        self.network = nn.ModuleList([nn.Linear(self.indim, self.hdim)])
        for i in range(1, self.nlayers):
            self.network.append(nn.Linear(self.hdim, self.hdim))
            
    def forward(self, x):
        for layer in self.network:
            x = self.activation(layer(x))
        return x
      
    
class LSTMBody(BaseBody):
    
    def __init__(self, config):
        super(LSTMBody, self).__init__(config)
        self.define_network()
        
    def define_network(self):
        self.network = nn.LSTM(self.indim, self.hdim, self.nlayers)
        
    def forward(self, x):
        return self.network(x)
                