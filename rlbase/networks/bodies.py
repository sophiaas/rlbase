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

    def define_network(self):
        return NotImplementedError


class ConvolutionalBody(BaseBody):
    def __init__(self, config):
        super(ConvolutionalBody, self).__init__(config)
        self.activation = self.config.activation
        self.define_network()

    def define_network(self):
        self.conv_layers = nn.ModuleList([])
        self.fc_layers = nn.ModuleList([])
        for cl in self.config.conv_layers:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=cl.in_channels,
                    out_channels=cl.out_channels,
                    kernel_size=cl.kernel_size,
                    stride=cl.stride,
                )
            )
            if cl.pool:
                self.conv_layers.append(nn.MaxPool2d(2, 2))

        indim = self.config.conv_layers[-1].out_channels

        for fcl in self.config.fc_layers[:-1]:
            self.fc_layers.append(nn.Linear(indim, fcl.hdim))
            indim = fcl.hdim
        self.fc_layers.append(nn.Linear(indim, self.config.hdim))

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = x.transpose(1, 3).transpose(2, 3)
        for layer in self.conv_layers:
            x = self.activation(layer(x))
        x = x.view(-1, torch.tensor(x.shape[1:]).prod())
        for layer in self.fc_layers:
            x = self.activation(layer(x)).squeeze()
        return x


class FullyConnectedBody(BaseBody):
    def __init__(self, config):
        super(FullyConnectedBody, self).__init__(config)
        self.indim = config.indim
        self.hdim = config.hdim
        self.nlayers = config.nlayers
        self.activation = config.activation
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
        self.indim = config.indim
        self.hdim = config.hdim
        self.nlayers = config.nlayers
        self.define_network()

    def define_network(self):
        self.network = nn.LSTMCell(input_size=self.indim, hidden_size=self.hdim)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        hidden = torch.zeros(x.shape[0], self.hdim)
        out = torch.randn(x.shape[0], self.hdim)

        for i in range(x.shape[1]):
            hidden, out = self.network(x[:, i, :], (hidden, out))

        return out
