import torch
import torch.nn as nn
from .architectures import Linear3D


class BaseHead(nn.Module):
    def __init__(self, config):
        super(BaseHead, self).__init__()
        self.config = config
        self.hdim = config.hdim
        self.outdim = config.outdim
        self.nlayers = config.nlayers
        self.activation = config.activation
        self.out_activation = config.out_activation

    def define_network(self):
        return NotImplementedError


class FullyConnectedHead(BaseHead):
    def __init__(self, config):
        super(FullyConnectedHead, self).__init__(config)
        self.define_network()

    def define_network(self):
        self.network = nn.ModuleList([])
        for i in range(self.nlayers - 1):
            self.network.append(nn.Linear(self.hdim, self.hdim))
        self.network.append(nn.Linear(self.hdim, self.outdim))

    def forward(self, x):
        if len(self.network) > 1:
            for layer in self.network[: len(self.network) - 1]:
                x = self.activation(layer(x))
        else:
            if self.config.out_activation is not None:
                x = self.config.out_activation(self.network[-1](x))
            else:
                x = self.network[-1](x)
        return x


class OptionCriticHead(BaseHead):
    def __init__(self, config):
        super(OptionCriticHead, self).__init__(config)
        self.n_options = self.config.n_options
        self.outdim = self.config.outdim
        self.define_network()

    def define_network(self):
        self.network = Linear3D(self.hdim, self.n_options, self.outdim)
        # TODO: Multilayer

    def forward(self, x, option):
        x = self.config.out_activation(self.network(x, option))
        return x
