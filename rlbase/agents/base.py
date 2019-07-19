import torch
import torch.nn as nn
from networks import CNN, RNN, MLP, MLP3D
from envs import *

class BaseAgent:
    def __init__(self, config):
        self.config = config
    
    def save(self):
        return NotImplementedError
    
    def load(self, filename):
        return NotImplementedError
    
    def set_network(self):
        body = self.config.network_body(config)
        self.network = self.config.network_head(config, body)
    
    def set_env(self):
        self.env = self.config.env(config)
