import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .ppo import PPO

"""
Option-Critic trained with Proximal Policy Optimization
"""

class PPOC(PPO):
    
    def __init__(self):
        super(PPOC, self).__init__(config)