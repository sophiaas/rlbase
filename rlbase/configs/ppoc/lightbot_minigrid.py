from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, SGD
import torch.nn.functional as F
from networks.heads import FullyConnectedHead, OptionCriticHead
from networks.bodies import FullyConnectedBody, ConvolutionalBody

HDIM = 64

experiment = ExperimentConfig(
    {'name': 'ppoc_lightbot_minigrid',
    }
)

algorithm = OCConfig(
    {
    }
)

training = TrainingConfig(
    {'max_episode_length': 50000,
    }
)

actor_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'out_activation': nn.Softmax(dim=-1),
     'architecture': OptionCriticHead,
     'outdim': None, # num actions
     'n_options': None
    }
)

option_actor_head =  FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'out_activation': nn.Softmax(dim=-1),
     'architecture': FullyConnectedHead,
     'outdim': None # num options
    }
)

critic_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'outdim': None, # num options
     'architecture': FullyConnectedHead
    }
)

termination_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'out_activation': nn.Sigmoid(),
     'architecture': FullyConnectedHead,
     'outdim': None # num options
    }
)

"""Convolutional body """
conv1 = ConvLayerConfig(
    {'in_channels': 3,
     'kernel_size': 2,
     'stride': 1,
     'out_channels': 16,
     'activation': nn.ReLU(),
     'pool': True
    }
)

conv2 = ConvLayerConfig(
    {'in_channels': 16,
     'kernel_size': 2,
     'stride': 1,
     'out_channels': 32,
     'activation': nn.ReLU(),
     'pool': False
    }
)

conv3 = ConvLayerConfig(
    {'in_channels': 32,
     'kernel_size': 2,
     'stride': 1,
     'out_channels': 64,
     'activation': nn.ReLU(),
     'pool': False
    }
)

body = ConvConfig({
    'n_layers': 3, 
    'conv_layers': [conv1, conv2, conv3],
    'architecture': ConvolutionalBody,
    'activation': nn.ReLU(),
    'hdim': HDIM
    }
)


network = NetworkConfig(
    {'heads': {
        'actor': actor_head, 
        'option_actor': option_actor_head,
        'critic': critic_head, 
        'termination': termination_head},
     'body': body
    }
)

env = LightbotMinigridConfig(
    {
    }
)

config = Config(
    {'experiment': experiment, 
     'algorithm': algorithm, 
     'training': training, 
     'network': network, 
     'env': env
    }
)

