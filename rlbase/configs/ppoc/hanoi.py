from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, SGD
import torch.nn.functional as F
from networks.heads import FullyConnectedHead, OptionCriticHead
from networks.bodies import FullyConnectedBody

HDIM = 256

experiment = ExperimentConfig(
    {'name': 'ppoc_hanoi',
    }
)

algorithm = OCConfig(
    {
    }
)

training = TrainingConfig(
    {'max_episode_length': 500,
     'max_episodes': 30000,
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

body = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 2,
     'activation': nn.ReLU(),
     'out_activation': nn.ReLU(),
     'architecture': FullyConnectedBody,
     'indim': None # observation dim
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

env = HanoiConfig(
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

def post_process(config):
    # post processing
    if config.env.n_disks == 2:
        print('000')
    elif config.env.n_disks == 3:
        print('111')
    elif config.env.n_disks == 4:
        print('222')
    else:
        assert False
    return config

