from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import torch.nn.functional as F
from networks.heads import FullyConnectedHead, OptionCriticHead
from networks.bodies import FullyConnectedBody

experiment = ExperimentConfig(
    {'name': 'ppoc_fourrooms_minibatch',
     'base_dir': 'experiments/',
     'save_episode_data': True,
     'debug': True
    }
)

algorithm = OCConfig(
    {'option_eps': 0.1,
     'dc': 0.1, #deliberation cost
     'n_options': 4
    }
)

training = TrainingConfig(
    {'max_episode_length': 300,
     'max_episodes': 20000,
     'update_every': 20000,
     'lr_scheduler': StepLR,
     'minibatch_size': 50,
     'lr': .002,
     'ent_coeff': 0.1,
     'optim': Adam,
     'cuda': True,
     'device': 1
    }
)

actor_head = FCConfig(
    {'hdim': 64, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'out_activation': nn.Softmax(dim=0), # make sure softmax happens over the right dimension
     'architecture': OptionCriticHead,
     'outdim': None, # num actions
     'n_options': None
    }
)

option_actor_head =  FCConfig(
    {'hdim': 64, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'out_activation': nn.Softmax(dim=0), # make sure softmax happens over the right dimension
     'architecture': FullyConnectedHead,
     'outdim': None # num options
    }
)

critic_head = FCConfig(
    {'hdim': 64, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'out_activation': nn.ReLU(),
     'outdim': None, # num options
#      'n_options': None,
     'architecture': FullyConnectedHead
    }
)

termination_head = FCConfig(
    {'hdim': 64, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'out_activation': nn.Softmax(dim=0),
     'architecture': FullyConnectedHead,
     'outdim': None # num options
#      'n_options': None
    }
)

body = FCConfig(
    {'hdim': 64, 
     'nlayers': 1,
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

env = FourRoomsConfig()

config = Config(
    {'experiment': experiment, 
     'algorithm': algorithm, 
     'training': training, 
     'network': network, 
     'env': env
    }
)

