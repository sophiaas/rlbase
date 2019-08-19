from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import torch.nn.functional as F
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody

HDIM = 256

experiment = ExperimentConfig(
    {'name': 'ppo_fourooms',
     'base_dir': 'experiments/',
     'save_episode_data': True,
     'debug': True
    }
)

algorithm = PPOConfig(
    {'gamma': 0.99,
     'tau': 0.95
    }
)

training = TrainingConfig(
    {'max_episode_length': 100,
     'max_episodes': 20000,
     'update_every': 100,
     'lr_scheduler': StepLR,
     'lr': 1e-3,
     'lr_gamma': 0.9,
     'optim': Adam,
     'cuda': True,
     'device': 1,
    }
)

policy_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'out_activation': nn.Softmax(dim=0),
     'architecture': FullyConnectedHead
    }
)

value_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'out_activation': None,
     'architecture': FullyConnectedHead,
     'outdim': 1
    }
)

body = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1,
     'activation': nn.ReLU(),
     'out_activation': nn.ReLU(),
     'architecture': FullyConnectedBody
    }
)

network = NetworkConfig(
    {'heads': {'actor': policy_head, 'critic': value_head},
     'body': body
    }
)

# network = ActorCriticConfig()

env = FourRoomsConfig()

config = Config(
    {'experiment': experiment, 
     'algorithm': algorithm, 
     'training': training, 
     'network': network, 
     'env': env
    }
)

