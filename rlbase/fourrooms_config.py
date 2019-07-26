from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import torch.nn.functional as F
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody

experiment = ExperimentConfig(
    {'name': 'test_fourrooms_2',
     'base_dir': 'experiments/',
     'save_episode_data': True,
     'debug': True
    }
)

algorithm = A2CConfig(
    {'gamma': 0.9
    }
)

training = TrainingConfig(
    {'max_episode_length': 100,
     'max_episodes': 20000,
     'update_every': 100,
     'lr_scheduler': StepLR,
     'lr': 1e-5,
     'optim': Adam,
     'cuda': True
    }
)

policy_head = FCConfig(
    {'hdim': 256, 
     'nlayers': 1,
     'activation': F.relu,
     'architecture': FullyConnectedHead
    }
)

value_head = FCConfig(
    {'hdim': 256, 
     'nlayers': 1,
     'activation': F.relu,
     'architecture': FullyConnectedHead
    }
)

body = FCConfig(
    {'hdim': 256, 
     'nlayers': 2,
     'activation': F.relu,
     'architecture': FullyConnectedBody
    }
)

network = NetworkConfig(
    {'policy_head': policy_head,
     'value_head': value_head,
     'body': body
    }
)

env = FourRoomsConfig(
    {'action_dim': 4
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

