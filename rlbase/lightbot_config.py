from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import torch.nn.functional as F
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody

experiment = ExperimentConfig(
    {'name': 'ppo_lightbot_v2',
     'base_dir': 'experiments/',
     'save_episode_data': True,
     'debug': True
    }
)

algorithm = PPOConfig()

training = TrainingConfig(
    {'max_episode_length': 300,
     'max_episodes': 20000,
     'weight_decay': 0.9,
     'update_every': 2000,
     'lr_scheduler': StepLR,
     'lr': .002,
     'betas': (0.9, 0.999),
     'optim': Adam,
     'cuda': True,
     'device': 1
    }
)

policy_head = FCConfig(
    {'hdim': 64, 
     'nlayers': 1,
     'activation': F.tanh,
     'out_activation': F.softmax,
     'architecture': FullyConnectedHead
    }
)

value_head = FCConfig(
    {'hdim': 64, 
     'nlayers': 1,
     'activation': F.tanh,
     'out_activation': F.tanh,
     'architecture': FullyConnectedHead
    }
)

body = FCConfig(
    {'hdim': 64, 
     'nlayers': 1,
     'activation': F.tanh,
     'out_activation': F.tanh,
     'architecture': FullyConnectedBody
    }
)

# network = NetworkConfig(
#     {'heads': {'policy': policy_head, 'value': value_head},
#      'body': body
#     }
# )

network = ActorCriticConfig()

env = LightbotConfig(
    {'puzzle_name': 'debug1'
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

