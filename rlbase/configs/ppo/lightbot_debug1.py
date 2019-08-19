from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody

HDIM = 512

experiment = ExperimentConfig(
    {'name': 'ppo_lightbot_debug1',
     'base_dir': 'experiments/',
     'save_episode_data': True,
     'log_interval': 20,
     'debug': True
    }
)

algorithm = PPOConfig(
    {'clip': 0.1,
     'clip_norm': 40,
     'optim_epochs': 5,
     'l2_reg': 1e-5
    }
)

training = TrainingConfig(
    {'max_episode_length': 50,
     'max_episodes': 20000,
     'update_every': 3000,
     'lr_scheduler': StepLR,
     'lr': 1e-3, #1e-3
     'lr_gamma': 0.9,
     'weight_decay': 1e-5, #1e-5
     'minibatch_size': 50,
     'optim': Adam,
     'cuda': True,
     'device': 1,
     'gamma': 0.99 #0.9
    }
)

policy_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1, #1
     'activation': nn.ReLU(),
     'out_activation': nn.Softmax(dim=0),
     'architecture': FullyConnectedHead
    }
)

value_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1, #1
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

env = LightbotConfig(
    {'puzzle_name': 'debug1',
     'reward_fn': '10,10,-1,-1'
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

