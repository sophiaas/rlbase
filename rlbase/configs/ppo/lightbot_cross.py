from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody

HDIM = 512

experiment = ExperimentConfig(
    {'name': 'ppo_lightbot_cross',
     'base_dir': 'experiments/',
     'save_episode_data': True,
     'log_interval': 100,
     'every_n_episodes': 100,
     'debug': True
    }
)

algorithm = PPOConfig(
    {'clip': 0.1,
     'clip_norm': 40,
     'optim_epochs': 5,
     'l2_reg': 1e-5,
     'gamma': 0.95,  # MC: usually I see 0.99, but shouldn't matter that much.
     'tau': 0.95
    }
)

training = TrainingConfig(
    {'max_episode_length': 100,
     'max_episodes': 10000,
     'update_every': 100,
     'lr_scheduler': StepLR,
     'lr': 1e-3,
     'lr_gamma': 0.9,
     'lr_step_interval': 1,
     'weight_decay': 1e-5,
     'minibatch_size': 50,
     'optim': Adam,
     'cuda': True,
     'device': 0
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

env = LightbotConfig(
    {'puzzle_name': 'cross',
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

