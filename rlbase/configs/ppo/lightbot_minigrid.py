from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody, ConvolutionalBody

HDIM = 512

experiment = ExperimentConfig(
    {'name': 'ppo_lightbot_minigrid',
     'base_dir': 'experiments/',
     'save_episode_data': True,
     'log_interval': 100,
     'every_n_episodes': 1,
     'debug': True
    }
)

algorithm = PPOConfig(
    {'clip': 0.1,
     'clip_norm': 40,
     'optim_epochs': 5,
     'gamma': 0.9,
     'tau': 0.95
    }
)

training = TrainingConfig(
    {'max_episode_length': 100,
     'max_timesteps': 500000,
     'update_every': 4096,
     'lr_scheduler': StepLR,
     'lr': 1e-3,
     'lr_gamma': 0.85,
     'lr_step_interval': 20,
     'minibatch_size': 50,
     'optim': Adam,
     'cuda': True,
     'device': 0
    }
)

policy_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1,
     'out_activation': nn.Softmax(dim=-1),
     'architecture': FullyConnectedHead
    }
)

value_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1,
     'out_activation': None,
     'architecture': FullyConnectedHead,
     'outdim': 1
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

fc1 = FCConfig(
    {'hdim': HDIM,
     'n_layers': 1,
     'activation': nn.ReLU(),
    }
)

body = ConvConfig({
    'n_layers': 4, 
    'conv_layers': [conv1, conv2, conv3],
    'fc_layers': [fc1],
    'architecture': ConvolutionalBody,
    'activation': nn.ReLU(),
    'hdim': HDIM
    }
)

"""
"""

network = NetworkConfig(
    {'heads': {'actor': policy_head, 'critic': value_head},
     'body': body
    }
)

env = LightbotMinigridConfig(
    {'puzzle_name': 'fractal_cross_0',
     'agent_view_size': 7,
     'toggle_ontop': False,
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

