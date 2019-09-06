from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, SGD
import torch.nn.functional as F
from networks.heads import FullyConnectedHead, OptionCriticHead
from networks.bodies import FullyConnectedBody, ConvolutionalBody

HDIM = 64

experiment = ExperimentConfig(
    {'name': 'ppoc_lightbot_minigrid',
     'base_dir': 'experiments/',
     'save_episode_data': True,
     'log_interval': 20,
     'every_n_episodes': 1,
     'debug': True
    }
)

algorithm = OCConfig(
    {'dc': 0.1, #deliberation cost
     'n_options': 4,
     'gamma': 0.99,
     'tau': 0.95
    }
)

training = TrainingConfig(
    {'max_episode_length': 500,
     'max_episodes': 20000,
     'update_every': 4096,
     'lr_scheduler': StepLR,
     'lr': 4e-5,  # TODO MC
     'lr_gamma': 0.99,  # TODO MC
     'lr_step_interval': 100,  # TODO MC
     'weight_decay': 1e-5, #1e-5
     'minibatch_size': 256, #50, for now let's not anneal
     'optim': SGD, # TODO MC
     'cuda': True,
     'device': 0
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

# body = FCConfig(
#     {'hdim': HDIM, 
#      'nlayers': 1,
#      'activation': nn.ReLU(),
#      'out_activation': nn.ReLU(),
#      'architecture': FullyConnectedBody,
#      'indim': None # observation dim
#     }
# )

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

# fc1 = FCConfig(
#     {'hdim': 128,
#      'n_layers': 1,
#      'activation': nn.ReLU(),
#     }
# )
# fc1 = FCConfig(
#     {'hdim': HDIM,
#      'n_layers': 1,
#      'activation': nn.ReLU(),
#     }
# )

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

