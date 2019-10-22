from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, SGD
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody, ConvolutionalBody

HDIM = 64

experiment = ExperimentConfig(
    {'name': 'ssc_lightbot_minigrid',
    }
)

algorithm = SSCConfig(
    {'n_hl_actions': 8,
     'n_learning_stages': 8,
     'max_actions': 40,
     'max_atoms': 20,
     'selection': 'choose_n',
     'selection_criterion': 1,
     'load_dir': None,
     'load_action_dir': 'rlbase/action_dictionaries/',
    }
)

training = TrainingConfig(
    {'max_episode_length': 500,
     'max_timesteps': 5000000
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

body = ConvConfig({
    'n_layers': 3, 
    'conv_layers': [conv1, conv2, conv3],
    'architecture': ConvolutionalBody,
    'activation': nn.ReLU(),
    'hdim': HDIM
    }
)

network = NetworkConfig(
    {'heads': {'actor': policy_head, 'critic': value_head},
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

def post_process(config):
    # post processing
    if config.env.puzzle_name == 'fractal_cross_0':
        print('000')
    elif config.env.puzzle_name == 'fractal_cross_1':
        print('111')
    elif config.env.puzzle_name == 'fractal_cross_2':
        print('222')
    else:
        assert False
    return config


