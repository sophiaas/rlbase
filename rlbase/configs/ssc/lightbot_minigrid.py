from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody, ConvolutionalBody

HDIM = 64

experiment = ExperimentConfig(
    {'name': 'ssc_lightbot_minigrid',
     'base_dir': 'experiments/',
     'save_episode_data': True,
     'log_interval': 100,
     'every_n_episodes': 100,
     'debug': True
    }
)

algorithm = SSCConfig(
    {'n_hl_actions': 10,
     'n_learning_stages': 10,
     'max_atoms': 20,
     'selection': 'choose_n',
     'selection_criterion': 1,
     'load_dir': 'experiments/sparse500000/sparse500000_ppo_lightbot_minigrid_fractal_cross_0_lr0.001/evaluate/',
#      'load_action_dir': 'experiments/ssc_lightbot_minigrid/',
     'gamma': 0.99,
     'tau': 0.95,
     'optim_epochs': 5,
     'clip': 0.1,
     'clip_norm': 40,
     'l2_reg': 1e-5,
     'anneal_epochs': True
    }
)

training = TrainingConfig(
    {'max_episode_length': 500000,
     'max_episodes': 10000,
     'update_every': 4096,
     'lr_scheduler': StepLR,
     'lr': 1e-3,
     'lr_gamma': 0.99,
     'lr_step_interval': 100,
     'minibatch_size': 256,
     'optim': Adam,
     'cuda': True,
     'device': 0
    }
)

policy_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1, #1
     'activation': nn.ReLU(),
     'out_activation': nn.Softmax(dim=-1),
     'architecture': FullyConnectedHead
    }
)

value_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1, #1
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
     'n_layers': 3,
     'activation': nn.ReLU(),
    }
)

body = ConvConfig({
    'n_layers': 3,
    'nlayers': 3,
    'conv_layers': [conv1, conv2, conv3],
    'fc_layers': [fc1],
    'architecture': ConvolutionalBody,
    'activation': nn.ReLU(),
    'hdim': HDIM
    }
)

""""""

network = NetworkConfig(
    {'heads': {'actor': policy_head, 'critic': value_head},
     'body': body
    }
)

env = LightbotMinigridConfig(
    {'puzzle_name': 'fractal_cross_1',
     'agent_view_size': 7,
     'toggle_ontop': False,
     'reward_fn': '100,-1,-1,-1'
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

