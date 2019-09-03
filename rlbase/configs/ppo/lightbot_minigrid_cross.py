from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, SGD
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody, ConvolutionalBody

# HDIM = 512
HDIM = 64

experiment = ExperimentConfig(
    {'name': 'ppo_lightbot_minigrid_cross',
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
     'l2_reg': 1e-3,#1e-5,
     'gamma': 0.99,#0.9,
     'tau': 0.95
    }
)

training = TrainingConfig(
    {'max_episode_length': 500,  # 50 MC
     'max_episodes': 10000,
     'update_every': 8,# 8 rougly 4096/500,  # 100 MC. Here we are updating every 8 episodes, where each episode has a maximum length of 500. 
     'lr_scheduler': StepLR,
     'lr': 4e-5, # 3e-5 MC
     'lr_gamma': 0.99,
     'lr_step_interval': 1,
     'weight_decay': 1e-5, #1e-5
     'minibatch_size': 256, #50,
     'optim': SGD, #Adam,
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

# fc2 = FCConfig(
#     {'hdim': HDIM,
#      'n_layers': 1,
#      'activation': nn.ReLU(),
#     }
# )

body = ConvConfig({
    'n_layers': 4, 
    'conv_layers': [conv1, conv2, conv3],
    'architecture': ConvolutionalBody,
    'activation': nn.ReLU(),
    'hdim': HDIM
    }
)


# class CNN(nn.Module):
#     # from rl-starter-files
#     def __init__(self, n, m):
#         super(CNN, self).__init__()
#         self.image_conv = nn.Sequential(
#             nn.Conv2d(3, 16, (2, 2)),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Conv2d(16, 32, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, (2, 2)),
#             nn.ReLU()
#         )
#         self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
#     def forward(self, x):
#         # (bsize, H, W, C) --> (bsize, C, H, W)
#         x = x.transpose(1, 3).transpose(2, 3)
#         x = self.image_conv(x)
#         x = x.reshape(x.shape[0], -1)
#         return x


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
     'reward_fn': '10,10,-1,-1'#'100,100,-1,-1'
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

