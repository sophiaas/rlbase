from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody

HDIM = 256

experiment = ExperimentConfig(
    {'name': 'ssc_hanoi',
     'base_dir': 'experiments/',
     'save_episode_data': True,
     'log_interval': 100,
     'every_n_episodes': 1,
     'debug': True
    }
)

algorithm = SSCConfig(
    {'n_hl_actions': 10,
     'n_learning_stages': 10,
     'max_atoms': 20,
     'selection': 'choose_n',
     'selection_criterion': 1,
#      'load_dir': 'experiments/ssc_hanoi/'
     'load_dir': 'experiments/betterlr_h_eplen500_me30000_hdim256/betterlr_h_eplen500_me30000_hdim256_ppo_hanoi_2disks_lr0.0005/evaluate/',
#      'load_action_dir': 'experiments/orig_10hl_ssc_hanoi/',
     'gamma': 0.99,
     'tau': 0.95,
     'clip': 0.1,
     'clip_norm': 40,
     'l2_reg': 1e-5
    }
)

training = TrainingConfig(
    {'max_episode_length': 500,
     'max_episodes': 10000,
     'update_every': 4096,
     'lr_scheduler': StepLR,
     'lr': 5e-4,
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

body = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 2, 
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

env = HanoiConfig(
    {'n_disks': 3,
     'n_pegs': 3,
     'initial_peg': None,
     'continual': True,
     'reward_fn': '100,-1'
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

