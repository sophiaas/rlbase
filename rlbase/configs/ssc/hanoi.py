from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody

HDIM = 256

experiment = ExperimentConfig(
    {'name': 'ssc_hanoi',
    }
)

algorithm = SSCConfig(
    {'n_hl_actions': 10,
     'n_learning_stages': 10,
     'max_atoms': 20,
     'selection': 'choose_n',
     'selection_criterion': 1,
     'load_dir': None,
     'load_action_dir': 'rlbase/action_dictionaries/',
    }
)

training = TrainingConfig(
    {'max_episode_length': 5000000,
     'max_timesteps': 5000000,
    }
)

policy_head = FCConfig(
    {'hdim': HDIM, 
     'nlayers': 1,
     'activation': nn.ReLU(),
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
    if config.env.n_disks == 2:
        print('000')
    elif config.env.n_disks == 3:
        print('111')
    elif config.env.n_disks == 4:
        print('222')
    else:
        assert False
    return config
