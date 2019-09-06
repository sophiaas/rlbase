from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, SGD
from networks.heads import FullyConnectedHead
from networks.bodies import FullyConnectedBody

HDIM = 128

experiment = ExperimentConfig(
    {'name': 'ppo_lightbot',
    }
)

algorithm = PPOConfig(
    {}
)

training = TrainingConfig(
    {
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

env = LightbotConfig(
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

