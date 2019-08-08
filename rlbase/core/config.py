import argparse
import torch
from envs import FourRooms, Lightbot
import torch.nn as nn
# from agents import A2C, PPO
from agents import PPO

class BaseConfig(object):
    
    def set_attributes(self, kwargs=None):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
                
            
"""Algorithm Config"""

class AlgorithmConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.gamma = 0.99
        self.tau = 0.95
        self.set_attributes(kwargs)
        

class A2CConfig(AlgorithmConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'A2C'
        self.init = A2C
        self.set_attributes(kwargs)
        

class PPOConfig(AlgorithmConfig):

    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'PPO'
        self.init = PPO
        self.optim_epochs = 4
        self.value_iters = 1
        self.minibatch_size = 50
        self.clip = 0.2
        self.clip_norm = 40
        self.l2_reg = 1e-3
        self.anneal_epochs = True
        self.set_attributes(kwargs)
        
        
class OCConfig(PPOConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'PPOC'
        self.n_options = 4
        self.set_attributes(kwargs)
        
        
"""Training Config"""

class TrainingConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.optim = None
        self.weight_decay = 0
        self.lr = 0.002
        self.lr_scheduler = None
        self.max_episode_length = 100
        self.max_episodes = 20000
        self.update_every = 2000
        self.lr_gamma = 0.9
        self.cuda = True
        self.betas = (0.9, 0.999)
        self.device = 1
        self.set_attributes(kwargs)
        

class ContinualLearningConfig(TrainingConfig):

    def __init__(self, kwargs=None):
        super().__init__()
        return NotImplementedError
    
        
"""Experiment Config"""

class ExperimentConfig(BaseConfig):
    
     def __init__(self, kwargs=None):
        self.name = ""
        self.seed = 543
        self.log_interval = 20
        self.save_episode_data = False
        self.base_dir = 'experiments/'
        self.render = False
        self.resume = ""
        self.eval = False
        self.adapt = False
        self.debug = False 
        self.plot_granularity = 50 # place datapoints every n episodes
        self.set_attributes(kwargs)
        
        
"""Env Config"""

class EnvConfig(BaseConfig):

    def __init__(self, kwargs=None):
        self.allow_impossible = False
        self.continual = False
        self.random_init = True
        self.solved_reward = None
        self.set_attributes(kwargs)
        

class LightbotConfig(EnvConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'lightbot'
        self.init = Lightbot
        self.reward_fn = "10,01,-1,-1"
        self.puzzle_name = "debug1"
        self.set_attributes(kwargs)
        
    def init_env(self):
        return self.init(self)
        
class LightbotMinigridConfig(EnvConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'lightbot'
        self.init = Lightbot
        self.reward_fn = "10,10,-1,-1"
        self.puzzle_name = "cross"
        self.agent_view_size = 0
        self.toggle_ontop = True
        self.set_attributes(kwargs)
        
    def init_env(self):
        return self.init(self)
        

class HanoiConfig(EnvConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'hanoi'
        self.init = None #ENV
        self.ndiscs = 3
        self.npegs = 3
        self.initial_peg = None
        self.set_attributes(kwargs)
        
    def init_env(self):
        return self.init(self)
        
        
class FourRoomsConfig(EnvConfig):

    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'fourrooms'
        self.init = FourRooms #ENV
        self.size = 20
        self.set_attributes(kwargs)
        
    def init_env(self):
        self.init()
        

class MujocoConfig(EnvConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'mujoco'
        self.env = None #ENV
        self.set_attributes(kwargs)
        
        
"""Network Config"""

class NetworkConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.heads = {}
        self.body = None
        self.set_attributes(kwargs)
        
    def init_body(self):
        return self.body.architecture(self.body)
    
#     def init_heads(self, body):
#         heads = {}
#         for name, config in self.heads.items():
#             heads[name] = config.architecture(config, body)
#         return heads

    def init_heads(self, body):
        heads = []
        for name, config in self.heads.items():
            heads.append(config.architecture(config, body))
        return heads
    
class ActorCriticConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.latent_dim = 64
        self.in_dim = None
        self.out_dim = None
        self.activation = nn.Tanh
        self.set_attributes(kwargs)
    
    
class ConvLayerConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.kernel_size = None
        self.stride = None
        self.out_channels = None
        self.set_attributes(kwargs)
        
        
class ConvConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.nlayers = 3
        self.layers = [ConvLayerConfig()]
        self.set_attributes(kwargs)
        
        
class FCConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.hdim = 256
        self.nlayers = 3
        self.set_attributes(kwargs)
      
    
class LSTMConfig(BaseConfig):

    def __init__(self, kwargs=None):
        self.hdim = 256
        self.nlayers = 3
        self.set_attributes(kwargs)
    

"""Model Config"""

class Config(BaseConfig):

    def __init__(self, kwargs=None):
        self.env = None
        self.algorithm = None
        self.network = None
        self.experiment = None
        self.training = None
        self.set_attributes(kwargs)
        
