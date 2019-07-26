import argparse
import torch
from envs import FourRooms
from agents import A2C

class BaseConfig(object):
    
    def set_attributes(self, kwargs=None):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
            
#         self.parser = argparse.ArgumentParser()
        
#     def add_argument(self, *args, **kwargs):
#         self.parser.add_argument(*args, **kwargs)

#     def merge(self, config_dict=None):
#         if config_dict is None:
#             args = self.parser.parse_args()
#             config_dict = args.__dict__
#         for key in config_dict.keys():
#             setattr(self, key, config_dict[key])
            
            
"""Algorithm Config"""

class AlgorithmConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.gamma = 1.0
        self.set_attributes(kwargs)
        

class A2CConfig(AlgorithmConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'A2C'
        self.init = A2C
        self.set_attributes(kwargs)
        

class PPOConfig(A2CConfig):

    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'PPO'
        self.optim_epochs = 5
        self.value_iters = 1
        self.minibatch_size = 50
        self.clip = 0.1
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
        self.lr = 1e-5
        self.lr_scheduler = None
        self.max_episode_length = 100
        self.max_episodes = 20000
        self.update_every = 100
        self.lr_gamma = 0.9
        self.cuda = True
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
        self.log_interval = 10
        self.save_episode_data = False
        self.base_dir = 'experiments/'
        self.episode_data_dir = 'episode_data'
        self.render = False
        self.resume = ""
        self.eval = False
        self.adapt = False
        self.debug = False
        self.set_attributes(kwargs)
        
        
"""Env Config"""

class EnvConfig(BaseConfig):

    def __init__(self, kwargs=None):
        self.allow_impossible = False
        self.continual = False
        self.random_init = True
        self.set_attributes(kwargs)
        

class LightbotConfig(EnvConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'lightbot'
        self.init = None #ENV
        self.reward_fn = ""
        self.puzzle_name = ""
        self.agent_view_size = 0
        self.toggle_ontop = True
        self.set_attributes(kwargs)
        

class HanoiConfig(EnvConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'hanoi'
        self.init = None #ENV
        self.ndiscs = 3
        self.npegs = 3
        self.initial_peg = None
        self.set_attributes(kwargs)
        
        
class FourRoomsConfig(EnvConfig):

    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'fourrooms'
        self.init = FourRooms #ENV
        self.size = 20
        self.set_attributes(kwargs)
        

class MujocoConfig(EnvConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'mujoco'
        self.env = None #ENV
        self.set_attributes(kwargs)
        
        
"""Network Config"""

class NetworkConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.head = None
        self.body = None
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
        