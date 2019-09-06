import argparse
import torch
from envs import FourRooms, Lightbot, Hanoi
import sys
sys.path.append('../')
from gym_minigrid.envs.lightbot import LightbotEnv as LightbotMiniGrid 
from gym_minigrid.envs.empty import EmptyRandomEnv5x5
from gym_minigrid.wrappers import ImgObsWrapper
import torch.nn as nn
from agents import PPO
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR


class BaseConfig(object):
    
    def set_attributes(self, kwargs=None):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
                
            
"""Algorithm Config"""

class AlgorithmConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.gamma = 0.99
        self.tau = 0.95 #lambda
        self.set_attributes(kwargs)
        

class PPOConfig(AlgorithmConfig):

    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'PPO'
        self.optim_epochs = 5
        self.clip = 0.1
        self.clip_norm = 40
        self.l2_reg = 1e-5
        self.anneal_epochs = True
        self.set_attributes(kwargs)
    
        
class OCConfig(PPOConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'PPOC'
        self.n_options = 4
        self.dc = 0.1
        self.block_ent_penalty = False
        self.sample_blocks = True
        self.n_block_samples = 10000
        self.block_ent_coeff = 0.1
        self.max_block_length = 8
        self.set_attributes(kwargs)

        
"""Training Config"""

class TrainingConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.optim = SGD
        self.weight_decay = 1e-5
        self.lr = 4e-5
        self.lr_scheduler = StepLR
        self.betas = (0.9, 0.999)
        self.minibatch_size = 256
        self.max_episode_length = 100
        self.max_episodes = 10000
        self.update_every = 4096
        self.lr_gamma = 0.99
        self.lr_step_interval = 100
        self.action_var = 0.5 # for continuous action spaces
        self.cuda = True
        self.device = 0
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
        self.every_n_episodes = 100
        self.save_episode_data = True
        self.base_dir = 'experiments/'
        self.render = False
        self.resume = ""
        self.eval = False
        self.adapt = False
        self.debug = True 
        self.plot_granularity = 1 # place datapoints every n episodes
        self.set_attributes(kwargs)
        
        
"""Env Config"""

class EnvConfig(BaseConfig):

    def __init__(self, kwargs=None):
        self.allow_impossible = False
        self.continual = False
        self.random_init = True
        self.solved_reward = None
        self.set_attributes(kwargs)
        
    def init_env(self):
        return NotImplementedErrors
        

class LightbotConfig(EnvConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'lightbot'
        self.init = Lightbot
        self.reward_fn = "10,10,-1,-1"
        self.puzzle_name = "cross"
        self.set_attributes(kwargs)
        
    def init_env(self):
        env = self.init(self)
        self.action_space = env.action_space
        self.action_dim = env.action_space.n
        self.obs_dim = env.observation_space.n
        return env
        
class LightbotMinigridConfig(EnvConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'lightbot_minigrid'
        self.init = LightbotMiniGrid
        self.reward_fn = "10,10,-1,-1"
        self.puzzle_name = None
        self.agent_view_size = 7
        self.toggle_ontop = False
        self.agent_start_pos = None
        self.agent_start_dir = None
        self.max_steps = 500
        self.set_attributes(kwargs)
        
    def init_env(self):
        env = ImgObsWrapper(self.init(self))
        env.reset()
        print('agent pos: {}'.format(env.agent_pos))
        self.action_space = env.action_space
        self.action_dim = env.action_space.n
        self.obs_dim = env.observation_space.shape
        return env

class HanoiConfig(EnvConfig):
    
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'hanoi'
        self.init = Hanoi
        self.n_disks = None
        self.n_pegs = 3
        self.initial_peg = None
#         self.random_init = True
        self.continual = True
        self.reward_fn = "100,-1"
        self.set_attributes(kwargs)
        
    def init_env(self):
        env = self.init(self)
        env.reset()
        self.action_space = env.action_space
        self.action_dim = env.action_space.n
        self.obs_dim = env.observation_space.n
        return env
        
        
class FourRoomsConfig(EnvConfig):

    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'fourrooms'
        self.init = FourRooms #ENV
        self.size = 20
        self.set_attributes(kwargs)

        
    def init_env(self):
        env = self.init()
        self.action_space = env.action_space
        self.action_dim = env.action_space.n
        self.obs_dim = env.observation_space.n
        return env

class MinigridEmptyRandom5x5(EnvConfig):
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = 'minigrid_random_empty_5x5'
        self.init = EmptyRandomEnv5x5
        self.set_attributes(kwargs)

    def init_env(self):
        env = ImgObsWrapper(self.init())
        env.reset()
        print('agent pos: {}'.format(env.agent_pos))
        self.action_space = env.action_space
        self.action_dim = env.action_space.n
        self.obs_dim = env.observation_space.shape
        return env
        

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

    def init_heads(self):
        heads = []
        for name, config in self.heads.items():
            heads.append(config.architecture(config))
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
        self.conv_layers = [ConvLayerConfig()]
        self.fc_layers = [FCConfig()]
        self.set_attributes(kwargs)
        
        
class FCConfig(BaseConfig):
    
    def __init__(self, kwargs=None):
        self.hdim = 256
        self.nlayers = 3
        self.activation = None
        self.out_activation=None
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
        
