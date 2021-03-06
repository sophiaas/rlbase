import argparse
import torch
from envs import FourRooms, Lightbot, Hanoi
import sys

sys.path.append("../")
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


"""Evaluator Config"""


class EvalConfig(BaseConfig):
    def __init__(self, kwargs=None):
        self.n_eval_steps = 2000
        self.model_dir = None
        self.episode = 20000
        self.max_episodes = 10000
        self.max_episode_length = 500
        self.render = False
        self.device = 0
        self.save_episode_data = True
        self.log_interval = 20
        self.set_attributes(kwargs)


"""Algorithm Config"""


class AlgorithmConfig(BaseConfig):
    def __init__(self, kwargs=None):
        self.gamma = 0.99
        self.tau = 0.95  # lambda
        self.set_attributes(kwargs)


class PPOConfig(AlgorithmConfig):
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = "PPO"
        self.optim_epochs = 5
        self.clip = 0.1
        self.clip_norm = 40
        self.l2_reg = 1e-5
        self.gamma = 0.99
        self.tau = 0.95
        self.anneal_epochs = True
        self.set_attributes(kwargs)


class OCConfig(PPOConfig):
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = "PPOC"
        self.n_options = 4
        self.dc = 0.1

        self.set_attributes(kwargs)


class SSCConfig(PPOConfig):
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = "SSC"
        self.n_hl_actions = 4
        self.n_learning_stages = 4
        self.max_actions = 40
        self.max_atoms = 20
        self.atom_length = 2
        self.sparsity = 0.9
        self.selection = "choose_n"
        self.selection_criterion = 1
        self.count_criterion = None
        self.reward_weighted = False
        self.reward_coeff = False
        self.load_dir = None
        self.load_action_dir = None
        self.set_attributes(kwargs)


"""Additional Costs and Regularizers"""


class CostConfig(BaseConfig):
    def __init__(self, kwargs=None):
        self.block_entropy = False
        self.mutual_information = False
        self.set_attributes(kwargs)


class BlockEntropy(CostConfig):
    def __init__(self, kwargs=None):
        self.block_entropy = True
        self.sample_blocks = True
        self.n_samples = 10000
        self.block_ent_coeff = 0.1
        self.max_block_length = 8
        self.set_attributes(kwargs)


class MutualInformation(CostConfig):
    def __init__(self, kwargs=None):
        self.mutual_information = True
        self.set_attributes(kwargs)


"""Training Config"""


class TrainingConfig(BaseConfig):
    def __init__(self, kwargs=None):
        self.optim = Adam
        self.weight_decay = 1e-5
        self.lr = 1e-4
        self.lr_scheduler = StepLR
        self.betas = (0.9, 0.999)
        self.minibatch_size = 256
        self.max_episode_length = 500
        self.max_timesteps = 1000000
        self.update_every = 4096
        self.lr_gamma = 0.95
        self.lr_step_interval = 100
        self.action_var = 0.5  # for continuous action spaces
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
        self.log_interval = 100
        self.every_n_episodes = 100
        self.num_steps_between_plot = 1000
        self.save_episode_data = True
        self.base_dir = "experiments/"
        self.render = False
        self.resume = ""
        self.eval = False
        self.adapt = False
        self.debug = True
        self.plot_granularity = 1  # place datapoints every n episodes
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
        self.name = "lightbot"
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


class HanoiConfig(EnvConfig):
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = "hanoi"
        self.init = Hanoi
        self.n_disks = None
        self.n_pegs = 3
        self.initial_peg = None
        self.continual = True
        self.reward_fn = "100,-1"
        self.set_attributes(kwargs)

    def init_env(self):
        env = self.init(self)
        env.reset()
        self.action_space = env.action_space
        self.action_dim = env.action_space.n
        self.obs_dim = env.num_pegs
        return env


class FourRoomsConfig(EnvConfig):
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = "fourrooms"
        self.init = FourRooms
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
        self.name = "minigrid_random_empty_5x5"
        self.init = EmptyRandomEnv5x5
        self.set_attributes(kwargs)

    def init_env(self):
        env = ImgObsWrapper(self.init())
        env.reset()
        print("agent pos: {}".format(env.agent_pos))
        self.action_space = env.action_space
        self.action_dim = env.action_space.n
        self.obs_dim = env.observation_space.shape
        return env


class MujocoConfig(EnvConfig):
    def __init__(self, kwargs=None):
        super().__init__()
        self.name = "mujoco"
        self.env = None
        self.set_attributes(kwargs)


"""Network Config"""


class NetworkConfig(BaseConfig):
    def __init__(self, kwargs=None):
        self.heads = {}
        self.body = None
        self.set_attributes(kwargs)

    def init_body(self):
        return self.body.architecture(self.body)

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
        self.out_activation = None
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
        self.costs = CostConfig()
        self.set_attributes(kwargs)
