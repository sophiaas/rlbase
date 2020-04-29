from core.config import *
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from networks.heads import FullyConnectedHead
from networks.bodies import LSTMBody

HDIM = 256

experiment = ExperimentConfig({"name": "ppo_hanoi_lstm", "num_steps_between_plot": 100})

algorithm = PPOConfig({})

training = TrainingConfig(
    {
        # 'max_episode_length': 500,
        # 'max_timesteps': 500000,
        "log_interval": 20
    }
)

policy_head = FCConfig(
    {
        "hdim": HDIM,
        "nlayers": 1,
        "activation": nn.ReLU(),
        "out_activation": nn.Softmax(dim=-1),
        "architecture": FullyConnectedHead,
    }
)

value_head = FCConfig(
    {
        "hdim": HDIM,
        "nlayers": 1,
        "out_activation": None,
        "architecture": FullyConnectedHead,
        "outdim": 1,
    }
)

body = LSTMConfig({"hdim": HDIM, "nlayers": 2, "architecture": LSTMBody})

network = NetworkConfig(
    {"heads": {"actor": policy_head, "critic": value_head}, "body": body}
)

env = HanoiConfig({})

config = Config(
    {
        "experiment": experiment,
        "algorithm": algorithm,
        "training": training,
        "network": network,
        "env": env,
    }
)


def post_process(config):
    # post processing
    if config.env.n_disks == 2:
        print("000")
    elif config.env.n_disks == 3:
        print("111")
    elif config.env.n_disks == 4:
        print("222")
    else:
        assert False
    return config
