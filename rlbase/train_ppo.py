import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
from envs import Lightbot
from agents import PPO
from core.config import *
from core.logger import Logger
from lightbot_config import config
from networks.actor_critic import ActorCritic

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def main():
    ppo = PPO(config)
    ppo.train()
            
if __name__ == '__main__':
    main()

