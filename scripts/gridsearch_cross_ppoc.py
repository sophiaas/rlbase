import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = 1 
import argparse
from agents import PPO, PPOC
import torch
import multiprocessing
import itertools
import copy

parser = argparse.ArgumentParser(description='Lightbot')

parser.add_argument('--config', type=str, default='lightbot_cross',
                    help='Name of config') 
parser.add_argument('--algo', type=str, default='ppoc',
                    help='Algorithm') 
parser.add_argument('--name', type=str, default=None,
                    help='Name to prepend to save dir') 
parser.add_argument('--device', type=int, default=None,
                    help='Device to run on') 

args = parser.parse_args()

from configs.ppoc import all_configs
agent = PPOC
    

def get_chunks(iterable, chunks=12):
    lst = list(iterable)
    return [lst[i::chunks] for i in range(chunks)]

def worker(grid):
    for a in grid:
        config = copy.deepcopy(all_configs[args.config])
        config.training.max_episodes = 10000
        config.experiment.every_n_episodes = 100
        config.training.lr = a[0]
        config.training.lr_gamma = a[1]
        config.algorithm.gamma = a[2]
        config.algorithm.tau = a[3]
        config.training.lr_step_interval = a[4]
        config.algorithm.clip = a[5]
        config.algorithm.dc = a[6]
        config.experiment.name = 'lightbot_cross_ppoc_lr{}_lrg{}_rg{}_t{}_si{}_c{}_dc{}'.format(a[0], a[1], a[2], a[3], a[4], a[5], a[6])
        model = agent(config)

        model.train()

"""
GRID
"""
lrs = [1e-3, 3e-4, 1e-4]
lr_gammas = [0.85, 0.8, 0.9]
rl_gammas = [0.99, 0.95, 0.9]
taus = [0.99, 0.95, 0.9]
lr_step_intervals = [10, 20, 5]
ppo_clips = [0.1, 0.2]
dc = [0.1, 0.05, 0.01]


if __name__ == '__main__':
    jobs = []
    grid = itertools.product(lrs, lr_gammas, rl_gammas, taus, lr_step_intervals, ppo_clips, dc)
    chunked_args = get_chunks(grid, chunks=6)
    pool = multiprocessing.Pool()
    pool.map(worker, chunked_args)
    pool.close()
    pool.join()

