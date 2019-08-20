import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = 1 
import argparse
from agents import PPO, PPOC
import torch
import multiprocessing
import itertools

parser = argparse.ArgumentParser(description='Lightbot')

parser.add_argument('--config', type=str, default='lightbot_cross',
                    help='Name of config') 
parser.add_argument('--algo', type=str, default='ppo',
                    help='Algorithm') 
parser.add_argument('--name', type=str, default=None,
                    help='Name to prepend to save dir') 
parser.add_argument('--device', type=int, default=None,
                    help='Device to run on') 

args = parser.parse_args()

if args.algo == 'ppo':
    from configs.ppo import all_configs
    agent = PPO

elif args.algo == 'ppoc':
    from configs.ppoc import all_configs
    agent = PPOC

else:
    raise ValueError('Specified algorithm is not yet implemented')
    
    

def get_chunks(iterable, chunks=12):
    lst = list(iterable)
    return [lst[i::chunks] for i in range(chunks)]

def worker(grid):
    for a in grid:
        config = all_configs[args.config]
        config.training.max_episodes = 7500
        config.experiment.every_n_episodes = 100
        config.training.lr = a[0]
        config.training.lr_gamma = a[1]
        config.algorithm.gamma = a[2]
        config.algorithm.tau = a[3]
        config.training.lr_step_interval = a[4]
        config.algorithm.clip = a[5]
        config.experiment.name += '_lr{}_lrg{}_rg{}_t{}_si{}_c{}'.format(a[0], a[1], a[2], a[3], a[4], a[5])
        model = agent(config)

        model.train()

"""
GRID
"""
lrs = [1e-3, 5e-4, 3e-4, 1e-4]
lr_gammas = [0.99, 0.9, 0.8]
rl_gammas = [0.99, 0.95, 0.9]
taus = [0.99, 0.95, 0.9]
lr_step_intervals = [1, 5]
ppo_clips = [0.1, 0.2]




if __name__ == '__main__':
    jobs = []
    grid = itertools.product(lrs, lr_gammas, rl_gammas, taus, lr_step_intervals, ppo_clips)
    chunked_args = get_chunks(grid, chunks=12)
    pool = multiprocessing.Pool()
    pool.map(worker, chunked_args)
    pool.close()
    pool.join()

