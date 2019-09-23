import os
import argparse
from agents import PPO, PPOC#, SSC
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='lightbot',
                    help='Name of config') 
parser.add_argument('--algo', type=str, default='ppo',
                    help='Algorithm') 
parser.add_argument('--name', type=str, default='',
                    help='Name to prepend to save dir') 
parser.add_argument('--puzzle', type=str, default=None,
                    help='puzzle name for Lightbot and LightbotMinigrid environments')
parser.add_argument('--n_disks', type=int, default=None,
                    help='number of disks for Hanoi environment')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--device', type=int, default=-1,
                    help='Device to run on')
parser.add_argument('--seed', type=int, default=0,
                    help='seed') 
parser.add_argument('--load_dir', type=str, default=None,
                    help='Directory to load pre-trained model data from, for SSC') 

args = parser.parse_args()

if args.algo == 'ppo':
    from configs.ppo import all_configs, all_post_processors
    agent = PPO

elif args.algo == 'ppoc':
    from configs.ppoc import all_configs, all_post_processors
    agent = PPOC
    
elif args.algo == 'ssc':
    from configs.ssc import all_configs
    agent = SSC
    
config = all_configs[args.config]
post_processor = all_post_processors[args.config]

if args.name:
    config.experiment.name = args.name + '_' + config.experiment.name
    
if args.puzzle:
    config.env.puzzle_name = args.puzzle
    config.experiment.name = config.experiment.name + '_' + args.puzzle

    config.training.max_episode_length = 500000
    config.training.max_timesteps = 500000
    config.training.lr_gamma = 0.99
    
if args.n_disks:
    config.env.n_disks = args.n_disks
    config.experiment.name = config.experiment.name + '_' + str(args.n_disks) + 'disks'

    if args.n_disks == 4:
        config.training.max_episode_length = 3000000
        config.training.max_timesteps = 3000000

    config.training.lr_gamma = 0.95

config.experiment.name += '{}steps'.format(config.training.max_timesteps)


config.training.lr = args.lr
config.experiment.name += '_lr{}'.format(args.lr)
config.experiment.name += '_seed{}'.format(args.seed)

config.experiment.base_dir += args.name + '/'

if args.device >= 0:
    config.training.device = args.device
else:
    config.training.device = 'cpu'

config = post_processor(config)

def main():
    model = agent(config)
    model.train()
            
if __name__ == '__main__':
    main()
