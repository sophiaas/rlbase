import os
import argparse
from agents import PPO, PPOC, SSC
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='lightbot',
                    help='Name of config') 
parser.add_argument('--algo', type=str, default='ppo',
                    help='Algorithm') 
parser.add_argument('--name', type=str, default=None,
                    help='Name to prepend to save dir') 
parser.add_argument('--puzzle', type=str, default=None,
                    help='puzzle name for Lightbot and LightbotMinigrid environments')
parser.add_argument('--n_disks', type=int, default=None,
                    help='number of disks for Hanoi environment')
parser.add_argument('--device', type=int, default=0,
                    help='Device to run on') 
parser.add_argument('--load_dir', type=str, default=None,
                    help='Directory to load pre-trained model data from, for SSC') 

args = parser.parse_args()

if args.algo == 'ppo':
    from configs.ppo import all_configs
    agent = PPO

elif args.algo == 'ppoc':
    from configs.ppoc import all_configs
    agent = PPOC
    
elif args.algo == 'ssc':
    from configs.ssc import all_configs
    agent = SSC
    
config = all_configs[args.config]

if args.name is not None:
    config.experiment.name = args.name + '_' + config.experiment.name
    
if args.puzzle:
    config.env.puzzle_name = args.puzzle
    config.experiment.name = config.experiment.name + '_' + args.puzzle
    
if args.n_disks:
    config.env.n_disks = args.n_disks
    config.experiment.name = config.experiment.name + '_' + str(args.n_disks) + 'disks'
    
if args.device is not None:
    config.training.device = args.device

def main():
    model = agent(config)
    model.train()
            
if __name__ == '__main__':
    main()
