import os
import argparse
from agents import SSC
import torch    
from configs.ssc import all_configs


parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='lightbot_minigrid',
                    help='Name of config') 
parser.add_argument('--name', type=str, default=None,
                    help='Name to prepend to save dir') 
parser.add_argument('--puzzle', type=str, default=None,
                    help='puzzle name for Lightbot and LightbotMinigrid environments')
parser.add_argument('--n_disks', type=int, default=None,
                    help='number of disks for Hanoi environment')
parser.add_argument('--device', type=int, default=0,
                    help='Device to run on') 
parser.add_argument('--load_dir', type=str, default=None,
                    help='Directory to load pre-trained model data') 
parser.add_argument('--action_file', type=str, default=None,
                    help='nam of filee') 
parser.add_argument('--seed', type=int, default=None,
                    help='random seed') 

args = parser.parse_args()

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
    
if args.seed is not None:
    config.experiment.seed = args.seed
    config.experiment.name = config.experiment.name + '_seed{}'.format(args.seed)
    
if args.load_dir is not None:
    config.algorithm.load_dir = args.load_dir
    
if args.action_file is not None:
    config.algorithm.load_action_dir += args.action_file + '/'
    
print(config.algorithm.load_action_dir)

def main():
    model = agent(config)
    model.train()
            
if __name__ == '__main__':
    main()
