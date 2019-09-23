import argparse
import copy
import os
import pickle
from agents import SSC
import torch
import pprint

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default=None,
                    help='Directory containing model to load') 
parser.add_argument('--params_episode', type=int, default=4000,
                    help='Episode to load params from') 
parser.add_argument('--episode', type=int, default=10000,
                    help='Episode to load trajectories from') 
parser.add_argument('--name', type=str, default='',
                    help='Name to prepend to save dir') 
parser.add_argument('--puzzle', type=str, default=None,
                    help='puzzle name for Lightbot and LightbotMinigrid environments')
parser.add_argument('--n_disks', type=int, default=None,
                    help='number of disks for Hanoi environment')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--device', type=int, default=0,
                    help='Device to run on') 
parser.add_argument('--load_dir', type=str, default=None,
                    help='Directory to load pre-trained model data') 
parser.add_argument('--action_file', type=str, default=None,
                    help='name of file') 
parser.add_argument('--seed', type=int, default=None,
                    help='random seed') 
parser.add_argument('--config', type=str, default='lightbot_minigrid',
                    help='Name of config') 


args = parser.parse_args()

agent = SSC
from configs.ssc import all_configs
    
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
    
else:
    config.algorithm.load_action_dir = None
    
print(config.algorithm.load_action_dir)

def main():
    with open(args.model_dir+'config.p', 'rb') as f:
        checkpoint_config = pickle.load(f)

        agent = SSC(config)

        checkpoint = torch.load(os.path.join(args.model_dir,'checkpoints','episode_{}'.format(args.episode)))
        agent.policy.load_state_dict(checkpoint['policy'])

        agent.train()
            
if __name__ == '__main__':
    main()

