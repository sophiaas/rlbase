import argparse
import copy
import os
import pickle
from agents import SSC
import torch
from configs.ssc import all_configs
import pprint
import torch
import torch.nn as nn

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
parser.add_argument('--lr', type=float, default=1e-4,
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

    
config = all_configs[args.config]

def produce_transfer_config(config):
    if args.puzzle:
        config.experiment.name += '_{}-to-{}_from-ep{}_1000000'.format(
            config.env.puzzle_name, args.puzzle, args.params_episode)
        config.env.puzzle_name = args.puzzle
        config.training.lr = 1e-4
        config.training.lr_gamma = 0.99
        if args.puzzle == 'fractal_cross_0-1':
            config.training.max_episode_length = 1000000
            config.training.max_timesteps = 1000000
        elif args.puzzle == 'fractal_cross_0-2':
            config.training.max_episode_length = 3000000
            config.training.max_timesteps = 3000000
            
    if args.load_dir:
        config.algorithm.n_hl_actions = 10
        config.algorithm.n_learning_stages=10

    if args.n_disks:
        config.experiment.name += '_{}-to-{}_from-ep{}'.format(
            config.env.n_disks, args.n_disks, args.params_episode)
        config.env.n_disks = args.n_disks
        config.training.lr = 1e-4
        config.training.lr_gamma = 0.95
        
        if args.n_disks == 3:
            config.training.max_episode_length = 1000000
            config.training.max_timesteps = 1000000
            
        if args.n_disks == 4:
            config.training.max_episode_length = 2000000
            config.training.max_timesteps = 2000000

    config.training.lr = args.lr  # new lr from before?
    config.training.device = args.device

    if args.name is not None:
        config.experiment.name = args.name + '_' + config.experiment.name

    if args.device is not None:
        config.training.device = args.device

    if args.seed is not None:
        config.experiment.seed = args.seed
        config.experiment.name = config.experiment.name + '_seed{}'.format(args.seed)

    if args.load_dir is not None:
        config.algorithm.load_dir = args.load_dir

    if args.action_file is not None:
        config.algorithm.load_action_dir = 'rlbase/action_dictionaries/'+args.action_file + '/'

    else:
        config.algorithm.load_action_dir = None
        
    config.experiment.name += '_lr{}_lrgamma_{}'.format(args.lr, config.training.lr_gamma)

    print(config.algorithm.load_action_dir)

    return config

def print_config(config):
    vars_dict = vars(config)
    print('config.experiment')
    pprint.pprint(vars(config.experiment))
    print('config.training')
    pprint.pprint(vars(config.training))  

def main():
    with open(args.model_dir+'config.p', 'rb') as f:
        checkpoint_config = pickle.load(f)
        
        print('{}\n{}\n{}'.format('#'*80,'CHECKPOINT CONFIG', '#'*80))

        transfer_config = produce_transfer_config(checkpoint_config)
        print_config(transfer_config)


        agent = SSC(transfer_config)

        checkpoint = torch.load(os.path.join(args.model_dir,'checkpoints','episode_{}'.format(args.params_episode)))
        agent.policy.load_state_dict(checkpoint['policy'])
#         print(torch.sum(agent.policy.actor.network[-1].weight.data[:5]))
#         print(torch.sum(agent.policy.actor.network[-1].weight.data[5:10]))
#         for l in agent.policy.actor.network:
#             l.reset_parameters()
#         for l in agent.policy.obs_transform_actor.conv_layers:
#             if isinstance(l, nn.Conv2d):
#                 l.reset_parameters()
#         for l in agent.policy.obs_transform_actor.fc_layers:
#             l.reset_parameters()

#         print(torch.sum(agent.policy.actor.network[-1].weight.data[:5]))
#         print(torch.sum(agent.policy.actor.network[-1].weight.data[5:10]))


        agent.train()
            
if __name__ == '__main__':
    main()

