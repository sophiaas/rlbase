import argparse
import copy
import os
import pickle
from agents import PPO, PPOC
import torch
import pprint

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default=None,
                    help='Directory containing model to load') 
parser.add_argument('--episode', type=int, default=9980,
                    help='Episode to load from') 
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

args = parser.parse_args()

def produce_transfer_config(config):
    if args.puzzle:
        config.experiment.name += '_{}->{}'.format(
            checkpoint_config.env.puzzle_name, args.puzzle)
        config.env.puzzle_name = args.puzzle
    if args.n_disks:
        config.experiment.name += '_{}->{}'.format(
            config.env.n_disks, args.n_disks)
        config.env.n_disks = args.n_disks

    config.training.lr = args.lr  # new lr from before?
    config.training.device = args.device

    config.experiment.name += '_lr{}'.format(args.lr)
    return config

def print_config(config):
    vars_dict = vars(config)
    print('config.experiment')
    pprint.pprint(vars(config.experiment))
    print('config.training')
    pprint.pprint(vars(config.training))

def main():
    with open(args.model_dir+'/config.p', 'rb') as f:
        checkpoint_config = pickle.load(f)

        print('{}\n{}\n{}'.format('#'*80,'CHECKPOINT CONFIG', '#'*80))
        print_config(checkpoint_config)

        transfer_config = produce_transfer_config(checkpoint_config)

        print('{}\n{}\n{}'.format('*'*80,'FOR TRANSFER', '*'*80))
        print_config(transfer_config)
        print('{}\n{}\n{}'.format('#'*80,'CHECKPOINT CONFIG', '#'*80))

        if args.device >= 0:
            transfer_config.training.device = args.device
        else:
            transfer_config.training.device = 'cpu'

        if checkpoint_config.algorithm.name == 'PPO':
            agent = PPO(checkpoint_config)
        elif checkpoint_config.algorithm.name == 'PPOC':
            agent = PPOC(checkpoint_config)
        else:
            raise ValueError('Unknown model type')

        checkpoint = torch.load(os.path.join(args.model_dir,'checkpoints','episode_{}'.format(args.episode)))
        agent.policy.load_state_dict(checkpoint['policy'])

        agent.train()
            
if __name__ == '__main__':
    main()

