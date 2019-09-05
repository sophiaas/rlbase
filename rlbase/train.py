import os
import argparse
from agents import PPO, PPOC
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='lightbot_cross',
                    help='Name of config') 
parser.add_argument('--algo', type=str, default='ppo',
                    help='Algorithm') 
parser.add_argument('--name', type=str, default=None,
                    help='Name to prepend to save dir') 
parser.add_argument('--device', type=int, default=0,
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
    
config = all_configs[args.config]

if args.name is not None:
    config.experiment.name = args.name + '_' + config.experiment.name
    
config.experiment.name = args.algo + '_' + config.experiment.name
    
if args.device is not None:
    config.training.device = args.device
else:
    config.training.device = 'cpu'

def main():
    model = agent(config)
    model.train()
            
if __name__ == '__main__':
    main()
