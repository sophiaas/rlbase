import argparse
from agents import PPO, PPOC
import torch
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Lightbot')

parser.add_argument('--config', type=str, default='lightbot_zigzag',
                    help='Name of config') 
parser.add_argument('--algorithm', type=str, default='ppo',
                    help='Algorithm') 
parser.add_argument('--name', type=str, default=None,
                    help='Name to prepend to save dir') 
parser.add_argument('--device', type=int, default=None,
                    help='Device to run on') 

args = parser.parse_args()

if args.algorithm == 'ppo':
    from configs.ppo import all_configs
    agent = PPO

elif args.algorithm == 'ppoc':
    from configs.ppoc import all_configs
    agent = PPOC
    
else:
    raise ValueError('Specified algorithm is not yet implemented')
    
config = all_configs[args.config]

if args.name is not None:
    config.experiment.name = args.name + '_' + config.experiment.name
    
if args.device is not None:
    config.training.device = args.device

def main():
    model = agent(config)
#     writer = SummaryWriter(config.experiment.base_dir 
#                            + config.experiment.name 
#                            + 'tensorboard/')
#     writer.add_graph(model.policy.actor, torch.autograd.Variable(torch.Tensor(1,1,512)).to(config.training.device))
#     writer.add_graph(model.policy.critic, torch.autograd.Variable(torch.Tensor(1,1,512)).to(config.training.device))
    
    model.train()
            
if __name__ == '__main__':
    main()
