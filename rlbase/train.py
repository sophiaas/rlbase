import argparse
from agents import PPO, PPOC


parser = argparse.ArgumentParser(description='Lightbot')

parser.add_argument('--config', type=str, default='lightbot_zigzag',
                    help='Name of config') 
parser.add_argument('--algorithm', type=str, default='ppo',
                    help='Algorithm') 
parser.add_argument('--name', type=str, default=None,
                    help='Name to prepend to save dir') 

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

def main():
    model = agent(config)
    model.train()
            
if __name__ == '__main__':
    main()
