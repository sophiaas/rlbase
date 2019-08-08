from agents import PPO
from configs.lightbot import config

def main():
    ppo = PPO(config)
    ppo.train()
            
if __name__ == '__main__':
    main()

