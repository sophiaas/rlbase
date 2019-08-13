from agents import PPO
#from configs.lightbot_zigzag_ppo import config
from configs.lightbot_cross_ppo import config
#from configs.fourrooms_ppo import config

def main():
    ppo = PPO(config)
    ppo.train()
            
if __name__ == '__main__':
    main()

