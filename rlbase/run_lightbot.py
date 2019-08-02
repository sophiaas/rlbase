from lightbot_config import config
import core

def run(config):
    agent = config.algorithm.init(config)
    agent.train()
    
run(config)