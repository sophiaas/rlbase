from agents import PPOC
from configs.fourrooms_ppoc import config
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    ppoc = PPOC(config)
    ppoc.train()
            
if __name__ == '__main__':
    main()

