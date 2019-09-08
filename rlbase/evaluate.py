import os
import argparse
from core.evaluator import Evaluator
from core.config import EvalConfig
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default=None,
                    help='Directory containing model to load') 
parser.add_argument('--episode', type=int, default=20000,
                    help='Episode to load from') 
parser.add_argument('--n_eval_steps', type=int, default=2000,
                    help='Number of steps to evaluate') 
parser.add_argument('--device', type=int, default=0,
                    help='Device to run on') 

args = parser.parse_args()
    
config = EvalConfig({'n_eval_steps': args.n_eval_steps,
                     'model_dir': args.model_dir,
                     'device': args.device,
                     'episode': args.episode
                    })

def main():
    evaluator = Evaluator(config)
    evaluator.model.evaluate()
            
if __name__ == '__main__':
    main()
