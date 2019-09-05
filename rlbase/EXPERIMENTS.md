# EXPERIMENTS

# Baselines
## Transfer Learning Experiments
*Cross all of the following:*

  **Algorithms**
  - PPO   `python rlbase/train.py --algo ppo`
    
  - PPOC  `python rlbase/train.py --algo ppoc`
  
  **Training setups**
  - No transfer (train from scratch on each puzzle)
  - Transfer weights accross puzzles
  
  **Environments**
  - Towers of Hanoi
    `python rlbase/train.py --algo XXX --config hanoi`

      - 2 Disks `--n_disks 2`
      - 3 Disks `--n_disks 3`
      - 4 Disks `--n_disks 4`
      
  - Lightbot Minigrid
    `python rlbase/train.py --algo XXX --config lightbot_minigrid`

      - Fractal Cross 0  `--puzzle fractal_cross_0`
      - Fractal Cross 1  `--puzzle fractal_cross_1`
      - Fractal Cross 2  `--puzzle fractal_cross_2`
     
---
## Comparison of Abstractions      
**Algorithms**
  - PPO    `python rlbase/train.py --algo ppo`
  - PPOC   `python rlbase/train.py --algo ppoc`
  
**Environments**
  - Lightbot Cross State Space  `--config lightbot_cross`
  - Four Rooms State Space      `--config fourrooms`
  
