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

      `reward_fn = '100,-1'; max_episode_length = 500`
      - 2 Disks `--n_disks 2`
      - 3 Disks `--n_disks 3`
      - 4 Disks `--n_disks 4`
      
  - Lightbot Minigrid
    `python rlbase/train.py --algo XXX --config lightbot_minigrid`

      `reward_fn = '10,10,-1,-1'; max_episode_length = 500`
      - Fractal Cross 0  `--puzzle fractal_cross_0`
      - Fractal Cross 1  `--puzzle fractal_cross_1`
      - Fractal Cross 2  `--puzzle fractal_cross_2`
     
---
## Comparison of Abstractions 
*Cross the following*
**Algorithms**
  - PPO    `python rlbase/train.py --algo ppo`
  - PPOC   `python rlbase/train.py --algo ppoc`
  
**Environments**
  - Lightbot Cross State Space  `--config lightbot`
  
    `reward_fn = 10,10,-1,-1'; max_episode_length = 100`
    
  - Four Rooms State Space      `--config fourrooms`
  
   `reward_fn = '100,-1'; max_episode_length = 500`
  
  
