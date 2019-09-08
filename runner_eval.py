import os
import itertools

"""
# Hanoi

'python rlbase/train.py --algo ppo --config hanoi --n_disks 2'
'python rlbase/train.py --algo ppo --config hanoi --n_disks 3'
'python rlbase/train.py --algo ppo --config hanoi --n_disks 4'

'python rlbase/train.py --algo ppoc --config hanoi --n_disks 2'
'python rlbase/train.py --algo ppoc --config hanoi --n_disks 3'
'python rlbase/train.py --algo ppoc --config hanoi --n_disks 4'

# lightbot minigrid

'python rlbase/train.py --algo ppo --config lightbot_minigrid --puzzle fractal_cross_0'
'python rlbase/train.py --algo ppo --config lightbot_minigrid --puzzle fractal_cross_1'
'python rlbase/train.py --algo ppo --config lightbot_minigrid --puzzle fractal_cross_2'

'python rlbase/train.py --algo ppoc --config lightbot_minigrid --puzzle fractal_cross_0'
'python rlbase/train.py --algo ppoc --config lightbot_minigrid --puzzle fractal_cross_1'
'python rlbase/train.py --algo ppoc --config lightbot_minigrid --puzzle fractal_cross_2'

# lightbot

'python rlbase/train.py --algo ppo --config lightbot'
'python rlbase/train.py --algo ppoc --config lightbot'

# fourrooms
'python rlbase/train.py --algo ppo --config fourrooms'
'python rlbase/train.py --algo ppoc --config fourrooms'
"""

prefix ='python rlbase/evaluate.py --model_dir'

model_dirs = [
    # # lightbot state space
    # '/home/mbchang/shared/ssc/rlbase/experiments/adam_ppo_lightbot_lr0.0001',
    # '/home/mbchang/shared/ssc/rlbase/experiments/adam_ppoc_lightbot_lr0.0005',

    # # fourrooms
    # '/home/mbchang/shared/ssc/rlbase/experiments/betterlr_ss/betterlr_ss_ppo_fourooms_lr0.001',
    # '/home/mbchang/shared/ssc/rlbase/experiments/betterlr_ss/betterlr_ss_ppoc_fourrooms_lr0.001',

    # hanoi
    '/home/mbchang/shared/ssc/rlbase/experiments/betterlr_h_eplen500_me30000_hdim256/betterlr_h_eplen500_me30000_hdim256_ppo_hanoi_2disks_lr0.0005',

    # # lightbot minigrid
    # '/home/mbchang/shared/ssc/rlbase/experiments/sparse500000/sparse500000_ppo_lightbot_minigrid_fractal_cross_0_lr0.001',
    # '/home/mbchang/shared/ssc/rlbase/experiments/sparse500000/sparse500000_ppoc_lightbot_minigrid_fractal_cross_0_lr0.001',
]

num_gpus = 2
i = 0

for model_dir in model_dirs:
    logfile = os.path.join(model_dir, 'eval.txt')
    command = 'CUDA_VISIBLE_DEVICES={} {} {} > {} &'.format(i, prefix, model_dir, logfile)
    print(command)
    # os.system(command)

    i += 1
    if i >= num_gpus:
        i = 0
