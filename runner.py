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

algos = ['ppo', 'ppoc']
configs = {
    'hanoi': ('n_disks', ['2', '3', '4']),
    'lightbot_minigrid': ('puzzle', ['fractal_cross_0', 'fractal_cross_1', 'fractal_cross_2']),
    'lightbot': (),
    'fourrooms': ()
}

gpu=True
num_gpus = 2
i = 0

def heading(algo, config, i, gpu):
    prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
    command = prefix + 'python rlbase/train.py --device 0 --algo {} --config {}'.format(
        algo, config)
    return command

def execute(command, i, num_gpus):
    command += ' &'
    print(command)
    # os.system(command)

    i += 1
    if i >= num_gpus:
        i = 0
    return i

for a, c in itertools.product(algos, configs.keys()):
    if len(configs[c]) > 0:
        flag, variants = configs[c]
        for variant in variants:
            command = heading(a, c, i, gpu)
            command += ' --{} {}'.format(flag, variant)
            command += ' > experiments/algo_{}_config_{}_{}.txt'.format(a, c, variant)
            i = execute(command, i, num_gpus)
    else:
        command = heading(a, c, i, gpu)
        command += ' > experiments/algo_{}_config_{}.txt'.format(a, c)
        i = execute(command, i, num_gpus)


