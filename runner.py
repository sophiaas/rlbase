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

############################################################################################

"""
9/5/19
I used SGD. It didn't work. Next I will try Adam.
"""

# algos = ['ppo', 'ppoc']
# configs = {
#     'hanoi': ('n_disks', ['2', '3', '4']),
#     'lightbot_minigrid': ('puzzle', ['fractal_cross_0', 'fractal_cross_1', 'fractal_cross_2']),
#     'lightbot': (),
#     'fourrooms': ()
# }

# gpu=True
# num_gpus = 2
# i = 0

# def heading(algo, config, i, gpu):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = prefix + 'python rlbase/train.py --device 0 --algo {} --config {}'.format(
#         algo, config)
#     return command

# def execute(command, i, num_gpus):
#     command += ' &'
#     print(command)
#     # os.system(command)

#     i += 1
#     if i >= num_gpus:
#         i = 0
#     return i

# for a, c in itertools.product(algos, configs.keys()):
#     if len(configs[c]) > 0:
#         flag, variants = configs[c]
#         for variant in variants:
#             command = heading(a, c, i, gpu)
#             command += ' --{} {}'.format(flag, variant)
#             command += ' > experiments/algo_{}_config_{}_{}.txt'.format(a, c, variant)
#             i = execute(command, i, num_gpus)
#     else:
#         command = heading(a, c, i, gpu)
#         command += ' > experiments/algo_{}_config_{}.txt'.format(a, c)
#         i = execute(command, i, num_gpus)

############################################################################################

"""
9/7/19
It learns with adam. Now will begin narrowing down the learning rates.
"""

# algos = ['ppo', 'ppoc']
# configs = {
#     # 'hanoi': ('n_disks', ['2', '3', '4']),
#     'lightbot_minigrid': ('puzzle', ['fractal_cross_0', 'fractal_cross_1', 'fractal_cross_2']),
#     # 'lightbot': (),
#     # 'fourrooms': ()
# }
# lrs = [5e-4, 1e-4, 5e-3]  # could try 1e-3

# gpu=True
# num_gpus = 6
# i = 4

# def heading(algo, config, r, i, gpu):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = prefix + 'python rlbase/train.py --device 0 --name sparse50000 --algo {} --config {} --lr {}'.format(
#         algo, config, r)
#     return command

# def execute(command, i, num_gpus):
#     command += ' &'
#     print(command)
#     # os.system(command)

#     i += 1
#     if i >= num_gpus:
#         i = 4
#     return i

# for a, c, r in itertools.product(algos, configs.keys(), lrs):
#     if len(configs[c]) > 0:
#         flag, variants = configs[c]
#         for variant in variants:
#             command = heading(a, c, r, i, gpu)
#             command += ' --{} {}'.format(flag, variant)
#             command += ' > experiments/sparse50000/algo_{}_config_{}_{}_lr{}.txt'.format(a, c,variant, r)
#             i = execute(command, i, num_gpus)
#     else:
#         command = heading(a, c, r, i, gpu)
#         command += ' > experiments/sparse50000/algo_{}_config_{}_lr{}.txt'.format(a, c, r)
#         i = execute(command, i, num_gpus)

############################################################################################

# state space

# algos = ['ppo', 'ppoc']
# configs = {
#     'lightbot': (),
#     'fourrooms': ()
# }
# lrs = [1e-3, 3e-4, 7e-4]

# group = 'betterlr'

# gpu=True
# num_gpus = 2
# i = 0

# def heading(algo, config, r, i, gpu):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = prefix + 'python rlbase/train.py --device 0 --name {} --algo {} --config {} --lr {}'.format(
#         group, algo, config, r)
#     return command

# def execute(command, i, num_gpus):
#     command += ' &'
#     print(command)
#     # os.system(command)

#     i += 1
#     if i >= num_gpus:
#         i = 0
#     return i

# for a, c, r in itertools.product(algos, configs.keys(), lrs):
#     if len(configs[c]) > 0:
#         flag, variants = configs[c]
#         for variant in variants:
#             command = heading(a, c, r, i, gpu)
#             command += ' --{} {}'.format(flag, variant)
#             command += ' > experiments/{}/algo_{}_config_{}_{}_lr{}.txt'.format(group, a, c,variant, r)
#             i = execute(command, i, num_gpus)
#     else:
#         command = heading(a, c, r, i, gpu)
#         command += ' > experiments/{}/algo_{}_config_{}_lr{}.txt'.format(group, a, c, r)
#         i = execute(command, i, num_gpus)

# hanoi

algos = ['ppo', 'ppoc']
configs = {
    'hanoi': ('n_disks', ['2', '3', '4']),
    # 'lightbot_minigrid': ('puzzle', ['fractal_cross_0', 'fractal_cross_1', 'fractal_cross_2']),
    # 'lightbot': (),
    # 'fourrooms': ()
}

lrs = {
    '2': [1e-2, 7e-3],
    '3': [3e-3, 7e-3],
    '4': [1e-5, 5e-5],
}

group = 'betterlr_h'

gpu=True
num_gpus = 2
i = 0

def heading(algo, config, r, i, gpu):
    prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
    command = prefix + 'python rlbase/train.py --device 0 --name {} --algo {} --config {} --lr {}'.format(
        group, algo, config, r)
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
    flag, variants = configs[c]
    for variant in variants:
        for r in lrs[variant]:
            command = heading(a, c, r, i, gpu)
            command += ' --{} {}'.format(flag, variant)
            command += ' > experiments/{}/algo_{}_config_{}_{}_lr{}.txt'.format(group, a, c,variant, r)
            i = execute(command, i, num_gpus)


# minigrid


