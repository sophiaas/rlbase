import argparse
import os
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--clean', action='store_true')
parser.add_argument('--zip', action='store_true')
parser.add_argument('--zipall', action='store_true')
parser.add_argument('--for-real', action='store_true')
parser.add_argument('--root', type=str, default='')
args = parser.parse_args()

model_dirs = {
    # # lightbot state space
    # '/home/mbchang/shared/ssc/rlbase/experiments/adam_ppo_lightbot_lr0.0001': {'episode': 9980},
    # '/home/mbchang/shared/ssc/rlbase/experiments/adam_ppoc_lightbot_lr0.0005': {'episode': 9980},

    # # fourrooms
    # '/home/mbchang/shared/ssc/rlbase/experiments/betterlr_ss/betterlr_ss_ppo_fourooms_lr0.001': {'episode': 9980},
    # '/home/mbchang/shared/ssc/rlbase/experiments/betterlr_ss/betterlr_ss_ppoc_fourrooms_lr0.001': {'episode': 9980},

    # hanoi
    '/home/mbchang/shared/ssc/rlbase/experiments/betterlr_h_eplen500_me30000_hdim256/betterlr_h_eplen500_me30000_hdim256_ppo_hanoi_2disks_lr0.0005': {'episode': 29980},
    '/home/mbchang/shared/ssc/rlbase/experiments/betterlr_h_eplen500_me30000_hdim256/betterlr_h_eplen500_me30000_hdim256_ppoc_hanoi_2disks_lr0.0005': {'episode': 29980},

    # # lightbot minigrid
    # '/home/mbchang/shared/ssc/rlbase/experiments/sparse500000/sparse500000_ppo_lightbot_minigrid_fractal_cross_0_lr0.001': {'episode': 9980},
    # '/home/mbchang/shared/ssc/rlbase/experiments/sparse500000/sparse500000_ppoc_lightbot_minigrid_fractal_cross_0_lr0.001': {'episode': 9980},
}

def evaluate():
    prefix ='python rlbase/evaluate.py --model_dir'

    num_gpus = 2
    i = 0

    for model_dir in model_dirs.keys():
        logfile = os.path.join(model_dir, 'eval.txt')

        config = model_dirs[model_dir]
        config_command_str = ''.join(['--{} {}'.format(k, v) for k, v in config.items()])

        command = 'CUDA_VISIBLE_DEVICES={} {} {} {} > {} &'.format(i, prefix, model_dir, config_command_str, logfile)
        print(command)
        if args.for_real:
            os.system(command)

        i += 1
        if i >= num_gpus:
            i = 0

def zip_folders():
    for model_dir in model_dirs.keys():
        command = 'zip -r {}.zip {} &'.format(
            model_dir, model_dir)
        print(command)
        if args.for_real:
            os.system(command)

def zip_all(root):
    for folder in os.listdir(root):
        if '.zip' not in folder and 'zipped' not in folder:
            command = 'zip -r {}.zip {} &'.format(
                os.path.join(root,'zipped',folder), os.path.join(root,folder))
            print(command)
            if args.for_real:
                os.system(command)

def clean_hanoi():
    root = '/Users/michaelchang/Documents/Researchlink/Berkeley/sophia/rlbase/rlbase/experiments/server/hanoi_maxsteps1000000_seeds_lrd95/zipped'
    remove_for_2_3_disks = lambda file: ('2disks' in file or '3disks' in file) and not 'lr0.0001' in file
    remove_for_4_disks = lambda file: '4disks' in file and not 'lr0.0004_' in file

    for file in [x for x in os.listdir(root) if 'hanoi' in x]:
        if remove_for_2_3_disks(file):
            print('Removing: {}'.format(file))
            if args.for_real:
                os.remove(os.path.join(root, file))
        elif remove_for_4_disks(file):
            print('Removing: {}'.format(file))
            if args.for_real:
                os.remove(os.path.join(root, file))

def clean_lbot():
    root = '/Users/michaelchang/Documents/Researchlink/Berkeley/sophia/rlbase/rlbase/experiments/server/lbot_minigrid_maxsteps500000_seeds_lrd99/zipped'
    remove_for_0 = lambda file: 'fractal_cross_0_lr' in file and not 'fractal_cross_0_lr0.0001_steps' in file
    remove_for_0_transfer = lambda file: 'fractal_cross_0-to-fractal_cross_0-1_from-ep4000_lr0.0001' in file and not 'from-ep4000_1000000_lr' in file

    for file in [x for x in os.listdir(root) if 'fractal' in x]:
        if remove_for_0(file):
            print('Removing: {}'.format(file))
            if args.for_real:
                os.remove(os.path.join(root, file))
        if remove_for_0_transfer(file):
            print('Removing: {}'.format(file))
            if args.for_real:
                os.remove(os.path.join(root, file))


if __name__ == '__main__':
    if args.zip:
        zip_folders()
    elif args.clean:
        # clean_hanoi()
        clean_lbot()
    elif args.zipall:
        roots = [
            # '/Users/michaelchang/Documents/Researchlink/Berkeley/sophia/rlbase/rlbase/experiments/server/lbot_minigrid_maxsteps5000000_seeds',
            # '/Users/michaelchang/Documents/Researchlink/Berkeley/sophia/rlbase/rlbase/experiments/server/hanoi_maxteps5000000_seeds',

            # '/Users/michaelchang/Documents/Researchlink/Berkeley/sophia/rlbase/rlbase/experiments/server/lbot_minigrid_maxsteps5000000_seeds',
            # '/Users/michaelchang/Documents/Researchlink/Berkeley/sophia/rlbase/rlbase/experiments/server/hanoi_maxsteps500000_seeds_lrd95',

            # '/Users/michaelchang/Documents/Researchlink/Berkeley/sophia/rlbase/rlbase/experiments/server/hanoi_maxsteps1000000_seeds_lrd99',
            # '/Users/michaelchang/Documents/Researchlink/Berkeley/sophia/rlbase/rlbase/experiments/server/lbot_minigrid_maxsteps500000_seeds_lrd99',

            '/Users/michaelchang/Documents/Researchlink/Berkeley/sophia/rlbase/rlbase/experiments/server/hanoi_maxsteps1000000_seeds_lrd95',
            '/Users/michaelchang/Documents/Researchlink/Berkeley/sophia/rlbase/rlbase/experiments/server/lbot_minigrid_maxsteps500000_seeds_lrd99',
        ]
        for root in roots:
            zip_all(root)
    else:
        evaluate()


