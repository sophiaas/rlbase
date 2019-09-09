import argparse
import os
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--zip', action='store_true')
parser.add_argument('--for-real', action='store_true')
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

if __name__ == '__main__':
    if args.zip:
        zip_folders()
    else:
        evaluate()
