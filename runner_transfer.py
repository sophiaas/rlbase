import argparse
import os
import itertools

from runner_eval import model_dirs

parser = argparse.ArgumentParser()
parser.add_argument('--for-real', action='store_true')
args = parser.parse_args()

prefix = '/home/mbchang/shared/ssc/rlbase/experiments/'

model_dirs = {
    # # # hanoi
    # '/home/mbchang/shared/ssc/rlbase/experiments/betterlr_h_eplen500_me30000_hdim256/betterlr_h_eplen500_me30000_hdim256_ppo_hanoi_2disks_lr0.0005': 
    #     [{'n_disks': 3, 'lr': 5e-4, 'episode': 29980}],
    # '/home/mbchang/shared/ssc/rlbase/experiments/betterlr_h_eplen500_me30000_hdim256/betterlr_h_eplen500_me30000_hdim256_ppoc_hanoi_2disks_lr0.0005':
    #     [{'n_disks': 3, 'lr': 5e-4, 'episode': 29980}],
    # # # 10000

    # prefix+'betterlr_h_eplen500_me30000_hdim256/betterlr_h_eplen500_me30000_hdim256_ppo_hanoi_2disks_lr0.0005_2-to-3_from-ep10000_lr0.0005': 
    #     [{'n_disks': 4, 'lr': 5e-4, 'episode': 9980}],

    # prefix+'betterlr_h_eplen500_me30000_hdim256/betterlr_h_eplen500_me30000_hdim256_ppoc_hanoi_2disks_lr0.0005_2-to-3_from-ep10000_lr0.0005': 
    #     [{'n_disks': 4, 'lr': 5e-4, 'episode': 9980}],

    # ****************************

    # prefix+'betterlr_h_eplen500_me30000_hdim256_seeds/betterlr_h_eplen500_me30000_hdim256_seeds_ppo_hanoi_2disks_lr0.0005_seed0':
    #         [{'n_disks': 3, 'lr': 5e-4, 'episode': 29980}],
    # prefix+'betterlr_h_eplen500_me30000_hdim256_seeds/betterlr_h_eplen500_me30000_hdim256_seeds_ppo_hanoi_2disks_lr0.0005_seed1':
    #         [{'n_disks': 3, 'lr': 5e-4, 'episode': 29980}],
    # prefix+'betterlr_h_eplen500_me30000_hdim256_seeds/betterlr_h_eplen500_me30000_hdim256_seeds_ppo_hanoi_2disks_lr0.0005_seed2':
    #         [{'n_disks': 3, 'lr': 5e-4, 'episode': 29980}],

    # prefix+'hanoi_maxteps5000000_seeds/hanoi_maxteps5000000_seeds_ppo_hanoi_2disks_lr0.0005_steps_seed3':
    #     [{'n_disks': 3, 'lr': 5e-4, 'episode': 3000}],
    # prefix+'hanoi_maxteps5000000_seeds/hanoi_maxteps5000000_seeds_ppo_hanoi_2disks_lr0.0005_steps_seed4':
    #     [{'n_disks': 3, 'lr': 5e-4, 'episode': 3000}],
    # prefix+'hanoi_maxteps5000000_seeds/hanoi_maxteps5000000_seeds_ppo_hanoi_2disks_lr0.0005_steps_seed5':
    #     [{'n_disks': 3, 'lr': 5e-4, 'episode': 3000}],

    # prefix+'hanoi_maxteps5000000_seeds/hanoi_maxteps5000000_seeds_ppo_hanoi_2disks_lr0.0005_steps_seed3_2-to-3_from-ep3000_lr0.0005':
    #     [{'n_disks': 4, 'lr': 5e-4, 'episode': 3000}],
    # prefix+'hanoi_maxteps5000000_seeds/hanoi_maxteps5000000_seeds_ppo_hanoi_2disks_lr0.0005_steps_seed4_2-to-3_from-ep3000_lr0.0005':
    #     [{'n_disks': 4, 'lr': 5e-4, 'episode': 3000}],
    # prefix+'hanoi_maxteps5000000_seeds/hanoi_maxteps5000000_seeds_ppo_hanoi_2disks_lr0.0005_steps_seed5_2-to-3_from-ep3000_lr0.0005':
    #     [{'n_disks': 4, 'lr': 5e-4, 'episode': 3000}],

    ###############################


    # # # lightbot minigrid
    # '/home/mbchang/shared/ssc/rlbase/experiments/sparse500000/sparse500000_ppo_lightbot_minigrid_fractal_cross_0_lr0.001': 
    #     [{'puzzle': 'fractal_cross_1', 'lr': 5e-4, 'episode': 9980}],
    # '/home/mbchang/shared/ssc/rlbase/experiments/sparse500000/sparse500000_ppoc_lightbot_minigrid_fractal_cross_0_lr0.001': 
    #     [{'puzzle': 'fractal_cross_1', 'lr': 5e-4, 'episode': 9980}],
    # # 3000  

    # prefix+'sparse500000/sparse500000_ppo_lightbot_minigrid_fractal_cross_0_lr0.001_fractal_cross_0-to-fractal_cross_1_from-ep3320_lr0.0005':
    #     [{'puzzle': 'fractal_cross_2', 'lr': 5e-4, 'episode': 9980}],

    # prefix+'sparse500000/sparse500000_ppoc_lightbot_minigrid_fractal_cross_0_lr0.001_fractal_cross_0-to-fractal_cross_1_from-ep3320_lr0.0005':
    #     [{'puzzle': 'fractal_cross_2', 'lr': 5e-4, 'episode': 9980}],

    # ****************************

    # prefix+'betterlr_lm_500000_seeds/betterlr_lm_500000_seeds_ppo_lightbot_minigrid_fractal_cross_0_lr0.0005_seed0':
    #         [{'puzzle': 'fractal_cross_1', 'lr': 5e-4, 'episode': 9980}],
    # prefix+'betterlr_lm_500000_seeds/betterlr_lm_500000_seeds_ppo_lightbot_minigrid_fractal_cross_0_lr0.0005_seed1':
    #         [{'puzzle': 'fractal_cross_1', 'lr': 5e-4, 'episode': 9980}],
    # prefix+'betterlr_lm_500000_seeds/betterlr_lm_500000_seeds_ppo_lightbot_minigrid_fractal_cross_0_lr0.0005_seed2':
    #         [{'puzzle': 'fractal_cross_1', 'lr': 5e-4, 'episode': 9980}],

   #  prefix+'lbot_minigrid_maxsteps5000000_seeds/lbot_minigrid_maxsteps5000000_seeds_ppo_lightbot_minigrid_fractal_cross_0_lr0.0005_steps_seed3':
   #      [{'puzzle': 'fractal_cross_1', 'lr': 5e-4, 'episode': 3000}],
   # prefix+'lbot_minigrid_maxsteps5000000_seeds/lbot_minigrid_maxsteps5000000_seeds_ppo_lightbot_minigrid_fractal_cross_0_lr0.0005_steps_seed4':
   #      [{'puzzle': 'fractal_cross_1', 'lr': 5e-4, 'episode': 3000}],
   #  prefix+'lbot_minigrid_maxsteps5000000_seeds/lbot_minigrid_maxsteps5000000_seeds_ppo_lightbot_minigrid_fractal_cross_0_lr0.0005_steps_seed5':
   #      [{'puzzle': 'fractal_cross_1', 'lr': 5e-4, 'episode': 3000}],


    prefix+'lbot_minigrid_maxsteps5000000_seeds/lbot_minigrid_maxsteps5000000_seeds_ppo_lightbot_minigrid_fractal_cross_0_lr0.0005_steps_seed3_fractal_cross_0-to-fractal_cross_1_from-ep3000_lr0.0005':
        [{'puzzle': 'fractal_cross_2', 'lr': 5e-4, 'episode': 3000}],
    prefix+'lbot_minigrid_maxsteps5000000_seeds/lbot_minigrid_maxsteps5000000_seeds_ppo_lightbot_minigrid_fractal_cross_0_lr0.0005_steps_seed4_fractal_cross_0-to-fractal_cross_1_from-ep3000_lr0.0005':
        [{'puzzle': 'fractal_cross_2', 'lr': 5e-4, 'episode': 3000}],
    prefix+'lbot_minigrid_maxsteps5000000_seeds/lbot_minigrid_maxsteps5000000_seeds_ppo_lightbot_minigrid_fractal_cross_0_lr0.0005_steps_seed5_fractal_cross_0-to-fractal_cross_1_from-ep3000_lr0.0005':
        [{'puzzle': 'fractal_cross_2', 'lr': 5e-4, 'episode': 3000}],


    ###############################
}

def myround(x, base=1):
    return base * round(x/base)

prefix = 'python rlbase/transfer.py'

num_gpus = 8
i = 0

for model_dir in model_dirs.keys():
    for config in model_dirs[model_dir]:

        config['episode'] = myround(config['episode']/1.0, base=20)  # try to do this sort of transfer

        config_logstr = ''.join(['{}_{}'.format(k, v) for k, v in config.items()])
        logfile = os.path.join(model_dir, 'transfer_{}.txt'.format(config_logstr))

        config_command_str = ''.join([' --{} {}'.format(k, v) for k, v in config.items()])
        command = 'CUDA_VISIBLE_DEVICES={} {} --model_dir {} {} > {} &'.format(
            i, prefix, model_dir, config_command_str, logfile)

        print(command)
        if args.for_real:
            os.system(command)

        i += 1
        if i >= num_gpus:
            i = 0
