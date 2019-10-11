import argparse
import os
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--for-real', action='store_true')
args = parser.parse_args()

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

class Runner():
    def __init__(self, command='python rlbase/train.py --device 0', gpus=[]):
        self.gpus = gpus
        self.command = command
        self.flags = {}

    def add_flag(self, flag_name, flag_values=''):
        self.flags[flag_name] = flag_values

    def generate_commands(self, execute=False):
        i = 0
        for flag_dict in product_dict(**self.flags):
            prefix = 'CUDA_VISIBLE_DEVICES={} '.format(self.gpus[i]) if len(self.gpus) > 0 else ''
            command = prefix+self.command
            for flag_name, flag_value in flag_dict.items():
                if type(flag_value) == bool:
                    if flag_value == True:
                        command += ' --{}'.format(flag_name)
                else:
                    command += ' --{} {}'.format(flag_name, flag_value)
            if len(self.gpus) == 0:
                command += ' --cpu'
            command += ' &'
            print(command)
            if execute:
                os.system(command)
            if len(self.gpus) > 0:
                i = (i + 1) % len(self.gpus)

def launch_hanoi():
    """
        9/23/19
        Launch hanoi experiments training from scratch
    """
    r = Runner(gpus=[0,1,2,3])
    r.add_flag('name', ['hanoi_maxsteps1000000_seeds_lrd95'])
    r.add_flag('algo', ['ppo', 'ppoc'])
    r.add_flag('config', ['hanoi'])
    r.add_flag('lr', [3.5e-4])
    r.add_flag('seed', [3,4,5])
    r.add_flag('n_disks', ['4'])
    r.generate_commands(execute=args.for_real)


def launch_minigrid():
    """
        9/23/19
        Launch minigrid experiments training from scratch
    """
    r = Runner(gpus=[4,5,6,7])
    r.add_flag('name', ['lbot_minigrid_maxsteps500000_seeds_lrd99'])
    r.add_flag('algo', ['ppo', 'ppoc'])
    r.add_flag('config', ['lightbot_minigrid'])
    r.add_flag('lr', [1e-4])
    r.add_flag('seed', [3,4,5])
    r.add_flag('puzzle', ['fractal_cross_0', 'fractal_cross_0-1'])
    r.generate_commands(execute=args.for_real)


if __name__ == '__main__':
    launch_minigrid()
    launch_hanoi()




