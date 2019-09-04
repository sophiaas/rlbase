from configs.ppo.fourrooms import config as fc
from configs.ppo.lightbot_cross import config as lcc
from configs.ppo.lightbot_zigzag import config as lzc
from configs.ppo.lightbot_debug1 import config as ld1
from configs.ppo.lightbot_minigrid_cross import config as lmc
from configs.ppo.hanoi_3_disks import config as h3d

all_configs = {
    'fourrooms': fc,
    'lightbot_cross': lcc,
    'lightbot_zigzag': lzc,
    'lightbot_debug1': ld1,
    'lightbot_minigrid_cross': lmc,
    'hanoi_3_disks': h3d
}
