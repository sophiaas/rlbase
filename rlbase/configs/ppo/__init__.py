from configs.ppo.fourrooms import config as fc
from configs.ppo.lightbot import config as l
from configs.ppo.lightbot_minigrid import config as lm
from configs.ppo.hanoi import config as h
from configs.ppo.minigrid_random_empty_5x5 import config as mgre

all_configs = {
    'fourrooms': fc,
    'lightbot': l,
    'lightbot_minigrid': lm,
    'hanoi': h,
    'minigrid_random_empty_5x5': mgre,
}


