from configs.ppo.fourrooms import config as fc, post_process as fc_pp
from configs.ppo.lightbot import config as l, post_process as l_pp
from configs.ppo.lightbot_minigrid import config as lm, post_process as lm_pp
from configs.ppo.hanoi import config as h, post_process as h_pp
from configs.ppo.hanoi_lstm import config as hl, post_process as hl_pp
# from configs.ppo.minigrid_random_empty_5x5 import config as mgre

all_configs = {
    'fourrooms': fc,
    'lightbot': l,
    'lightbot_minigrid': lm,
    'hanoi': h,
    'hanoi_lstm': hl
#     'minigrid_random_empty_5x5': mgre,
}

all_post_processors = {
    'fourrooms': fc_pp,
    'lightbot': l_pp,
    'lightbot_minigrid': lm_pp,
    'hanoi': h_pp,
    'hanoi_lstm': hl_pp
}

