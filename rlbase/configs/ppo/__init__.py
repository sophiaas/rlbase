from configs.ppo.fourrooms import config as fc, post_process as fc_pp
from configs.ppo.lightbot import config as l, post_process as l_pp
from configs.ppo.hanoi import config as h, post_process as h_pp
from configs.ppo.hanoi_lstm import config as hl, post_process as hl_pp

all_configs = {
    "fourrooms": fc,
    "lightbot": l,
    "hanoi": h,
    "hanoi_lstm": hl,
}

all_post_processors = {
    "fourrooms": fc_pp,
    "lightbot": l_pp,
    "hanoi": h_pp,
    "hanoi_lstm": hl_pp,
}
