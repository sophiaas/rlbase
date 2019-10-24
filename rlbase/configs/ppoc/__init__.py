from configs.ppoc.fourrooms import config as fc, post_process as fc_pp
from configs.ppoc.lightbot import config as l, post_process as l_pp
from configs.ppoc.lightbot_minigrid import config as lm, post_process as lm_pp
from configs.ppoc.hanoi import config as h, post_process as h_pp
from configs.ppoc.lightbot_block_entropy import config as lbe, post_process as lbe_pp

all_configs = {
    'fourrooms': fc,
    'lightbot': l,
    'lightbot_minigrid': lm,
    'hanoi': h,
    'lightbot_block_entropy': lbe
}

all_post_processors = {
    'fourrooms': fc_pp,
    'lightbot': l_pp,
    'lightbot_minigrid': lm_pp,
    'hanoi': h_pp,
    'lightbot_block_entropy': lbe_pp
}
