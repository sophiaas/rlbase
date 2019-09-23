from configs.ssc.lightbot import config as l, post_process as l_pp
from configs.ssc.hanoi import config as h, post_process as h_pp
from configs.ssc.lightbot_minigrid import config as lm, post_process as lm_pp
from configs.ssc.fourrooms import config as f, post_process as fc_pp


all_configs = {
    'lightbot': l,
    'hanoi': h,
    'lightbot_minigrid': lm,
    'fourrooms': f
}


all_post_processors = {
    'fourrooms': fc_pp,
    'lightbot': l_pp,
    'lightbot_minigrid': lm_pp,
    'hanoi': h_pp
}

