import warnings

from .eval_metrics import eval_llcm, eval_sysu, eval_regdb

__factory = {
    'LLCM': eval_llcm,
    'SYSU': eval_sysu,
    'RegDB': eval_regdb
}

def names():
    return sorted(__factory.keys())


def create(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name]

def use(name, distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20)




