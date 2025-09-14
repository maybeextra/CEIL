from .ecn import ECN
# from .gcrv import GCRV
from .gnn_reranking import gnn_reranking
from .kr import re_ranking
from .gcn import gcn
__factory = {
    'ECN': ECN,
    'KR': re_ranking,
    # 'GCRV': GCRV,
    'GCN': gcn,
    'GCR': gnn_reranking,
}


def creat(name):
    if name in __factory:
        return __factory[name]
    else:
        raise KeyError("Unknown dataset:", name)
