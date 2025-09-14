from __future__ import absolute_import

from .ClusterContrast.cross_memory import creat_Memory
from .clip.make_model import make_clip

__factory = {
    'clip': make_clip,
    'memory': creat_Memory,
}


def names():
    return sorted(__factory.keys())


def create(name, *args):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args)
