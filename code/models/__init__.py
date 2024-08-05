from __future__ import absolute_import
from .clip.make_model import make_clip
from .ClusterContrast.cross_memory import creat_Memory
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
