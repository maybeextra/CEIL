from .sampler_by_cam import Sampler_by_cam
from .sampler_by_identity import Sampler_by_identity

__factory = {
    'sampler_by_cam':Sampler_by_cam,
    'sampler_by_identity': Sampler_by_identity
}


def creat(name, batch_size, num_instances, num_iter = None):
    if name in __factory:
        return __factory[name](batch_size, num_instances, num_iter)
    else:
        raise KeyError("Unknown dataset:", name)
