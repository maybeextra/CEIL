from .cam_associate_loss import get_cam_associate_loss
from .proxy_associate_loss import get_proxy_associate_loss


__factory = {
    'cam_associate': get_cam_associate_loss,
    'proxy_associate': get_proxy_associate_loss,
}

def names():
    return sorted(__factory.keys())


def create(name):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name]
