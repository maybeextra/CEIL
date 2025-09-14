import warnings

from .dataset import train_Dataset, test_Dataset, Pre_extract

__factory = {
    'pre':Pre_extract,
    'train': train_Dataset,
    'test': test_Dataset
}

def names():
    return sorted(__factory.keys())


def create(name, data_dir = None, extend_dir = None,trial=1, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](data_dir = data_dir, extend_dir = extend_dir, trial=trial, *args, **kwargs)


def get_dataset(name, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name,*args, **kwargs)

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None
        self.new_epoch()

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            # 如果迭代器有下一个元素，则返回下一个数据批次。
            return next(self.iter)
        except:
            # 如果迭代器没有下一个元素（即已经遍历完一个训练周期）。
            # 将迭代器重置为数据加载器的迭代器，以便在下一个训练周期重新遍历数据集。
            self.iter = iter(self.loader)
            # 返回下一个数据批次。
            return next(self.iter)






