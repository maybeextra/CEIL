from torch.utils.data.sampler import Sampler


class Base_create:
    def __init__(self, batch_size,num_instances):
        self.num_iter = None
        self.batch_size = batch_size
        self.num_instances = num_instances

        self.proxy2instance = None
        self.cam2proxy = None
        self.unique_cameras = None

    class BaseSampler(Sampler):
        def __init__(self, sequence):
            self.sequence = sequence

        def __iter__(self):
            # 返回经过批次打乱后的数据索引的迭代器。
            return iter(self.sequence)

        def __len__(self):
            return len(self.sequence)

    def update(self,memory):
        self.unique_cameras = memory.unique_cameras
        self.cam2proxy = memory.cam2proxy
        self.proxy2instance = memory.proxy2instance


    def refresh(self):
        self.proxy2instance = None
        self.cam2proxy = None
        self.unique_cameras = None