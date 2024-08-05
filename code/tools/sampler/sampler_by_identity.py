import numpy as np
import math
from code.tools.sampler.base_sampler_creat import Base_create
import torch

class Sampler_by_identity(Base_create):
    def __init__(self, batch_size, num_instances, num_iter):
        super().__init__(batch_size,num_instances)
        self.num_iter = num_iter
    def shuffle(self, ret, magnification=1):
        uni_label = torch.Tensor(range(len(self.proxy2instance)))

        batch_size = self.batch_size * magnification
        num_proxy_per_batch = math.floor(batch_size / self.num_instances)

        for j in range(self.num_iter):
            batch_idx = np.random.choice(uni_label, num_proxy_per_batch, replace=False).astype(int)
            for i in range(num_proxy_per_batch):

                sample_index = np.random.choice(self.proxy2instance[batch_idx[i]], self.num_instances)

                if j == 0 and i == 0:
                    index = sample_index
                else:
                    index = np.hstack((index, sample_index))
        return index

    def creat(self, magnification =1):
        sequence = self.shuffle(np.array([]).astype(int), magnification)
        sampler = self.BaseSampler(sequence)
        return sampler