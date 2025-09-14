import math

import numpy as np
import torch
from code.tools.sampler.base_sampler_creat import Base_create


class Sampler_by_cam(Base_create):
    def __init__(self, batch_size, num_instances, num_iter):
        super().__init__(batch_size,num_instances)
        self.num_iter = num_iter

    @staticmethod
    def select(source, num):
        if len(source) >= num:
            sample = np.random.choice(source, num, replace=False)
        else:
            random_index = torch.randperm(source.size(0))
            sample = source[random_index]
            remaining = num - len(sample)
            sample = np.concatenate([sample, np.random.choice(source, remaining)])
        return sample

    def shuffle(self, ret, magnification=1):
        batch_size = self.batch_size * magnification
        batch_idxes = []

        num_instance_per_cam = math.floor(batch_size / len(self.unique_cameras))
        num_proxy_per_batch = math.floor(num_instance_per_cam / self.num_instances)

        while len(batch_idxes) < self.num_iter:
            mini_batch = np.array([]).astype(int)
            for cam in self.unique_cameras:
                per_cam_proxy = self.cam2proxy[cam].cpu()
                proxies = self.select(per_cam_proxy, num_proxy_per_batch)

                for proxy in proxies:
                    sample = self.select(self.proxy2instance[proxy], self.num_instances)
                    mini_batch = np.concatenate([mini_batch, sample])

            if len(mini_batch) < batch_size:
                remaining = batch_size - len(mini_batch)
                mini_batch = np.concatenate([mini_batch, np.random.choice(mini_batch, remaining)])

            batch_idxes.append(mini_batch)

        batch_idxes = np.concatenate(batch_idxes, axis=0)
        ret = np.hstack((ret, batch_idxes))
        return ret

    def creat(self, magnification =1):
        sequence = self.shuffle(np.array([]).astype(int), magnification)
        sampler = self.BaseSampler(sequence)
        return sampler