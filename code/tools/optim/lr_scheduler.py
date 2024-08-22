
from bisect import bisect_right
import torch
import math

class WarmupMultiStepCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, multistep, multistep_min_lr, max_epochs, min_warmup_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.multistep = multistep
        self.multistep_min_lr = multistep_min_lr
        self.min_lr = min_warmup_lr
        self.max_epochs = max_epochs
        super(WarmupMultiStepCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性warmup
            lr = (self.base_lrs[0] - self.min_lr) * (self.last_epoch / self.warmup_epochs) + self.min_lr
        elif self.warmup_epochs <= self.last_epoch < self.multistep[-1]:
            base_epoch = self.warmup_epochs
            max_epoch = self.multistep[0]
            base_ir = self.base_lrs[0]
            min_ir = self.multistep_min_lr[0]

            for i,max_epoch in enumerate(self.multistep):
                if self.last_epoch > max_epoch:
                    base_epoch = max_epoch
                    base_ir = min_ir
                    max_epoch = self.multistep[i + 1]
                    min_ir = self.multistep_min_lr[i+1]
                else:
                    break
            lr = min_ir + (base_ir - min_ir) * (1 + math.cos(
                math.pi * (self.last_epoch - base_epoch) / (max_epoch - base_epoch))) / 2
        else:
            base_epoch = self.multistep[-1]
            base_ir = self.multistep_min_lr[-1]

            lr = self.min_lr + (base_ir - self.min_lr) * (1 + math.cos(
                math.pi * (self.last_epoch - base_epoch) / (self.max_epochs - base_epoch))) / 2

        return [lr for _ in self.base_lrs]

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,  # steps
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
