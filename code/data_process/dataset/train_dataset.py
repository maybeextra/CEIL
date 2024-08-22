# encoding: utf-8
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
class train_Dataset(data.Dataset):
    def __init__(self, train_image, proxy, label, cam, transform_1, transform_2=None, cross_proxy=None, cross_label=None, cross_class=None, **kwargs):
        self.train_image = train_image
        self.train_proxy = proxy
        self.train_label = label
        self.train_cam = cam
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.cross_proxy = cross_proxy
        self.cross_label = cross_label
        self.cross_class = cross_class
    def __getitem__(self, index):
        image = self.train_image[index]
        proxy = self.train_proxy[index]
        label = self.train_label[index]
        cam = self.train_cam[index]

        img_trans1 = self.transform_1(image)
        items = [img_trans1, proxy, label, cam]

        if self.transform_2 is not None:
            img_trans2 = self.transform_2(img_trans1) # img_trans1 img
            items.insert(1, img_trans2)

        if self.cross_proxy is not None and self.cross_label is not None and self.cross_class is not None:
            cross_proxy = self.cross_proxy[index]
            cross_label = self.cross_label[index]
            cross_class = self.cross_class[index]
            items.extend([cross_proxy, cross_label, cross_class])

        return tuple(items)

    def __len__(self):
        return len(self.train_image)

