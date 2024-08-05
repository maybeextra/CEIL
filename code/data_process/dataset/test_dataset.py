import torch.utils.data as data
import numpy as np

class test_Dataset(data.Dataset):
    def __init__(self, test_img, test_label, test_cam, transform=None, **kwargs):
        self.test_image = test_img
        self.test_label = test_label
        self.test_cam = test_cam
        self.transform = transform

    def __getitem__(self, index):
        image, label, cam = self.test_image[index], self.test_label[index], self.test_cam[index]
        image = self.transform(image)
        return image, label, cam

    def __len__(self):
        return len(self.test_image)