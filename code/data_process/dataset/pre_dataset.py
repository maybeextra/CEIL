import torch.utils.data as data
import numpy as np


class Pre_extract(data.Dataset):
    def __init__(self, data_dir, extend_dir, transform, data_type, **kwargs):
        data_type = 'rgb' if data_type == 1 else 'ir'
        self.train_image = np.load(data_dir + f'{extend_dir}train_{data_type}_resized_img.npy')
        self.train_label = np.load(data_dir + f'{extend_dir}train_{data_type}_resized_label.npy')
        self.train_cam = np.load(data_dir + f'{extend_dir}train_{data_type}_resized_cam.npy')
        self.transform = transform
        self.len = len(self.train_image)

    def __getitem__(self, index):
        image, label, cam = self.train_image[index], self.train_label[index], self.train_cam[index]
        image = self.transform(image)
        return image, label, cam

    def __len__(self):
        return self.len
