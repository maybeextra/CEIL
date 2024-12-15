import numpy as np
import torch.utils.data as data


class Pre_extract(data.Dataset):
    def __init__(self, data_dir, extend_dir, data_type, transform_1, transform_2=None, **kwargs):
        data_type = 'rgb' if data_type == 1 else 'ir'
        self.train_image = np.load(data_dir + f'{extend_dir}train_{data_type}_resized_img.npy')
        self.train_label = np.load(data_dir + f'{extend_dir}train_{data_type}_resized_label.npy')
        self.train_cam = np.load(data_dir + f'{extend_dir}train_{data_type}_resized_cam.npy')
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.len = len(self.train_image)

    def __getitem__(self, index):
        image, label, cam = self.train_image[index], self.train_label[index], self.train_cam[index]
        img_trans1 = self.transform_1(image)
        items = [img_trans1, label, cam]
        if self.transform_2 is not None:
            img_trans2 = self.transform_2(image)
            items.insert(1, img_trans2)
        return tuple(items)

    def __len__(self):
        return self.len
