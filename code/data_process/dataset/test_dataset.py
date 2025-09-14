import torch.utils.data as data


class test_Dataset(data.Dataset):
    def __init__(self, test_img, test_label, test_cam, transform_1, transform_2=None, **kwargs):
        self.test_image = test_img
        self.test_label = test_label
        self.test_cam = test_cam
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __getitem__(self, index):
        image, label, cam = self.test_image[index], self.test_label[index], self.test_cam[index]
        img_trans1 = self.transform_1(image)
        items = [img_trans1, label, cam]
        if self.transform_2 is not None:
            img_trans2 = self.transform_2(image)
            items.insert(1, img_trans2)
        return tuple(items)

    def __len__(self):
        return len(self.test_image)