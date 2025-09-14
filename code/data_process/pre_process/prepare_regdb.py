import os

import numpy as np
from PIL import Image

# You only need to change this line to your dataset download path
data_dir = '../dataset/RegDB/'
if not os.path.isdir(data_dir):
    print('please change the data_dir')

fix_image_width = 144
fix_image_height = 288

base_target = f'{data_dir}{fix_image_width}x{fix_image_height}/'
if not os.path.isdir(base_target):
    os.mkdir(base_target)

def load(data_type, load_model):
    data_re = 'thermal' if data_type == 'ir' else 'visible'
    for trial in range(1,11):
        resource_file_path = os.path.join(data_dir, f'idx/{load_model}_{data_re}_' + str(trial) + '.txt')
        with open(resource_file_path) as f:
            data_file_list = open(resource_file_path, 'rt').read().splitlines()
            # Get full list of image and labels
            files_ir = [data_dir + s for s in data_file_list]

        train_image = []
        train_label = []
        train_cam = []

        for file_path in files_ir:
            src_path = file_path.split(' ')[0]
            ID = file_path.split(' ')[1]

            if data_type == 'rgb':
                train_cam.append(0)
            else:
                train_cam.append(1)
            image = Image.open(src_path).convert('RGB')
            image = image.resize((fix_image_width, fix_image_height), Image.Resampling.LANCZOS)
            train_image.append(np.array(image))
            train_label.append(ID)

        train_image = np.array(train_image)
        train_label = np.array(train_label).astype(np.int64)
        train_cam = np.array(train_cam).astype(np.int64)

        target = f'{base_target}/{trial}'
        if not os.path.isdir(target):
            os.mkdir(target)

        np.save(f'{target}/{load_model}_{data_type}_resized_img.npy', train_image)
        np.save(f'{target}/{load_model}_{data_type}_resized_label.npy', train_label)
        np.save(f'{target}/{load_model}_{data_type}_resized_cam.npy', train_cam)

load("ir", "train")
load("rgb", "train")
load("ir", "test")
load("rgb", "test")