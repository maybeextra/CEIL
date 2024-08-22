from __future__ import print_function, absolute_import
from PIL import Image
import numpy as np
import re
import os
# You only need to change this line to your dataset download path
data_dir = '../dataset/LLCM/'
fix_image_width = 144
fix_image_height = 288

def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image_path = []
        file_label = []
        file_cam = []
        for s in data_file_list:
            s = s.split(' ')
            file_image_path.append(s[0])
            file_label.append(int(s[1]))
            match = re.search(r'c(\d+)', s[0])
            file_cam.append(int(match.group(1)))

    return file_image_path, file_label, file_cam

def read_train_file(datatype):

    train_list = data_dir + f'idx/train_{datatype}.txt'

    file_image_path, train_label, train_cam = load_data(train_list)

    train_image = []
    for i in range(len(file_image_path)):
        img = Image.open(data_dir + file_image_path[i])
        img = img.resize((fix_image_width, fix_image_height), Image.Resampling.LANCZOS)
        pix_array = np.array(img)
        train_image.append(pix_array)

    train_image = np.array(train_image)
    train_label = np.array(train_label)
    train_cam = np.array(train_cam)

    return train_image, train_label, train_cam

# rgb imges
train_img_rgb, train_label_rgb, train_cam_rgb = read_train_file('vis')
# ir imges
train_img_ir, train_label_ir, train_cam_ir = read_train_file('nir')

target = f'{data_dir}{fix_image_width}x{fix_image_height}'
if not os.path.isdir(target):
    os.mkdir(target)

np.save(f'{target}/train_rgb_resized_img.npy', train_img_rgb)
np.save(f'{target}/train_rgb_resized_label.npy', train_label_rgb)
np.save(f'{target}/train_rgb_resized_cam.npy', train_cam_rgb)

np.save(f'{target}/train_ir_resized_img.npy', train_img_ir)
np.save(f'{target}/train_ir_resized_label.npy', train_label_ir)
np.save(f'{target}/train_ir_resized_cam.npy', train_cam_ir)


