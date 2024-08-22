import numpy as np
from PIL import Image
import os

data_path = '../dataset/SYSU-MM01/'

rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']
cameras2id = {
    'cam1': 0,
    'cam2': 1,
    'cam3': 2,
    'cam4': 3,
    'cam5': 4,
    'cam6': 5
}
fix_image_width = 144
fix_image_height = 288

def get_train_id():
    # load id info
    file_path_train = os.path.join(data_path, 'exp/train_id.txt')
    file_path_val = os.path.join(data_path, 'exp/val_id.txt')
    with open(file_path_train, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_train = ["%04d" % x for x in ids]

    with open(file_path_val, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_val = ["%04d" % x for x in ids]

    # combine train and val split
    id_train.extend(id_val)
    return id_train

def get_path(_id, cameras):
    files = []
    for id in sorted(_id):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files.extend(new_files)
    return files

def relabel(files):
    # relabel
    pid_container = set()
    for img_path in files:
        pid = int(img_path[-13:-9])
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    return pid2label

def read_imgs(files_path, pid2label):
    train_img = []
    train_label = []
    train_cam = []

    for img_path in files_path:
        # img
        img = Image.open(img_path).convert('RGB')
        img = img.resize((fix_image_width, fix_image_height), Image.Resampling.LANCZOS)
        pix_array = np.array(img)
        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)

        # cam
        cam = img_path.split('/')[3]
        cam = cameras2id[cam]
        train_cam.append(cam)
    return np.array(train_img), np.array(train_label), np.array(train_cam)


train_id = get_train_id()
files_rgb = get_path(train_id, rgb_cameras)
files_ir = get_path(train_id, ir_cameras)
pid2label_rgb = relabel(files_rgb)
pid2label_ir = relabel(files_ir)

# rgb imges
train_img_rgb, train_label_rgb, train_cam_rgb = read_imgs(files_rgb, pid2label_rgb)
# ir imges
train_img_ir, train_label_ir, train_cam_ir = read_imgs(files_ir, pid2label_ir)

target = f'{data_path}{fix_image_width}x{fix_image_height}'
if not os.path.isdir(target):
    os.mkdir(target)

np.save(f'{target}/train_rgb_resized_img.npy', train_img_rgb)
np.save(f'{target}/train_rgb_resized_label.npy', train_label_rgb)
np.save(f'{target}/train_rgb_resized_cam.npy', train_cam_rgb)

np.save(f'{target}/train_ir_resized_img.npy', train_img_ir)
np.save(f'{target}/train_ir_resized_label.npy', train_label_ir)
np.save(f'{target}/train_ir_resized_cam.npy', train_cam_ir)