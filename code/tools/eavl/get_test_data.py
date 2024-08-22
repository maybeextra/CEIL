from __future__ import print_function, absolute_import
import os
import numpy as np
import random
import torch.utils.data as data
from code.tools.trans.transform_sysu import transform_extract_SYSU
from code.tools.trans.transform_regdb import transform_extract_RegDB
from code.tools.trans.transform_llcm import transform_extract_LLCM
from code import data_process
from PIL import Image

def get_path(data_path, extend_dir, cameras, _type):
    file_path = os.path.join(data_path, f'{extend_dir}/test_id.txt')
    files = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])

                if _type == 'query':
                    files.extend(new_files)
                elif _type == 'gallery':
                    files.append(random.choice(new_files))
                elif _type == 'image':
                    files.extend(random.sample(new_files, 5))
                else:
                    raise
    return files

def get_image_info_sysu(img_path):
    camid = int(img_path[-15])
    pid = int(img_path[-13:-9])
    return camid, pid

def get_image_info_llcm(img_path):
    camid = int(img_path.split('cam')[1][0])
    pid = int(img_path.split('cam')[1][2:6])
    return camid, pid

def process_images(data_path, img_w, img_h, mode, dataset, data_type, trial=1):
    random.seed(trial)

    if dataset == 'SYSU':
        if data_type == 'query':
            cameras = ['cam3', 'cam6']
        elif data_type == 'gallery' or data_type == 'image':
            if mode == 'all':
                cameras = ['cam1', 'cam2', 'cam4', 'cam5']
            elif mode == 'indoor':
                cameras = ['cam1', 'cam2']
            else:
                raise ValueError("Invalid mode for SYSU dataset")
        get_image_info = get_image_info_sysu
        extend_dir = 'exp'
    elif dataset == 'LLCM':
        if data_type == 'query':
            if mode == 'rgb2ir':
                cameras = ['test_vis/cam1', 'test_vis/cam2', 'test_vis/cam3', 'test_vis/cam4', 'test_vis/cam5',
                           'test_vis/cam6', 'test_vis/cam7', 'test_vis/cam8', 'test_vis/cam9']
            elif mode == 'ir2rgb':
                cameras = ['test_nir/cam1', 'test_nir/cam2', 'test_nir/cam4', 'test_nir/cam5', 'test_nir/cam6',
                           'test_nir/cam7', 'test_nir/cam8', 'test_nir/cam9']
            else:
                raise ValueError("Invalid mode for LLCM dataset")
        elif data_type == 'gallery':
            if mode == 'rgb2ir':
                cameras = ['test_nir/cam1', 'test_nir/cam2', 'test_nir/cam4', 'test_nir/cam5', 'test_nir/cam6',
                           'test_nir/cam7', 'test_nir/cam8', 'test_nir/cam9']
            elif mode == 'ir2rgb':
                cameras = ['test_vis/cam1', 'test_vis/cam2', 'test_vis/cam3', 'test_vis/cam4', 'test_vis/cam5',
                           'test_vis/cam6', 'test_vis/cam7', 'test_vis/cam8', 'test_vis/cam9']
            else:
                raise ValueError("Invalid mode for LLCM dataset")
        get_image_info = get_image_info_llcm
        extend_dir = 'idx'
    else:
        raise ValueError("Invalid dataset")

    files = get_path(data_path, extend_dir, cameras, data_type)

    img_list, id_list, cam_list = [], [], []
    for img_path in files:
        camid, pid = get_image_info(img_path)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((img_w, img_h), Image.Resampling.LANCZOS)
        img_list.append(np.array(image))
        id_list.append(pid)
        cam_list.append(camid - 1)

    return img_list, np.array(id_list), np.array(cam_list)

def process_test_regdb(data_path, mode, kind, trial=1, extend_dir='144x288'):
    mode_kind_map = {
        ('ir2rgb', 'gallery'): 'rgb',
        ('rgb2ir', 'gallery'): 'ir',
        ('ir2rgb', 'query'): 'ir',
        ('rgb2ir', 'query'): 'rgb'
    }
    if (mode, kind) in mode_kind_map:
        kind = mode_kind_map[(mode, kind)]

    train_image = np.load(data_path + f'{extend_dir}/{trial}/test_{kind}_resized_img.npy')
    train_label = np.load(data_path + f'{extend_dir}/{trial}/test_{kind}_resized_label.npy')
    train_cam = np.load(data_path + f'{extend_dir}/{trial}/test_{kind}_resized_cam.npy')
    return train_image, train_label, train_cam


def creat_test_data(args, trial=1, mode=None, kind=None):
    if args.dataset == 'SYSU':
        img, label, cam = process_images(data_path=args.data_dir, img_w=args.img_w, img_h=args.img_h, mode=mode, dataset='SYSU',data_type=kind, trial=trial)
        transform = transform_extract_SYSU
    elif args.dataset == 'RegDB':
        img, label, cam = process_test_regdb(data_path=args.data_dir, trial=trial, mode=mode, kind=kind)
        transform = transform_extract_RegDB
    elif args.dataset == 'LLCM':
        img, label, cam = process_images(data_path=args.data_dir, img_w=args.img_w, img_h=args.img_h, mode=mode, dataset='LLCM',data_type=kind, trial=trial)
        transform = transform_extract_SYSU
    else:
        raise ValueError('Please input correct dataset!!')

    dataset = data_process.create(name='test', test_img=img, test_label=label, test_cam=cam,
                                   transform=transform)

    dataloader = data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False,
                                   num_workers=args.test_num_workers)

    return dataloader