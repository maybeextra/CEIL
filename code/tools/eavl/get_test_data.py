from __future__ import print_function, absolute_import
import random
from code import data_process

import numpy as np
import torch.utils.data as data
from PIL import Image
from code.tools.eavl.img_get import datasets_info, get_path
from code.tools import trans

def process_images(data_path, img_w, img_h, mode, dataset, data_type, trial=1):
    random.seed(trial)

    if dataset not in datasets_info:
        raise ValueError(f"Invalid dataset '{dataset}'. Available datasets: {list(datasets_info.keys())}")

    dataset_conf = datasets_info[dataset]
    extend_dir = dataset_conf['extend_dir']
    get_image_info = dataset_conf['get_image_info']
    modes_conf = dataset_conf['modes']

    # Determine the cameras based on dataset, data_type, and mode
    if data_type not in ('query', 'gallery', 'image'):
        raise ValueError(f"Invalid data_type '{data_type}'. Must be 'query', 'gallery', or 'select'.")

    if dataset == 'SYSU':
        if data_type == 'query':
            cameras = modes_conf['query']['cameras']
        else:
            if mode not in modes_conf[data_type]:
                raise ValueError(f"Invalid mode '{mode}' for SYSU dataset.")
            cameras = modes_conf[data_type][mode]['cameras']

    elif dataset == 'LLCM':
        if mode not in modes_conf:
            raise ValueError(f"Invalid mode '{mode}' for LLCM dataset.")
        if data_type not in modes_conf[mode]:
            raise ValueError(f"Invalid data_type '{data_type}' for mode '{mode}' in LLCM dataset.")
        cameras = modes_conf[mode][data_type]['cameras']

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

    img_list = np.load(data_path + f'{extend_dir}/{trial}/test_{kind}_resized_img.npy')
    id_list = np.load(data_path + f'{extend_dir}/{trial}/test_{kind}_resized_label.npy')
    cam_list = np.load(data_path + f'{extend_dir}/{trial}/test_{kind}_resized_cam.npy')
    return img_list, id_list, cam_list

def creat_test_data(args, trial=1, mode=None, kind=None):
    if args.dataset == 'SYSU' or args.dataset == 'LLCM':
        img, label, cam = process_images(data_path=args.data_dir, img_w=args.img_w, img_h=args.img_h, mode=mode, dataset=args.dataset,data_type=kind, trial=trial)
    elif args.dataset == 'RegDB':
        img, label, cam = process_test_regdb(data_path=args.data_dir, trial=trial, mode=mode, kind=kind)
    else:
        raise ValueError('Please input correct dataset!!')

    transform_extract, transform_extract_f = trans.creat_extract(args.dataset)
    dataset = data_process.create(
        name='test',
        test_img=img, test_label=label, test_cam=cam,
        transform_1=transform_extract, transform_2=transform_extract_f
    )

    dataloader = data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False,
                                   num_workers=args.test_num_workers)

    return dataloader