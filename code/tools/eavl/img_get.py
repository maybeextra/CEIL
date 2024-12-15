import os
import random

def get_image_info_sysu(img_path):
    camid = int(img_path[-15])
    pid = int(img_path[-13:-9])
    return camid, pid

def get_image_info_llcm(img_path):
    camid = int(img_path.split('cam')[1][0])
    pid = int(img_path.split('cam')[1][2:6])
    return camid, pid

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
                elif _type == 'select':
                    files.extend(random.sample(new_files, 5))
                else:
                    raise
    return files

# Centralized dataset configurations
datasets_info = {
    'SYSU': {
        'extend_dir': 'exp',
        'get_image_info': get_image_info_sysu,
        'modes': {
            'query': {
                'cameras': ['cam3', 'cam6']
            },
            'gallery': {
                'all': {
                    'cameras': ['cam1', 'cam2', 'cam4', 'cam5']
                },
                'indoor': {
                    'cameras': ['cam1', 'cam2']
                }
            },
            'select': {
                'all': {
                    'cameras': ['cam1', 'cam2', 'cam4', 'cam5']
                },
                'indoor': {
                    'cameras': ['cam1', 'cam2']
                }
            }
        }
    },
    'LLCM': {
        'extend_dir': 'idx',
        'get_image_info': get_image_info_llcm,
        'modes': {
            'rgb2ir': {
                'query': {
                    'cameras': [
                        'test_vis/cam1', 'test_vis/cam2', 'test_vis/cam3', 'test_vis/cam4',
                        'test_vis/cam5', 'test_vis/cam6', 'test_vis/cam7', 'test_vis/cam8',
                        'test_vis/cam9'
                    ]
                },
                'gallery': {
                    'cameras': [
                        'test_nir/cam1', 'test_nir/cam2', 'test_nir/cam4', 'test_nir/cam5',
                        'test_nir/cam6', 'test_nir/cam7', 'test_nir/cam8', 'test_nir/cam9'
                    ]
                }
            },
            'ir2rgb': {
                'query': {
                    'cameras': [
                        'test_nir/cam1', 'test_nir/cam2', 'test_nir/cam4', 'test_nir/cam5',
                        'test_nir/cam6', 'test_nir/cam7', 'test_nir/cam8', 'test_nir/cam9'
                    ]
                },
                'gallery': {
                    'cameras': [
                        'test_vis/cam1', 'test_vis/cam2', 'test_vis/cam3', 'test_vis/cam4',
                        'test_vis/cam5', 'test_vis/cam6', 'test_vis/cam7', 'test_vis/cam8',
                        'test_vis/cam9'
                    ]
                }
            }
        }
    }
}