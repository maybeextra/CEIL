import torch.utils.data as data
import torch
import logging
import time

from code import data_process
from code.data_process import IterLoader
from code.tools.output.log_utils import show_dateset_info, save_model
from code.tools.utils.train_utils import map_crosses, mask_outlier, rename, extract_features, get_proxy
from code.tools import trans

def extract_and_cluster(mode, epoch, args, trainer, cluster):
    kind = 'RGB' if mode == 1 else 'IR'
    features, pseudo_labels, num_clusters, cams, n_class, n_instance, images, real_labels = extract_features(mode, args, trainer, cluster)
    proxy_labels, num_proxies = get_proxy(pseudo_labels, cams)
    logging.info(f"Epoch[{epoch}]==> Create {kind} cluster {num_clusters} classes, proxy {num_proxies} kinds")

    info = [features, pseudo_labels, proxy_labels, cams, images, n_class, n_instance, real_labels]
    return info

def fitter(rgb_info, ir_info):
    [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb, real_labels_rgb] = rgb_info
    [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir, real_labels_ir] = ir_info

    mask_rgb = mask_outlier(proxy_labels_rgb)
    mask_ir = mask_outlier(proxy_labels_ir)

    real_labels_rgb, real_labels_ir = rename(real_labels_rgb[mask_rgb]), rename(real_labels_ir[mask_ir])
    proxy_labels_rgb, proxy_labels_ir = rename(proxy_labels_rgb[mask_rgb]), rename(proxy_labels_ir[mask_ir])
    pseudo_labels_rgb, pseudo_labels_ir = rename(pseudo_labels_rgb[mask_rgb]), rename(pseudo_labels_ir[mask_ir])

    cams_rgb, cams_ir = cams_rgb[mask_rgb], cams_ir[mask_ir]
    features_rgb, features_ir = features_rgb[mask_rgb], features_ir[mask_ir]
    images_rgb, images_ir = images_rgb[mask_rgb], images_ir[mask_ir]
    del mask_rgb, mask_ir

    rgb_info = [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb, real_labels_rgb]
    ir_info = [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir, real_labels_ir]
    return rgb_info,ir_info

def prepare_datasets(args, rgb_info, ir_info, memory):
    [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb, real_labels_rgb] = rgb_info
    [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir, real_labels_ir] = ir_info

    transform_train_rgb_1, transform_train_rgb_2, transform_train_ir = trans.create_transform(args.dataset, args.data_enhancement)

    magnification = 2 if transform_train_rgb_2 else 1

    i2r_proxy, r2i_proxy, i2r_label, r2i_label = None, None, None, None
    if len(memory.i2r_proxy) != 0 and len(memory.i2r_proxy) != 0:
        i2r_proxy, r2i_proxy = map_crosses(proxy_labels_ir, memory.i2r_proxy), map_crosses(proxy_labels_rgb, memory.r2i_proxy)
        i2r_label, r2i_label = map_crosses(i2r_proxy, memory.rgb_memory.proxy2label), map_crosses(r2i_proxy, memory.ir_memory.proxy2label)

    i2r_class, r2i_class = None, None
    if len(memory.i2r_label) != 0 and len(memory.r2i_label) != 0:
        i2r_class, r2i_class = map_crosses(pseudo_labels_ir, memory.i2r_label), map_crosses(pseudo_labels_rgb, memory.r2i_label)


    train_dataset_rgb = data_process.create(
        'train',
        train_image=images_rgb, proxy=proxy_labels_rgb, label=pseudo_labels_rgb, cam=cams_rgb, cross_proxy=r2i_proxy, cross_label=r2i_label, cross_class=r2i_class,
        transform_1=transform_train_rgb_1, transform_2=transform_train_rgb_2
    )
    train_dataset_ir = data_process.create(
        'train',
        train_image=images_ir, proxy=proxy_labels_ir, label=pseudo_labels_ir, cam=cams_ir, cross_proxy=i2r_proxy, cross_label=i2r_label, cross_class=i2r_class,
        transform_1=transform_train_ir
    )

    show_dateset_info(
        n_class_rgb, n_class_ir, n_instance_rgb, n_instance_ir,
        len(torch.unique(pseudo_labels_rgb)), len(torch.unique(pseudo_labels_ir)), len(train_dataset_rgb), len(train_dataset_ir),
    )

    return train_dataset_rgb, train_dataset_ir, magnification

def create_dataloaders(train_dataset_rgb, train_dataset_ir, args, rgb_sampler, ir_sampler, magnification = 1):
    batch_size = args.train_batch_size
    rgb_trainloader = IterLoader(
        data.DataLoader(train_dataset_rgb, batch_size=batch_size, sampler=rgb_sampler,
                        num_workers=args.train_num_workers,
                        drop_last=True)
    )


    ir_trainloader = IterLoader(
        data.DataLoader(train_dataset_ir, batch_size=batch_size * magnification, sampler=ir_sampler,
                        num_workers=args.train_num_workers,
                        drop_last=True)
    )


    return rgb_trainloader, ir_trainloader


def valid(args, trainer, epoch, best_cmc):
    start = time.time()
    logging.info(f"Epoch[{epoch}] Test start")
    cmc_rank_1 = trainer.valid(args, args.test_mode_1, args.mode_1)
    cmc_rank_2 = trainer.valid(args, args.test_mode_2, args.mode_2)
    mean_cmc = (cmc_rank_1+cmc_rank_2) / 2
    if mean_cmc > best_cmc:
        logging.info(
            f"Epoch [{epoch}], save better model Rank-1: {mean_cmc:.2%} to replace original model Rank-1: {best_cmc:.2%}")
        best_cmc = mean_cmc
        save_model(args, trainer, epoch, 0)

    logging.info(f"Epoch [{epoch}] Test end, time cost {time.time() - start}")
    return best_cmc

def save(args, trainer, epoch):
    logging.info(f"Epoch [{epoch}], save model for record")
    save_model(args, trainer, epoch, 1)