
import numpy as np
import torch

import random
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from sklearn.cluster import DBSCAN

from code.tools.output.log_utils import create_logger,show_train_prepare_info
from code.tools.optim.make_optimizer import make_optimizer
from code.tools.optim.lr_scheduler import WarmupMultiStepLR
from code import models

import logging
import yaml
import easydict

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def merge_configs(config_a, config_b):
    config_a.update(config_b)
    return config_a

def create_model(backbone, config):
    model = models.create(backbone, config)
    return model

def load_model(args, path, resume):
    model = create_model(args.backbone, args)

    if resume:
        logging.info("==> Load train_prepare model")
        checkpoint = torch.load(f'{path}')
        start_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info("==> Start New Model")
        start_epoch = 1

    model.cuda()
    optimizer = make_optimizer(args.base_lr, args.weight_decay, args.bias_lr_factor, args.weight_decay_bias, model)
    scheduler = WarmupMultiStepLR(optimizer, args.steps, args.gamma, args.warmup_factor, args.warmup_iters, args.warmup_method)
    scaler = torch.cuda.amp.GradScaler()

    if resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        #scaler.load_state_dict(checkpoint['scaler'])
    return model, optimizer, scheduler, scaler, start_epoch

def creat_config(kind, stage):
    config_common_path = 'config/common.yaml'
    config_base_path = f'config/{kind}/base.yaml'
    config_stage_path = f'config/{kind}/{stage}.yaml'

    config_common = load_config(config_common_path)
    config_base = load_config(config_base_path)
    config_stage = load_config(config_stage_path)

    config = merge_configs(config_common, config_base)
    config = merge_configs(config, config_stage)
    config = easydict.EasyDict(config)
    return config

def do_pre(args, kind):
    log_path = args.base_dir + args.log_name + args.logs_dir
    create_logger(log_path)
    show_train_prepare_info(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    else:
        cudnn.benchmark = True

    if kind == 'train':
        writer = SummaryWriter(log_dir=log_path)
        return writer

def setup_dbscan(args):
    logging.info(f'==> Creat DBSCAN')
    cluster = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='precomputed', n_jobs=-1)
    return cluster