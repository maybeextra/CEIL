import time

import torch
import logging
from code.train.train_models import extract_and_cluster, fitter, prepare_datasets, create_dataloaders, valid, save
from code.tools import sampler
import gc

def do_train(
        args,
        trainer,
        start_epoch,
        cluster,
        stage
):
    best_cmc = 0
    end_epoch = args.max_epoch

    rgb_sampler_creat = sampler.creat(args.sampler, args.train_batch_size, args.train_num_instances, args.num_iter)
    ir_sampler_creat = sampler.creat(args.sampler, args.train_batch_size, args.train_num_instances, args.num_iter)

    for epoch in range(start_epoch, end_epoch+1):
        gc.collect()
        torch.cuda.empty_cache()
        trainer.epoch = epoch
        ####################################################################################################################################################################
        # pre
        ####################################################################################################################################################################
        logging.info(f'######################################################')
        logging.info(f'Epoch[{epoch}]==> Create pseudo labels')
        with torch.no_grad():
            rgb_info = extract_and_cluster(1, epoch, args, trainer, cluster)
            ir_info = extract_and_cluster(2, epoch, args, trainer, cluster)
        rgb_info, ir_info = fitter(rgb_info, ir_info)

        logging.info(f'Epoch[{epoch}]==> Update Memory')
        trainer.memory.update(rgb_info, ir_info)

        if stage == 2:
            logging.info(f'Epoch[{epoch}]==> Start cross modal matching')
            trainer.memory.creat_cross()

        logging.info(f'Epoch[{epoch}]==> Creat dataset')
        train_dataset_rgb, train_dataset_ir, magnification = prepare_datasets(args, rgb_info, ir_info, trainer.memory)
        del rgb_info, ir_info

        logging.info(f'Epoch[{epoch}]==> Creat sampler')
        rgb_sampler_creat.update(trainer.memory.rgb_memory)
        ir_sampler_creat.update(trainer.memory.ir_memory)
        rgb_sampler = rgb_sampler_creat.creat()
        ir_sampler = ir_sampler_creat.creat(magnification)

        logging.info(f'Epoch[{epoch}]==> Creat dataloader')
        rgb_trainloader, ir_trainloader = create_dataloaders(train_dataset_rgb, train_dataset_ir, args, rgb_sampler, ir_sampler, magnification)
        ####################################################################################################################################################################
        # train
        ####################################################################################################################################################################
        logging.info(f'Epoch[{epoch}]==> Train stage start')
        start = time.time()
        trainer.run(rgb_trainloader, ir_trainloader, magnification)
        logging.info(f"Epoch[{epoch}]==> Train end, time cost {time.time() - start}")

        rgb_sampler_creat.refresh()
        ir_sampler_creat.refresh()
        trainer.memory.refresh()
        gc.collect()
        torch.cuda.empty_cache()
        del train_dataset_rgb,train_dataset_ir,rgb_trainloader,ir_trainloader
        ####################################################################################################################################################################
        # valid and save
        ####################################################################################################################################################################
        if epoch % args.save_epoch == 0 or (epoch == end_epoch):
            save(args, trainer, epoch)
        if epoch % args.eval_step == 0 or (epoch == end_epoch):
            best_cmc = valid(args, trainer, epoch, best_cmc)