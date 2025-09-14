import logging
import time

from code.tools.utils.pre_utils import load_model, do_pre, creat_config, setup_dbscan
from code.train.trainer.trainer_stage1 import Trainer_stage1

from train.train import do_train


def main_worker(args, writer):
    start_time = time.monotonic()

    model, optimizer, scheduler, scaler, start_epoch = load_model(args, f'{args.base_dir}{args.resume_path}', args.resume)

    cluster = setup_dbscan(args)

    logging.info("==> Start train stage1")
    trainer_stage1 = Trainer_stage1(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        args=args,
        kind='train',
    )
    do_train(
        start_epoch = start_epoch,
        args = args,
        trainer = trainer_stage1,
        cluster = cluster,
        stage=1
    )

    logging.info(f'stage1 running time: {time.monotonic() - start_time}')
    writer.close()

if __name__ == '__main__':
    kind = 'sysu'

    config = creat_config(kind, 'stage1')
    writer = do_pre(config, 'train')
    main_worker(config, writer)
