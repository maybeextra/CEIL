import logging
import time

from code.tools.utils.pre_utils import load_model, do_pre, creat_config, setup_dbscan
from code.train.trainer.trainer_stage2 import Trainer_stage2

from train.train import do_train


def main_worker(args, writer):
    start_time = time.monotonic()

    model, optimizer, scheduler, scaler, start_epoch = load_model(args, f'{args.base_dir}{args.resume_path}', True)

    cluster = setup_dbscan(args)

    logging.info("==> Start train stage2")
    trainer_stage2 = Trainer_stage2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        args=args,
        kind='train',
    )

    do_train(
        start_epoch = 51,
        args = args,
        trainer = trainer_stage2,
        cluster = cluster,
        stage= 2
    )

    logging.info(f'stage2 running time: {time.monotonic() - start_time}')
    writer.close()

if __name__ == '__main__':
    kind = 'sysu'

    config = creat_config(kind, 'stage2')
    _writer = do_pre(config, 'train')
    main_worker(config, _writer)