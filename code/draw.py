import logging

from code.tools.utils.pre_utils import load_model, do_pre, creat_config
from code.train.trainer.drawer import Drawer


def main_worker(args):
    model, _, _, _, _ = load_model(args, f'{args.base_dir}{args.resume_path}', True)

    logging.info("==> Start test")
    drawer = Drawer(
        model=model,
        args=args,
    )

    drawer.draw(args, args.test_mode_1, args.mode_1)
if __name__ == '__main__':
    kind = 'sysu'

    config = creat_config(kind, 'visualize')
    do_pre(config, 'test')
    main_worker(config)