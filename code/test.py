import logging

from code.train.trainer.tester import Tester
from code.tools.utils.pre_utils import load_model, do_pre, creat_config

def main_worker(args):
    model, _, _, _, _ = load_model(args, f'{args.base_dir}{args.resume_path}', True)

    logging.info("==> Start test")
    tester = Tester(
        model=model,
        args=args,
    )

    tester.test(args, args.test_mode_1, args.mode_1)
    tester.test(args, args.test_mode_2, args.mode_2)
if __name__ == '__main__':
    kind = 'sysu'


    config = creat_config(kind, 'test')
    do_pre(config, 'test')
    main_worker(config)