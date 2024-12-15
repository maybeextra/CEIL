
import glob
import logging
import os
import time
from pathlib import Path

import torch


def setup_logger(final_output_dir, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(phase, time_str)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s:' + ' %(message)s'
    logging.basicConfig(
        filename=str(final_log_file), format=head
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(head)
    )
    logging.getLogger('').addHandler(console)

def create_logger(final_output_dir, phase='train'):
    final_output_dir = Path(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(final_output_dir, phase)

def show_dateset_info(
        real_rgb_class, real_ir_class, real_len_rgb, real_len_ir,
        new_rgb_class, new_ir_class, new_len_rgb, new_len_ir
):
    logging.info(f'''
        ============== Dataset Statistics =============
        -----------------------------------------------
               subset  | # ids | # images
          real visible | {real_rgb_class:5d} | {real_len_rgb:8d}
          real thermal | {real_ir_class:5d} | {real_len_ir:8d}
          
          now  visible | {new_rgb_class:5d} | {new_len_rgb:8d}
          now  thermal | {new_ir_class:5d} | {new_len_ir:8d}
        -----------------------------------------------
        ===============================================
            ''')

def show_train_prepare_info(args):
    # 将args对象转换为字典
    args_dict = vars(args)

    # 计算最长的属性名长度
    max_attr_len = max(len(attr) for attr in args_dict.keys())

    # 遍历字典，输出居中对齐的属性名和属性值
    string_config = '\n======================== Training details ========================\n------------------------------------------------------------------\n'
    for attr, value in args_dict.items():
        string_config += f'\t\t{attr:{max_attr_len}}: {value}\n'
    string_config += '------------------------------------------------------------------\n=============================== END ==============================='
    logging.info(string_config)

def save_model(args, trainer, epoch, kind):
    state = {
        "state_dict": trainer.model.state_dict(),
        "epoch": epoch,
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        'scaler': trainer.scaler.state_dict()
    }

    path = args.base_dir + args.log_name + args.model_dir
    if not os.path.exists(path):
        os.makedirs(path)

    if kind == 0:
        stage_prefix = "best"
        concat = f"{stage_prefix}_epoch"
        # 在保存之前，删除具有相同模式的旧文件
        pattern = os.path.join(path, f"{concat}_*.pth")
        for file in glob.glob(pattern):
            os.remove(file)
    else:
        steps = args.steps
        stage_prefix = "train"
        warmup_iters, change_1, change_2 = args.warmup_iters, steps[0], steps[1]

        if epoch <= warmup_iters:
            file_suffix = f"0"
        elif warmup_iters < epoch <= change_1:
            file_suffix = f"1"
        elif change_1 < epoch <= change_2:
            file_suffix = f"2"
        else:
            file_suffix = f"3"
        concat = f"{stage_prefix}_{file_suffix}_epoch"

    # 构造文件名并保存模型
    file_name = f"{concat}_{epoch}.pth"
    save_path = os.path.join(path, file_name)
    torch.save(state, save_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

