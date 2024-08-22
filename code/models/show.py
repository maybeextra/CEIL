import torch

from code.tools.utils.pre_utils import load_model, do_pre, creat_config

def main_worker(args):
    model, optimizer, scheduler, scaler, start_epoch = load_model(args, f'{args.base_dir}{args.resume_path}', args.resume)
    print(model)
    temp = torch.rand(5,3,384,128).cuda()
    x = model(temp,temp)



if __name__ == '__main__':
    kind = 'regdb'
    stage = 'stage1'

    config = creat_config(kind, stage)

    # 调用main函数并传递解析的命令行参数args。
    writer = do_pre(config, 'test')
    main_worker(config)
