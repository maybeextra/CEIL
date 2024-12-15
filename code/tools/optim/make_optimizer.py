import torch

def make_optimizer(baselr, weight_decay, bias_lr_factor, weight_decay_bias, model_net):
    # 创建了一个空列表 params，用于存储需要优化的参数。
    params = []

    # 遍历 model_net 的命名参数，检查参数名称中是否包含特定的子字符串，
    # 如 "text_encoder" 和 "prompt_learner"。如果参数名称中包含这些子字符串，
    # 将设置参数的 requires_grad 属性为 False，表示不需要对这些参数进行梯度计算和优化。
    for key, value in model_net.named_parameters():
        if not value.requires_grad:
            # 检查参数的 requires_grad 属性，如果为 False，表示不需要对该参数进行梯度计算和优化，可以跳过该参数的处理
            continue

        # 对于需要优化的参数，根据参数名称中是否包含 "bias" 子字符串来设置不同的学习率和权重衰减（weight decay）值。
        lr = baselr
        weight_decay = weight_decay
        if "bias" in key:
            lr = baselr * bias_lr_factor
            weight_decay = weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer


