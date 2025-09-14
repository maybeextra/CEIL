import torch
import torch.nn.functional as F

def get_proxy_associate_loss(cam2proxy, s_proxies, scores_proxy, bg_knn):
    # cam2proxy: (num_cams, num_proxy)，其中1表示包含，-1表示不包含
    # s_proxies: (batch_size, num_cam)，其中-1表示在该相机下不存在，其他值表示代理索引
    # scores_proxy: (batch_size, num_proxy)，每个样本对所有代理的得分

    # 将cam2proxy转换为布尔掩码
    cam2proxy_mask = (cam2proxy != -1)  # 形状: (num_cams, num_proxy)

    # 获取批次大小和代理数量
    batch_size, num_proxy = scores_proxy.shape
    # 获取摄像机数量
    num_cam = s_proxies.shape[1]

    # 准备per_label_proxies，将无效条目设置为num_proxy（越界索引）
    per_label_proxies = s_proxies.clone()
    per_label_proxies[per_label_proxies == -1] = num_proxy

    # 创建temp_sim，通过将自身代理设置为高值（10000.0）
    temp_sim = scores_proxy.clone()  # 形状: (batch_size, num_proxy)
    valid_mask = (per_label_proxies != num_proxy)  # 形状: (batch_size, num_cam)

    # 创建批次索引的方式（用于后续索引操作）
    batch_indices = torch.arange(batch_size, device='cuda:0').unsqueeze(1).expand(-1, num_cam)  # 形状: (batch_size, num_cam)
    valid_batch_indices = batch_indices[valid_mask]  # 有效的批次索引的1D张量
    valid_proxies = per_label_proxies[valid_mask]    # 有效的代理索引的1D张量

    # 在temp_sim中给自身代理赋高值
    temp_sim[valid_batch_indices, valid_proxies] = 10000.0

    # 初始化用于存储每个摄像机损失
    loss = torch.zeros((batch_size, num_cam), device='cuda:0')

    # 遍历每个摄像机
    for cam in range(num_cam):
        # 找到在该摄像机下有有效数据的样本索引
        samples_with_cam = torch.where(valid_mask[:, cam])[0]  # 在此摄像机下存在有效数据的样本索引

        if samples_with_cam.numel() == 0:
            continue  # 如果没有样本，继续下一个摄像机

        # 获取与此摄像机关联的代理
        per_cam_proxies = torch.where(cam2proxy_mask[cam])[0]  # 此摄像机的代理索引

        # 提取每个摄像机的相似度
        per_cam_sim = scores_proxy[samples_with_cam][:, per_cam_proxies]      # 形状: (samples_with_cam, num_proxies)
        per_cam_temp_sim = temp_sim[samples_with_cam][:, per_cam_proxies]     # 与上面相同的形状

        # 对per_cam_temp_sim进行排序并选择 (bg_knn + 1) 个最高索引
        # bg_knn + 1 > per_cam_proxies 时全选, 反之选择 bg_knn + 1
        sorted_indices = torch.argsort(per_cam_temp_sim, dim=1)
        select_indices = sorted_indices[:, -bg_knn - 1:]

        # 收集选中的相似度
        select_sim = torch.gather(per_cam_sim, dim=1, index=select_indices)   # 形状: (samples_with_cam, bg_knn + 1)

        # 创建目标张量，其中最后一个条目为1，其余为0
        select_target = torch.zeros_like(select_sim)
        select_target[:, -1] = 1.0

        # 计算当前摄像机的损失
        loss[samples_with_cam,cam] = -1.0 * (F.log_softmax(select_sim, dim=1) * select_target).sum(dim=1) # 形状: (samples_with_cam,)

    # 计算每个样本的均值
    # 对 num_cam 维度进行求和，然后除以 valid_mask 为 True 的数量
    per_sample_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).squeeze()  # 防止除以零
    per_sample_mean = loss.sum(dim=1) / per_sample_counts

    # 计算所有样本的均值
    final_loss_avg = per_sample_mean.mean()
    return final_loss_avg