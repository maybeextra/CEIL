import torch
import torch.nn.functional as F
from code.tools.output.log_utils import AverageMeter

def get_cam_associate_loss(cam2proxy, per_label_proxies, proxy, sim, bg_knn):
    loss_inter = AverageMeter()

    have_cam = torch.where(per_label_proxies != -1)[0]
    self_proxies = per_label_proxies[have_cam]

    temp_sim = sim.clone()
    temp_sim[self_proxies] = 10000.0

    for cam in have_cam:
        per_cam_proxies = cam2proxy[cam]

        per_cam_sim = sim[per_cam_proxies]
        per_cam_temp_sim = temp_sim[per_cam_proxies]

        select_index = torch.sort(per_cam_temp_sim)[1][-bg_knn - 1:]
        select_sim = per_cam_sim[select_index]

        select_target = torch.zeros_like(select_sim)
        select_target[-1] = 1.0

        loss = -1.0 * (F.log_softmax(select_sim.unsqueeze(0), dim=1) * select_target.unsqueeze(0)).sum()
        loss_inter.update(loss)

    return loss_inter.avg