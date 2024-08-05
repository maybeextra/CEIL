import torch
import torch.nn.functional as F

def get_proxy_associate_loss(cam2proxy, per_label_proxies, proxy, sim, bg_knn):
    sel_target = torch.zeros((len(sim)), dtype=sim.dtype).to(torch.device('cuda'))
    sel_target[proxy] = 1.0

    loss = -1.0 * (F.log_softmax(sim.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
    return loss