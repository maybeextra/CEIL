import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init


# KCM
class catcher(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = 2048 ** -0.5

        self.patch_num = 162 # 162 128

        self.pos_embed = Parameter(torch.FloatTensor(1, self.patch_num, 2048))
        init.trunc_normal_(self.pos_embed.data, mean=0.0, std=0.02)

        self.q = nn.Linear(dim, dim, bias=False)
        self.g = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q, g, cam_embed=None):
        ########################################
        rel_q = self.q(q)
        ########################################
        B, N, C = g.size()

        g = g.reshape(B * N // self.patch_num, self.patch_num, C)

        rel = F.normalize(g, dim=-1) # B x HW x C
        rel = torch.bmm(rel, rel.transpose(1, 2)) # B x HW x HW
        rel = torch.softmax(rel, dim=-1)

        g = torch.bmm(rel, g)
        rel_g = g + self.pos_embed

        if cam_embed is not None:
            cam_embed = cam_embed.reshape(B * N // self.patch_num, 1, C)
            rel_g = g + cam_embed

        g = g.reshape(B, N, C)
        rel_g = rel_g.reshape(B, N, C)
        ########################################
        rel_g = self.g(rel_g)
        ########################################
        rel = torch.bmm(rel_q, rel_g.transpose(1, 2)) # B x HW x HW
        rel = rel * self.scale

        rel = torch.softmax(rel, dim=-1)

        out = torch.bmm(rel, g)

        return out
        
# DMM
class deltaor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = 2048 ** -0.5

        self.normx = nn.LayerNorm(dim, elementwise_affine=False)
        self.normg = nn.LayerNorm(dim, elementwise_affine=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.g = nn.Linear(dim, dim, bias=False)

    def forward(self, x, g):
        q = self.normx(x)
        rel_q = self.q(q)

        g = g - x.mean(dim=1, keepdim=True)#.clone().detach()
        g = self.normg(g)
        rel_g = self.g(g)

        rel = torch.bmm(rel_q, rel_g.transpose(1, 2)) # B x HW x HW
        rel = rel * self.scale
        rel = torch.softmax(rel, dim=-1)

        out = x + torch.bmm(rel, g)

        return out


class circle_fine_grained_extractor(nn.Module):
    def __init__(self, dim, q_num):
        super().__init__()
        
        self.P2R = self.Q2R = catcher(dim) # KCM

        self.R2P = deltaor(dim) # DMM

        self.num_instance = 8
        
        self.query_v = Parameter(torch.FloatTensor(1, q_num, dim))
        self.query_i = Parameter(torch.FloatTensor(1, q_num, dim))
        init.trunc_normal_(self.query_v.data, mean=0.0, std=0.02)
        self.query_i.data = self.query_v.data

        self.prototype = Parameter(torch.FloatTensor(1, 1024, dim))
        init.trunc_normal_(self.prototype.data, mean=0.0, std=0.02)

        self.weights = Parameter(torch.ones(1, q_num, 1))

        cam_num = 6
        self.cam_embed = Parameter(torch.FloatTensor(cam_num, 1, dim))
        init.trunc_normal_(self.cam_embed.data, mean=0.0, std=0.02)

    def forward(self, x, cam_ids):
        # 获取输入 x 的形状，分别赋值给 B（批大小）、C（通道数）、H（高度）和 W（宽度）。
        B, C, H, W = x.shape

        # 将输入 x 重塑为 (B, C, H*W)
        # 转置 x，使其形状变为 (B, H*W, C)。
        x = x.reshape(B, C, -1).transpose(1, 2)

        # query = torch.cat([self.query_v.repeat(x.size(0), 1, 1)[sub], self.query_i.repeat(x.size(0), 1, 1)[~sub]], dim=0)
        # 将 self.query_v 复制并扩展到 (B, q_num, dim)，其中 B 为批大小。
        query = self.query_v.repeat(x.size(0), 1, 1)
        # 将 self.prototype 复制并扩展到 (B, 1024, dim)。
        prototype = self.prototype.repeat(x.size(0), 1, 1)  # B x N x C
        # 根据 cam_ids 选择相应的 cam_embed

        cam_embed = self.cam_embed[cam_ids]

        # 调用 self.Q2R 方法计算 f_rel
        f_rel = self.Q2R(query, x, cam_embed)

        # 调用 self.R2P 方法计算 f_pro DMM
        f_pro = self.R2P(f_rel, prototype)

        # f_rec, f_cor = None, None
        # if self.training:
        #     # 调用 self.P2R 方法计算 f_rec 和 _。
        #     f_rec = self.P2R(f_pro, x, cam_embed)
        #
        #     # 对 cam_embed 进行重塑和扩展
        #     cam_embed = cam_embed.reshape(
        #         B // self.num_instance, 1, self.num_instance, 1, C
        #     ).repeat(1, self.num_instance, 1, 1, 1).reshape(B, self.num_instance, C)
        #
        #     # 对 x 进行重塑和扩展
        #     x = x.reshape(
        #         B // self.num_instance, 1, self.num_instance, H*W, C
        #     ).repeat(1, self.num_instance, 1, 1, 1).reshape(B, self.num_instance*H*W, C)
        #     # 调用 self.P2R 方法计算 f_cor 和 _。
        #     f_cor = self.P2R(f_pro, x, cam_embed)

        # 使用 softmax 对 self.weights 进行归一化，并与 f_pro 相乘。
        f_pro = torch.softmax(self.weights, dim=1) * f_pro

        # return f_rel, f_pro, f_rec, f_cor
        return f_rel, f_pro