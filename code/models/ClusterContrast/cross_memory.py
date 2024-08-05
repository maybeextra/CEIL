import torch
import torch.nn.functional as F
from torch import nn
from code.tools.cross_modal_match.match import cross_modal_matching
from .record_memory import RecordMemory

class CrossMemory(nn.Module):
    def __init__(self, gnn_k1_cross, gnn_k2_cross, temp, smooth_weight, loss_type, bg_knn, use_id_loss, use_weight, stage):
        super(CrossMemory, self).__init__()
        self.smooth_weight, self.temp = smooth_weight, temp
        self.k1, self.k2 = gnn_k1_cross, gnn_k2_cross

        self.r2i_proxy, self.i2r_proxy = {}, {}
        self.r2i_label, self.i2r_label = {}, {}
        self.use_id_loss = use_id_loss
        self.stage = stage

        self.rgb_memory = RecordMemory(self.smooth_weight, self.temp, loss_type, bg_knn, use_id_loss, use_weight, stage)
        self.ir_memory = RecordMemory(self.smooth_weight, self.temp, loss_type, bg_knn, use_id_loss, use_weight, stage)
    def refresh(self):
        self.r2i_proxy, self.i2r_proxy = {}, {}
        self.r2i_label, self.i2r_label = {}, {}
        self.rgb_memory.refresh()
        self.ir_memory.refresh()

    def update(self, rgb_info, ir_info):
        self.rgb_memory.update(rgb_info)
        self.ir_memory.update(ir_info)

    def creat_cross(self):
        self.r2i_proxy, self.i2r_proxy = cross_modal_matching(self.k1, self.k2, self.rgb_memory.proxy_centers, self.ir_memory.proxy_centers)

        if self.use_id_loss and self.stage == 2:
            self.r2i_label, self.i2r_label = cross_modal_matching(self.k1, self.k2, self.rgb_memory.class_centers, self.ir_memory.class_centers)

    def forward(self,
                inputs_rgb, proxy_rgb, label_rgb, cross_proxy_rgb, cross_label_rgb, cross_class_rgb,
                inputs_ir, proxy_ir, label_ir, cross_proxy_ir, cross_label_ir, cross_class_ir
                ):

        inputs_rgb, inputs_ir = map(lambda x: F.normalize(x, dim=1), (inputs_rgb, inputs_ir))
        loss_proxy_rgb = self.rgb_memory.compute_loss(inputs_rgb, proxy_rgb, label_rgb, label_rgb)
        loss_proxy_ir = self.ir_memory.compute_loss(inputs_ir, proxy_ir, label_ir, label_ir)

        loss_cross_rgb = self.ir_memory.compute_loss(inputs_rgb, cross_proxy_rgb, cross_label_rgb, cross_class_rgb)
        loss_cross_ir = self.rgb_memory.compute_loss(inputs_ir, cross_proxy_ir, cross_label_ir, cross_class_ir)

        return loss_proxy_rgb, loss_proxy_ir, loss_cross_rgb, loss_cross_ir

def creat_Memory(args):
    model = CrossMemory(
        gnn_k1_cross=args.gnn_k1_cross,
        gnn_k2_cross=args.gnn_k2_cross,
        temp = args.temp,
        loss_type = args.loss_type,
        bg_knn = args.bg_knn,
        use_id_loss = args.use_id_loss,
        smooth_weight=args.smooth_weight,
        use_weight=args.use_weight,
        stage=args.stage
    )
    return model

