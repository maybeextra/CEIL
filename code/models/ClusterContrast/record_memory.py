import torch
import torch.nn.functional as F
from code.tools import loss
from code.tools.output.log_utils import AverageMeter
from code.tools.utils.train_utils import generate_center_features, creat_dic
from torch import nn

from .cm import CM,HM

class RecordMemory(nn.Module):
    def __init__(self, smooth_weight, temp, loss_type, bg_knn, use_id_loss, use_weight, stage):
        super(RecordMemory, self).__init__()

        self.unique_cameras = None
        self.label2proxy,self.cam2proxy,self.proxy2label,self.proxy2cam = None, None, None, None
        self.label2instance, self.cam2instance, self.proxy2instance, self.instance2proxy, self.instance2cam, self.instance2label = None, None, None, None, None, None
        self.proxy_centers = None

        self.get_proxy_associate_loss = loss.create(loss_type)
        self.temp = temp
        self.smooth_weight = smooth_weight
        self.use_weight = use_weight
        self.use_id_loss = use_id_loss
        self.bg_knn = bg_knn
        self.stage = stage

        if self.stage == 2:
            self.class_centers = None
    def refresh(self):
        self.unique_cameras = None
        self.label2proxy,self.cam2proxy,self.proxy2label,self.proxy2cam = None, None, None, None
        self.label2instance, self.cam2instance, self.proxy2instance, self.instance2proxy, self.instance2cam, self.instance2label = None, None, None, None, None, None
        self.proxy_centers = None

        if self.stage == 2:
            self.class_centers = None

    def update(self, info):
        [features, pseudo_labels, proxy_labels, cams, images, n_class, n_instance, real_labels] = info

        unique_cameras = torch.unique(cams).cpu()
        self.unique_cameras= unique_cameras
        self.label2proxy, self.cam2proxy, self.proxy2label, self.proxy2cam,\
        self.label2instance, self.proxy2instance, self.cam2instance, self.instance2label, self.instance2proxy, self.instance2cam = \
        creat_dic(pseudo_labels,proxy_labels,unique_cameras,cams)

        self.class_centers = generate_center_features(features.cuda(),pseudo_labels.cuda())
        self.proxy_centers = generate_center_features(features.cuda(),proxy_labels.cuda())


    def compute_loss(self, inputs, proxies, labels, classes = None):
        loss_proxy, loss_class = 0, 0

        # ALL
        if proxies is not None and labels is not None:
            outputs_proxy = CM.apply(inputs, proxies, self.proxy_centers, self.smooth_weight, self.use_weight)
            scores_proxy = outputs_proxy / self.temp

            s_proxies = self.label2proxy[labels]
            loss_proxy = self.get_proxy_associate_loss(self.cam2proxy, s_proxies, scores_proxy, bg_knn=self.bg_knn)

        if self.use_id_loss and classes is not None:
            outputs_label = CM.apply(inputs, classes, self.class_centers, self.smooth_weight, self.use_weight)
            scores_label = outputs_label / self.temp

            loss_class = F.cross_entropy(scores_label, classes.to(torch.long))

        # CAM LOSS
        # if proxies is not None and labels is not None:
        #     outputs_proxy = HM.apply(inputs, proxies, self.proxy_centers, 0.8)
        #     scores_proxy = outputs_proxy / self.temp
        #
        #     s_proxies = self.label2proxy[labels]
        #     loss_proxy = self.get_proxy_associate_loss(self.cam2proxy, s_proxies, scores_proxy, bg_knn=self.bg_knn)
        #
        # if self.use_id_loss and classes is not None:
        #     outputs_label = HM.apply(inputs, classes, self.class_centers, 0.8)
        #     scores_label = outputs_label / self.temp
        #
        #     loss_class = F.cross_entropy(scores_label, classes.to(torch.long))

        # Memory
        # if self.use_id_loss and classes is not None:
        #     outputs_label = CM.apply(inputs, classes, self.class_centers, self.smooth_weight, self.use_weight)
        #     scores_label = outputs_label / self.temp
        #
        #     loss_class = F.cross_entropy(scores_label, classes.to(torch.long))

        # BASE
        # if self.use_id_loss and classes is not None:
        #     outputs_label = HM.apply(inputs, classes, self.class_centers, 0.8)
        #     scores_label = outputs_label / self.temp
        #
        #     loss_class = F.cross_entropy(scores_label, classes.to(torch.long))
        return loss_proxy + loss_class