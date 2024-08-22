from torch import nn

class RecordMemory(nn.Module):
    def __init__(self, smooth_weight, temp, loss_type, bg_knn, use_id_loss, use_weight, stage):
        super(RecordMemory, self).__init__()

        self.unique_cameras = None
        self.label2proxy,self.cam2proxy,self.proxy2label,self.proxy2cam = None, None, None, None
        self.label2instance, self.cam2instance, self.proxy2instance, self.instance2proxy, self.instance2cam, self.instance2label = None, None, None, None, None, None
        self.proxy_centers = None
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