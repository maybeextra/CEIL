import torch
import torch.nn as nn

from .utils import clip


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power # 设置规范化的范数参数

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# 加载预训练的 OpenAI CLIP 模型并将其部署到 CPU 上。
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    # backbone_name：CLIP 模型的骨干网络名称。
    # h_resolution：输入图像的高度（分辨率）。
    # w_resolution：输入图像的宽度（分辨率）。
    # vision_stride_size：视觉特征提取的步幅大小。

    # 根据 backbone_name 从 CLIP 的模型字典 _MODELS 中获取模型的下载 URL，并使用 _download 函数下载模型。
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    # 尝试加载 JIT 归档。
    try:
        # 成功加载 JIT 归档，则将模型设置为评估模式（eval()）并将 state_dict 设置为 None。
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        # 无法加载 JIT 归档（可能由于 JIT 编译失败），则使用 torch.load 函数从模型路径中加载状态字典（state_dict）。
        state_dict = torch.load(model_path, map_location="cpu")

    # 使用加载的 state_dict 或模型的 state_dict() 构建 CLIP 模型。
    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class visible_module(nn.Module):
    def __init__(self, model_name, h_resolution, w_resolution, vision_stride_size):
        super(visible_module, self).__init__()
        model_v = load_clip_to_cpu(model_name, h_resolution,w_resolution,vision_stride_size)
        self.conv1 = model_v.visual.conv1
        self.bn1 = model_v.visual.bn1
        self.conv2 = model_v.visual.conv2
        self.bn2 = model_v.visual.bn2
        self.conv3 = model_v.visual.conv3
        self.bn3 = model_v.visual.bn3
        self.relu = model_v.visual.relu
        self.pool = model_v.visual.avgpool
        del model_v

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.pool(x)
        return x

class thermal_module(nn.Module):
    def __init__(self, model_name, h_resolution, w_resolution, vision_stride_size):
        super(thermal_module, self).__init__()
        model_t = load_clip_to_cpu(model_name, h_resolution,w_resolution,vision_stride_size)
        self.conv1 = model_t.visual.conv1
        self.bn1 = model_t.visual.bn1
        self.conv2 = model_t.visual.conv2
        self.bn2 = model_t.visual.bn2
        self.conv3 = model_t.visual.conv3
        self.bn3 = model_t.visual.bn3
        self.relu = model_t.visual.relu
        self.pool = model_t.visual.avgpool
        del model_t

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.pool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, model_name, h_resolution, w_resolution, vision_stride_size):
        super(base_resnet, self).__init__()
        model_base = load_clip_to_cpu(model_name, h_resolution, w_resolution, vision_stride_size)
        self.layer1 = model_base.visual.layer1
        self.layer2 = model_base.visual.layer2
        self.layer3 = model_base.visual.layer3
        self.layer4 = model_base.visual.layer4

        del model_base

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class GEM(nn.Module):
    def __init__(self, p=3, eps=1e-12):
        super(GEM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = (torch.mean(x ** self.p, dim=-1) + self.eps) ** (1.0 / self.p)

        return x


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.model_name = args.arch # * res50

        self.h_resolution = int((args.img_h-16) // args.stride_size[0] + 1)
        self.w_resolution = int((args.img_w-16) // args.stride_size[1] + 1)
        self.vision_stride_size = args.stride_size[0]

        self.visible_module = visible_module(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        self.thermal_module = thermal_module(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        self.base_resnet = base_resnet(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)

        self.num_features = 2048
        self.pool = GEM()

        self.BatchNorm = nn.BatchNorm1d(self.num_features)
        self.BatchNorm.bias.requires_grad_(False)
        self.l2norm = Normalize(2)

    def forward(self, x1=None, x2=None, modal=0):
        # 图像特征提取
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        else:
            raise

        x = self.base_resnet(x) # [batch_size, 2048, 18, 9]

        x_pool = self.pool(x)
        feats = self.BatchNorm(x_pool)

        if not self.training:
            return self.l2norm(feats)
        else:
            return feats

def make_clip(args):
    model = Net(args)
    return model




























