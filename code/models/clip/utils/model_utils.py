from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import logging

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim) 
        self.num_heads = num_heads

    def forward(self, x):
        # print(x.shape) # [256, 2048, 18, 9]
        x = x.flatten(start_dim=2).permute(2, 0, 1) # NCHW -> (HW)NC  #32,2048,7,7 ->49, 32, 2048
        # print(x.shape) # torch.Size([162, 256, 2048])
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC  50,32,2048
        # print(x.shape) # torch.Size([163, 256, 2048])
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # print(x.shape) # torch.Size([163, 256, 2048])

        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        ) 

        return x.squeeze(0)

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        # 这段代码定义了一个名为ModifiedResNet的类，继承自nn.Module。
        # 在__init__方法中，定义了类的初始化函数，并接收一系列参数，
        # 包括layers（残差块的数量和层数）、output_dim（输出维度）、heads（注意力头数）、input_resolution（输入图像的分辨率，默认为224）和width（通道宽度，默认为64）。
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        # conv1：一个输入通道为3、输出通道为width // 2的卷积层，卷积核大小为3x3，步长为2，填充为1。
        # bn1：对conv1的输出进行批归一化。
        # conv2：一个输入通道为width // 2、输出通道为width // 2的卷积层，卷积核大小为3x3，填充为1。
        # bn2：对conv2的输出进行批归一化。
        # conv3：一个输入通道为width // 2、输出通道为width的卷积层，卷积核大小为3x3，填充为1。
        # bn3：对conv3的输出进行批归一化。
        # avgpool：一个2x2的平均池化层。
        # relu：ReLU激活函数。
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        # 这里定义了一系列的残差块层和Attention Pooling层，用于构建ResNet的主体部分。这些层包括：
        # layer1：通过调用_make_layer方法创建的第一个残差块层，输入通道数为width，输出通道数为width，步长为1，块的数量由layers[0]定义。
        # layer2：通过调用_make_layer方法创建的第二个残差块层，输入通道数为width * 2，输出通道数为width * 2，步长为2，块的数量由layers[1]定义。
        # layer3：通过调用_make_layer方法创建的第三个残差块层，输入通道数为width * 4，输出通道数为width * 4，步长为2，块的数量由layers[2]定义。
        # layer4：通过调用_make_layer方法创建的第四个残差块层，输入通道数为width * 8，输出通道数为width * 8，步长为1，块的数量由layers[3]定义。
        # attnpool：一个Attention Pooling层，输入分辨率为input_resolution，嵌入维度为embed_dim，注意力头数为heads，输出维度为output_dim。
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1) 
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution, embed_dim, heads, output_dim)

    # 这是一个辅助方法，用于创建残差块层。
    # 根据输入的通道数planes、块的数量blocks和步长stride，创建一个包含多个Bottleneck模块的序列。
    # 首先，创建一个带有指定输入通道数、输出通道数和步长的Bottleneck模块，并添加到layers列表中。
    # 然后，根据块的数量，使用循环创建其他的Bottleneck模块，并添加到layers列表中。
    # 最后，使用nn.Sequential将layers列表中的模块组合成一个序列。
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): 
        def stem(x):
            # 定义了一个名为stem的函数，用于对输入的图像进行前处理。
            # 在stem函数中，通过循环遍历卷积层和批归一化层，将输入图像依次经过卷积、批归一化和ReLU激活函数的处理，并使用平均池化层对特征图进行下采样。
            # 然后，将输入的图像经过前处理的结果x传入残差块层进行特征提取。
            # 最后，将最后一个残差块层的特征图x输入到Attention Pooling层进行特征池化，得到池化后的特征向量x_pool。
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype) 
        x = stem(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        x_pool = self.attnpool(x)

        return x, x_pool


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution*w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, cv_emb = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if cv_emb != None: 
            x[:,0] = x[:,0] + cv_emb
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        x11 = self.transformer.resblocks[:11](x) 
        x12 = self.transformer.resblocks[11](x11) 
        x11 = x11.permute(1, 0, 2)  # LND -> NLD  
        x12 = x12.permute(1, 0, 2)  # LND -> NLD  

        x12 = self.ln_post(x12)  

        if self.proj is not None:
            xproj = x12 @ self.proj   

        return x11, x12, xproj


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 h_resolution: int, 
                 w_resolution: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            # True
            # 如果vision_layers是一个元组或列表，则将vision_heads设置为vision_width乘以32除以64的结果，并使用这些参数来实例化ModifiedResNet。
            # ModifiedResNet的参数包括layers（网络层数）、output_dim（输出维度）、heads（注意力头数）、input_resolution（输入分辨率）和width（通道宽度）。
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=h_resolution*w_resolution,
                width=vision_width
            )
        else:
            # 如果vision_layers不是元组或列表，则将vision_heads设置为vision_width除以64的结果，并使用这些参数来实例化VisionTransformer。
            # VisionTransformer的参数包括h_resolution（高分辨率）、w_resolution（宽分辨率）、patch_size（图像块大小）、stride_size（滑动窗口的步幅）、
            # width（通道宽度）、layers（网络层数）、heads（注意力头数）和output_dim（输出维度）。
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                h_resolution = h_resolution,
                w_resolution = w_resolution,
                patch_size = vision_patch_size,
                stride_size = vision_stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # 实例化一个Transformer类，用于文本编码。
        # 传入了transformer_width、transformer_layers、transformer_heads和attn_mask参数。
        # attn_mask参数调用了self.build_attention_mask()方法来生成一个注意力掩码。
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        # 初始化了一些模型参数。
        # token_embedding是一个词嵌入层，将文本中的词索引转换为词嵌入向量。
        # positional_embedding是一个位置嵌入向量，用于表示文本中每个词的位置信息。
        # ln_final是一个层归一化层，用于对文本编码后的特征进行归一化处理。
        # text_projection是一个可学习的线性投影矩阵，用于将文本特征投影到指定的维度上。
        # logit_scale是一个可学习的参数，用于缩放模型的输出。
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    # 这是一个初始化参数的函数。
    # 使用正态分布初始化了token_embedding和positional_embedding的权重。
    # 如果self.visual是ModifiedResNet的实例，将使用正态分布初始化ModifiedResNet中的一些参数。
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            # True
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # 使用正态分布初始化了Transformer中的一些参数。
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # 如果text_projection不为空，使用正态分布初始化text_projection的权重。
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # 这是一个构建注意力掩码的函数。
    # 创建了一个上三角矩阵，并将其填充为负无穷。
    # 这个上三角矩阵用于实现Transformer模型的自注意力机制中的序列顺序限制。
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    # 这是一个属性方法，返回self.visual.conv1.weight的数据类型。
    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    # 这是一个用于图像编码的方法。
    # 将输入的图像数据转换为与模型的权重数据类型相同的类型，并通过self.visual对图像进行编码。
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    # 这是一个用于文本编码的方法。
    # 首先，将输入的文本索引text通过self.token_embedding转换为词嵌入向量x。
    # 然后，将位置嵌入向量加到词嵌入向量上。
    # 接下来，通过对x进行维度变换，将序列长度放在第1维。
    # 然后，将x输入到Transformer模型中进行编码。
    # 编码后的结果再次进行维度变换，将序列长度放回第0维。
    # 然后，通过层归一化层对编码后的结果进行归一化处理。
    # 最后，通过对x的每个样本取最大值的索引，选择出每个样本的最重要的词嵌入向量，并通过self.text_projection进行投影，得到文本的编码。
    def encode_text(self, text): 
        x = self.token_embedding(text).type(self.dtype)  

        x = x + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype) 

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection 

        return x

    # 这是模型的前向传播方法。
    # 首先，分别对输入的图像和文本进行编码，得到图像特征image_features和文本特征text_features。
    # 然后，对图像特征和文本特征进行归一化处理。
    # 接下来，通过指数函数对self.logit_scale进行求指数，得到缩放因子logit_scale。
    # 然后，通过矩阵乘法计算图像特征和文本特征之间的相似度，得到logits_per_image和logits_per_text。
    # 最后，将它们作为模型的输出返回。
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, h_resolution: int, w_resolution: int, vision_stride_size: int):
    # 检查给定的状态字典中是否包含"visual.proj"这个键，以判断模型类型是Vision Transformer（ViT）还是ResNet50（RN50）。
    vit = "visual.proj" in state_dict # False
    if vit:
        # 如果是ViT模型：
        # 提取出ViT模型的相关参数，如视觉层的数量vision_layers，视觉宽度vision_width，视觉块的大小vision_patch_size，以及图像分辨率image_resolution。
        # 通过计算，得到视觉块的网格大小grid_size，并计算出图像分辨率image_resolution。
        # 其他参数，如嵌入维度embed_dim，文本上下文长度context_length，词汇表大小vocab_size，
        # Transformer的宽度transformer_width，Transformer的注意力头数transformer_heads，Transformer的层数transformer_layers，
        # 图像的高分辨率h_resolution和宽分辨率w_resolution，都从状态字典中获取。
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else: #RN50
        # 如果是RN50模型：
        # 提取出RN50模型的相关参数，如视觉层的数量vision_layers（通过计算每个阶段的层数），视觉宽度vision_width，图像分辨率image_resolution。
        # 检查参数的一致性，确保output_width的平方加1等于"visual.attnpool.positional_embedding"的形状。
        # 其他参数和ViT模型一样，从状态字典中获取。
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0] #77 (77,512)
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size, vision_stride_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        h_resolution, w_resolution
    )

    # 根据模型类型（根据vit变量的值判断是ViT还是RN50），调用resize_pos_embed函数来调整位置嵌入向量的大小，以适应给定的高分辨率和宽分辨率。
    # 如果是ViT模型，则调整visual.positional_embedding；
    # 如果是RN50模型，则调整visual.attnpool.positional_embedding。
    if vit:
        state_dict["visual.positional_embedding"] = resize_pos_embed(state_dict["visual.positional_embedding"], model.visual.positional_embedding, h_resolution, w_resolution)
    else: #RN50
        state_dict["visual.attnpool.positional_embedding"] = resize_pos_embed(state_dict["visual.attnpool.positional_embedding"], model.visual.attnpool.positional_embedding, h_resolution, w_resolution)
    
    # 接下来，删除状态字典中的一些不需要的键，包括input_resolution、context_length和vocab_size。
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # 调用convert_weights函数，将模型的权重数据类型转换为与模型实例的权重数据类型一致。
    convert_weights(model)

    model.load_state_dict(state_dict)
    return model.eval()

import math
def resize_pos_embed(posemb, posemb_new, hight, width):
    # 这段代码定义了一个名为resize_pos_embed的函数，用于调整位置嵌入向量的大小。
    # 该函数接受四个参数：posemb（要调整大小的原始位置嵌入向量）、posemb_new（调整后的位置嵌入向量）、hight（高度）和width（宽度）。

    # 首先，获取调整后的位置嵌入向量的长度ntok_new，通过posemb_new.shape[0]获得。
    # 然后，从原始位置嵌入向量中分离出第一个位置嵌入向量posemb_token和其余的位置嵌入向量posemb_grid。通过posemb[:1]获得第一个位置嵌入向量，posemb[1:]获得其余位置嵌入向量。
    # 接下来，计算原始位置嵌入向量的网格大小gs_old，通过将posemb_grid的长度开方得到。
    # 然后，将posemb_grid重新调整为形状为[1, gs_old, gs_old, -1]的张量，并通过转置和重塑操作，将形状变为[1, hight * width, -1]，以适应新的高度和宽度。
    # 最后，将第一个位置嵌入向量和调整后的位置嵌入向量连接在一起，返回调整后的位置嵌入向量。

    # 该函数的作用是将原始位置嵌入向量调整为新的高度和宽度，以适应模型的输入尺寸。
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    logging.info(f'Resized position embedding: {posemb.shape[0]}x{posemb.shape[1]} to {posemb_new.shape[0]}x{posemb_new.shape[1]}') # 50x2048 to 163x2048
      
    ntok_new = posemb_new.shape[0] #129,2048

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid))) #14
    logging.info(f'Position embedding resize to height:{hight} width: {width}') # 18x9
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2) 
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear') 
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)
    return posemb