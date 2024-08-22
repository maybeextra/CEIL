import torchvision.transforms as transforms
from code.tools.trans.ChannelAug import ChannelAdapGray, ChannelRandomErasing, ChannelExchange

img_h_LLCM = 288
img_w_LLCM = 144

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_extract_LLCM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h_LLCM, img_w_LLCM)),
    transforms.ToTensor(),
    normalize,
])

train_transformer_LLCM_rgb_weak = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((img_h_LLCM, img_w_LLCM)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normalize,
    ChannelRandomErasing(probability=0.5)
])
train_transformer_LLCM_rgb_strong = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((img_h_LLCM, img_w_LLCM)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    normalize,
    ChannelRandomErasing(probability=0.5),
    ChannelExchange(gray=2),
])
train_transformer_LLCM_ir = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((img_h_LLCM, img_w_LLCM)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ChannelRandomErasing(probability=0.5),
    ChannelAdapGray(probability=0.5)
])

