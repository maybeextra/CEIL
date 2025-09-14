import torchvision.transforms as transforms
from code.tools.trans.ChannelAug import ChannelAdapGray, ChannelRandomErasing, ChannelExchange

img_h_RegDB = 288
img_w_RegDB = 144

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_extract_RegDB = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    normalize,
])

transform_extract_RegDB_f = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
    normalize,
])

train_transformer_RegDB_rgb_weak = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((img_h_RegDB, img_w_RegDB)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normalize,
    ChannelRandomErasing(probability=0.5)
])
train_transformer_RegDB_rgb_strong = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((img_h_RegDB, img_w_RegDB)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    normalize,
    ChannelRandomErasing(probability=0.5),
    ChannelExchange(gray=2),
])
train_transformer_RegDB_ir = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((img_h_RegDB, img_w_RegDB)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ChannelRandomErasing(probability=0.5),
    ChannelAdapGray(probability=0.5)
])