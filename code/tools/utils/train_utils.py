from code import data_process

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from code.tools import trans
from code.tools.rerank.kr import compute_jaccard_distance


def mask_outlier(labels):
    count = torch.zeros(int(labels.max() + 1)).int()
    for label in labels:
        if label!= -1:
            count[label.item()] += 1
    valid_labels = torch.nonzero(count > 1).squeeze()

    # 创建掩码，指示哪些标签是有效的
    # 如果标签 i 在 valid_labels 中并且不是 -1，则对应位置的掩码值为 True，否则为 False。
    mask = torch.tensor([
        (i.item() in valid_labels and i.item() != -1) for i in labels
    ])

    return mask

def rename(labels):
    ids_container = list(torch.unique(labels))
    id2label = {id_.item(): label for label, id_ in enumerate(ids_container)}
    for i, label in enumerate(labels):
        labels[i] = id2label[label.item()]
    return labels

def map_crosses(proxies, mapping):
    return torch.tensor([mapping[key.item()] for key in proxies])


@torch.no_grad()
def generate_center_features(features, labels):
    unique_labels = torch.unique(labels)
    centers = torch.zeros(len(unique_labels), features.size(1)).cuda()

    for label in unique_labels:
        ind = torch.nonzero(labels == label).squeeze(-1)
        features_for_label = features[ind]
        centers[label] = features_for_label.mean(0)

    centers = F.normalize(centers, dim=1).cuda()
    return centers

def get_cluster_loader(dataset, batch_size, workers):
    # 使用torch.utils.data.DataLoader创建了一个数据加载器对象cluster_loader，并将提供的数据集、批次大小、工作线程数量等参数传递给它。
    # shuffle=False表示不对数据进行洗牌，pin_memory=True表示将加载的数据存储在固定内存中，以提高数据传输效率。
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return cluster_loader

def extract_features(data_type, args, trainer, cluster):
    transform_extract, transform_extract_f = trans.creat_extract(args.dataset)
    unlabel_dataset = data_process.create(
        name ='pre',
        data_dir=args.data_dir,
        extend_dir=args.extend_dir,
        data_type=data_type,
        transform_1=transform_extract,
        transform_2=transform_extract_f,
    )

    n_class, n_instance = get_class(unlabel_dataset.train_label), len(unlabel_dataset.train_image)

    cluster_loader = get_cluster_loader(unlabel_dataset, args.extract_batch_size, args.extract_num_workers)
    features, labels, cameras = trainer.extract_features_train(cluster_loader, modal=data_type, boost=args.train_boost)

    features= features.cpu()
    cameras = torch.from_numpy(cameras)
    labels = torch.from_numpy(labels)
    images = unlabel_dataset.train_image
    del cluster_loader, unlabel_dataset

    rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
    pseudo_labels = cluster.fit_predict(rerank_dist)
    del rerank_dist

    num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    return features, torch.from_numpy(pseudo_labels), num_cluster, cameras, n_class, n_instance, images, labels


def get_proxy(pseudo_labels, cams):
    proxy_labels = -1 * torch.ones(len(pseudo_labels), dtype=torch.int).cuda()
    count = 0
    # 通过循环遍历伪标签的每个类别，找到每个类别对应的样本索引。
    for i in range(0, int(pseudo_labels.max() + 1)):
        inds = torch.nonzero(pseudo_labels == i).squeeze().to(cams.device)
        # 获取各个类别的样本的cams
        temp_cams = cams[inds]
        # 对于每个类别，再根据样本的摄像头标识符（cams_rgb）进行进一步的划分。
        for cam in torch.unique(temp_cams):
            # 通过计数器cnt来维护代理标签的数量。
            proxy_labels[inds[temp_cams == cam]] = count
            count += 1
    # 通过计算代理标签的class减去-1的数量（如果-1存在），得到代理标签的数量num_proxies。
    num_proxies = len(torch.unique(proxy_labels)) - (1 if -1 in proxy_labels else 0)
    return proxy_labels.cpu(), num_proxies

def creat_dic(pseudo_labels, proxy_labels, unique_cameras, cams):
    num_labels = int(pseudo_labels.max()) + 1
    num_cameras = int(unique_cameras.max()) + 1
    num_proxies = int(proxy_labels.max()) + 1

    cam2instance = [[] for _ in range(num_cameras)]
    cam2proxy = -torch.ones((num_cameras, num_proxies), dtype=torch.int)
    for cam in unique_cameras:
        index = torch.where(cams == cam)[0]
        cam2instance[cam] = index
        index_2 = proxy_labels[index].unique().cuda()
        cam2proxy[cam, index_2] = 1

    proxy2label = -torch.ones(num_proxies, dtype=torch.int)
    proxy2cam = -torch.ones(num_proxies, dtype=torch.int)
    proxy2instance = [[] for _ in range(num_proxies)]
    for proxy in range(0, num_proxies):
        index = torch.where(proxy_labels == proxy)[0]
        labels = pseudo_labels[index]
        cameras = cams[index]
        # 去重并添加
        proxy2instance[proxy] = index

        proxy2label[proxy] = labels.unique()
        proxy2cam[proxy] = cameras.unique()

    label2instance = [[] for _ in range(num_labels)]
    label2proxy = -torch.ones((num_labels, num_cameras),dtype=torch.int)
    for label in range(0, num_labels):
        index = torch.where(pseudo_labels == label)[0]
        label2instance[label] = index
        proxy_label = proxy_labels[index].unique()
        for proxy in proxy_label:
            cam = proxy2cam[proxy]
            label2proxy[label][cam] = proxy

    instance2proxy = proxy_labels
    instance2cam = cams
    instance2label = pseudo_labels
    return label2proxy.cuda(), cam2proxy.cuda(), proxy2label, proxy2cam, label2instance, proxy2instance, cam2instance, instance2label, instance2proxy, instance2cam

def get_class(label):
    if -1 in label:
        # 如果 -1 存在于 label 中，
        # 那么 n_class 的值将是 np.unique(label) 中不同类别的数量减去 1。
        n_class = len(np.unique(label)) - 1
    else:
        # 否则，n_class 的值将是 np.unique(label) 中不同类别的数量。
        n_class = len(np.unique(label))
    return n_class

