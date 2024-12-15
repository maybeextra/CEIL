#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""


import logging
import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(x.untyped_storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype

    return faiss.cast_integer_to_idx_t_ptr(x.untyped_storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr, k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I


# k-reciprocal nearest neighbor 算法
# 这个算法的思路就是：如果两张图片A，B相似，那么B应该会在A的前K个近邻里面，反过来，A也会在B的前K个近邻里面。但如果两张图C，D不相似，即使C在D的前K个近邻里面，D也不会在C的前K个近邻里面。
def k_reciprocal_neigh(initial_rank, i, k1):
    #     # 通过索引 i 从初始排名矩阵 initial_rank 中获取图像 i 前 k1+1 个近邻的索引，并将它们存储在 forward_k_neigh_index 中。
    forward_k_neigh_index = initial_rank[i,:k1+1]
    # 通过 forward_k_neigh_index 中的索引，再次从初始排名矩阵中获取这些近邻的前 k1+1 个近邻的索引，并将它们存储在 backward_k_neigh_index 中。
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    # 使用 np.where 函数找到 backward_k_neigh_index 中等于 i 的索引，并将它们存储在 fi 中。
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(query_feat, gall_feat, query_cam = None, gall_cam = None, k1=20, k2=6, lambda_value=0.3):
    q_g_dist = np.dot(query_feat.cpu(), np.transpose(gall_feat.cpu()))
    q_q_dist = np.dot(query_feat.cpu(), np.transpose(query_feat.cpu()))
    g_g_dist = np.dot(gall_feat.cpu(), np.transpose(gall_feat.cpu()))
    # # 在函数中，首先将查询-查询、查询-图库和图库-图库的距离矩阵拼接起来，形成一个综合的距离矩阵original_dist。然后，将距离矩阵转换为相似度矩阵，即将距离值转换为相似度值（这里使用欧式距离公式进行转换）。
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    original_dist = 2. - 2 * original_dist  # np.power(original_dist, 2).astype(np.float32) 余弦距离转欧式距离
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))  # 归一化

    # 创建一个与original_dist形状相同的全零矩阵V，用于存储权重值。然后，使用np.argpartition函数对original_dist进行排序，得到初始排序的索引矩阵initial_rank。
    V = np.zeros_like(original_dist).astype(np.float32)
    # initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1))  # 取前20，返回索引号

    # 获得查询集的数量query_num和所有样本的数量all_num。
    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    # 对于每个样本，计算其k-reciprocal邻居的索引，然后进行互相关扩展，将相似的样本添加到扩展索引中。接着，计算权重值，通过指数函数进行归一化，得到权重矩阵V。
    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)  # 取出互相是前20的
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):  # 遍历与第i张图片互相是前20的每张图片
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            # 比较candidate_k_reciprocal_index和k_reciprocal_index的交集大小与candidate_k_reciprocal_index的2/3的关系，判断是否将candidate_k_reciprocal_index添加到k_reciprocal_expansion_index中。
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(candidate_k_reciprocal_index):
                # 如果交集大小大于2/3 * candidate_k_reciprocal_index的大小，说明candidate_k_reciprocal_index是k-reciprocal邻居的一部分，可以将其添加到k_reciprocal_expansion_index中。
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        # 增广k_reciprocal_neigh数据，形成k_reciprocal_expansion_index
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  # 避免重复，并从小到大排序
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])  # 第i张图片与其前20+图片的权重
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)  # V记录第i个对其前20+个近邻的权重，其中有0有非0，非0表示没权重的，同时归一化

    # 裁剪original_dist矩阵，仅保留查询集的部分。
    # 接着，如果k2不等于1，进行k-reciprocal扩展的均值操作，得到扩展后的权重矩阵V。
    # 然后，删除initial_rank，并创建一个空列表invIndex，用于存储非零权重值的索引。
    original_dist = original_dist[:query_num, ]  # original_dist裁剪到 只有query x query+g

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):  # 遍历所有图片
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)  # 第i张图片在initial_rank前k2的序号的权重平均值
            # 第i张图的initial_rank前k2的图片对应全部图的权重平均值
            # 若V_qe中(i,j)=0，则表明i的前k2个相似图都与j不相似
        V = V_qe
        del V_qe
    del initial_rank

    # 逐列查找，返回行号
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        # 逐行查找，返回列号
        indNonZero = np.where(V[i, :] != 0)[0]
        # 根据列号在预先准备好的invIndex中获取行号
        indImages = [invIndex[ind] for ind in indNonZero]

        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2. - temp_min) # 1- min/max

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return -final_dist


def compute_jaccard_distance(target_features, k1=20, k2=6, use_float16=False):
    end = time.time()

    logging.info('Start computing jaccard distance')

    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
    index.add(target_features.numpy())
    _, initial_rank = search_index_pytorch(index, target_features, k1)
    res.syncDefaultStreamCurrentDevice()
    initial_rank = initial_rank.numpy()

    nn_k1 = []
    nn_k1_half = []
    # 创建了两个空列表 nn_k1 和 nn_k1_half，用于存储每个图像的 k 近邻和 k/2 近邻。
    # 通过一个循环迭代数据集中的每个图像，从 0 到 N-1。

    for i in range(N):
        # 在每次迭代中，使用 k_reciprocal_neigh 函数来获取当前图像的 k 互相近邻和 k/2 互相近邻，并将它们分别添加到 nn_k1 和 nn_k1_half 列表中。
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, round(k1/2)))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique

        # 余弦距离转马氏距离
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    # 如果 k2 不等于 1，则进行查询扩展。
    if k2 != 1:
        # 创建一个与 V 矩阵具有相同形状和数据类型的零矩阵 V_qe。
        V_qe = np.zeros_like(V, dtype=mat_type)
        # 循环遍历数据集中的每个图像，将其索引存储在变量 i 中。
        for i in range(N):
            # 对于每个图像 i，从 initial_rank 中获取其前 k2 个近邻的索引，然后从 V 矩阵中获取这些近邻的行，并计算它们的平均值。这个平均值将作为查询扩展后的相似性得分。
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)

        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]

        for j in range(len(indNonZero)):
            _temp_min = np.minimum(V[i, indNonZero[j]],V[indImages[j], indNonZero[j]])
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + _temp_min

        jaccard_dist[i] = 1 - (temp_min / (2 - temp_min))
    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    del pos_bool

    logging.info("Jaccard distance compute end, time cost: {}".format(time.time()-end))

    return jaccard_dist
