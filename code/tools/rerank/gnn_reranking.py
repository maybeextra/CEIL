"""
    Edit From Project: https://github.com/Xuanmeng-Zhang/gnn-re-ranking
"""

import torch
import numpy as np

import build_adjacency_matrix
import gnn_propagate

def propagate(initial_rank, A, S, k2, flag):
    for i in range(flag):
        # 将邻接矩阵变成无向图的邻接矩阵。这样做是为了保证图的对称性，确保每个节点的邻居节点都能相互连接。
        A = A + A.T
        # 将矩阵A的每一行除以该行的L2范数。对矩阵进行归一化，使得每一行的元素都在[0, 1]的范围内。
        A.div_(torch.norm(A, p=1, dim=1, keepdim=True))
        # 在图数据上进行信息传播
        A = gnn_propagate.forward(A, initial_rank[:, :k2].contiguous(), S[:, :k2].contiguous())
    return A

def gnn_reranking(query, gallery, query_cam = None, gall_cam = None, k1=28, k2=8, flag=3):
    query_num, gallery_num = query.shape[0], gallery.shape[0]

    X_all = torch.cat((query, gallery), axis=0)

    original_score = torch.mm(X_all, X_all.t())
    del X_all, query, gallery,

    S, initial_rank = original_score.topk(k=k1, dim=1, largest=True, sorted=True)
    del original_score

    D = torch.ones_like(S).half()
    if query_cam is not None and gall_cam is not None:
        labels_cam = np.concatenate((query_cam, gall_cam))
        for i, rank in enumerate(initial_rank):
            main_cam = labels_cam[i]
            same = labels_cam[rank.cpu()] == main_cam
            D[i, :][same] *= 0.5
            D[i, :][~same] *= 2

    initial_rank = initial_rank.int()
    S = (S ** 2).half()

    A = build_adjacency_matrix.forward(initial_rank, D)

    A = propagate(initial_rank, A, S, k2, flag)
    del initial_rank, S

    gnn_similarity = torch.mm(A[:query_num,:], A[query_num:,:].t())
    del A

    return gnn_similarity.cpu().numpy() # 返回相似度的负值便于排序
