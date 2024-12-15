"""
    Edit From Project: https://github.com/Xuanmeng-Zhang/gnn-re-ranking
"""

import build_adjacency_matrix
import gnn_propagate
import numpy as np
import torch


def propagate(initial_rank, A, S, k2, flag):
    for i in range(flag):
        A = A + A.T
        A.div_(torch.norm(A, p=1, dim=1, keepdim=True))
        A = gnn_propagate.forward(A, initial_rank[:, :k2].contiguous(), S[:, :k2].contiguous())
    return A

def gnn_reranking(query, gallery, query_cam = None, gall_cam = None, k1=28, k2=8, flag=3, la=0.3):
    query_num, gallery_num = query.shape[0], gallery.shape[0]

    X_all = torch.cat((query, gallery), axis=0)

    original_score = torch.mm(X_all, X_all.t())
    del X_all, query, gallery,

    S, initial_rank = original_score.topk(k=k1, dim=1, largest=True, sorted=True)
    del original_score

    D = torch.ones_like(S).half()

    if query_cam is not None and gall_cam is not None:
        labels_cam = np.concatenate((query_cam, gall_cam))
        labels_cam_tensor = torch.from_numpy(labels_cam).to(S.device)
        main_cam = labels_cam_tensor.unsqueeze(1)
        ranked_cam = labels_cam_tensor[initial_rank]
        same_camera_mask = (ranked_cam == main_cam)
        D[same_camera_mask] *= la
        D[~same_camera_mask] *=  (1 / la)

    initial_rank = initial_rank.int()
    S = (S ** 2).half()

    A = build_adjacency_matrix.forward(initial_rank, D)

    A = propagate(initial_rank, A, S, k2, flag)
    del initial_rank, S

    gnn_similarity = torch.mm(A[:query_num,:], A[query_num:,:].t())
    del A

    return gnn_similarity.cpu().numpy() # 返回相似度的负值便于排序
