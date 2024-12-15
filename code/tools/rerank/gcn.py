"""
    Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective

    Xuanmeng Zhang, Minyue Jiang, Zhedong Zheng, Xiao Tan, Errui Ding, Yi Yang

    Project Page : https://github.com/Xuanmeng-Zhang/gnn-re-ranking

    Paper: https://arxiv.org/abs/2012.07620v2

    ======================================================================

    On the Market-1501 dataset, we accelerate the re-ranking processing from 89.2s to 9.4ms
    with one K40m GPU, facilitating the real-time post-processing. Similarly, we observe
    that our method achieves comparable or even better retrieval results on the other four
    image retrieval benchmarks, i.e., VeRi-776, Oxford-5k, Paris-6k and University-1652,
    with limited time cost.
"""

import torch

import build_adjacency_matrix
import gnn_propagate


def gcn(query, gallery, query_cam = None, gall_cam = None, k1 = 28, k2 = 8, flag = 3):
    query_num, gallery_num = query.shape[0], gallery.shape[0]
    original_cos = torch.mm(query, gallery.t())

    X_u = torch.cat((query, gallery), axis=0)
    original_score = torch.mm(X_u, X_u.t())
    del X_u, query, gallery

    # initial ranking list
    S, initial_rank = original_score.topk(k=k1, dim=-1, largest=True, sorted=True)

    # stage 1
    A = build_adjacency_matrix.forward(initial_rank.int(), S.half())
    S = S * S

    # stage 2
    if k2 != 1:
        for i in range(flag):
            A = A + A.T
            A = gnn_propagate.forward(A, initial_rank[:, :k2].contiguous().int(), S[:, :k2].contiguous().float().half())
            A_norm = torch.norm(A, p=2, dim=1, keepdim=True)
            A = A.div(A_norm.expand_as(A))

    cosine_similarity = torch.mm(A[:query_num, ], A[query_num:, ].t())
    del A, S
    cosine_similarity = 0.7 * cosine_similarity + 0.3 * original_cos

    return cosine_similarity.cpu().numpy()