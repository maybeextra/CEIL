import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from code.tools.rerank.gnn_reranking import gnn_reranking
from code.tools.rerank.kr import re_ranking

def cross_modal_matching(k1, k2, proxies_features_rgb, proxies_features_ir):
    # 初始化两个空的字典，用来存储RGB到红外和红外到RGB的匹配关系。
    i2r = {}
    r2i = {}
    num_proxy_rgb = len(proxies_features_rgb)
    num_proxy_ir = len(proxies_features_ir)
    flag = True if num_proxy_rgb >= num_proxy_ir else False

    # 对可见光图像和红外图像的聚类特征进行归一化。
    proxies_features_rgb = F.normalize(proxies_features_rgb, dim=1)
    proxies_features_ir = F.normalize(proxies_features_ir, dim=1)
    # 如果RGB图像的聚类数量大于等于红外图像的聚类数量

    if flag:
        similarity = torch.from_numpy(
            gnn_reranking(proxies_features_rgb, proxies_features_ir, k1=k1, k2=k2)
        ).exp()
    else:
        similarity = torch.from_numpy(
            gnn_reranking(proxies_features_ir, proxies_features_rgb, k1=k1, k2=k2)
        ).exp()

    cost =  1 / similarity

    row_ind, col_ind = linear_sum_assignment(cost)

    for row, col in zip(row_ind, col_ind):
        if flag:
            r2i[row] = col
            i2r[col] = row
        else:
            r2i[col] = row
            i2r[row] = col

    # 找出所有被匹配的行
    matched_rows = np.unique(row_ind)
    # 找出所有没有被匹配的行
    all_rows = np.arange(cost.shape[0])
    unmatched_rows = np.setdiff1d(all_rows, matched_rows)

    if flag:
        r2i = cross_modal_matching_stage2(r2i, unmatched_rows, cost)
    else:
        i2r = cross_modal_matching_stage2(i2r, unmatched_rows, cost)

    return r2i, i2r

def cross_modal_matching_stage2(diction, unmatched_rows, cost):
    while len(unmatched_rows) != 0:
        unmatched_cost = cost[unmatched_rows]
        unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
        for row, col in zip(unmatched_row_ind, unmatched_col_ind):
            diction[unmatched_rows[row]] = col
        unmatched_rows = np.delete(unmatched_rows, unmatched_row_ind)

    return  diction

def cross_modal_matching_by_cam(args, cluster_features_rgb, cluster_features_ir, unique_cameras_rgb, unique_cameras_ir, cam2proxy_rgb, cam2proxy_ir, label2proxy_rgb, label2proxy_ir):
    # 初始化两个空的字典，用来存储RGB到红外和红外到RGB的匹配关系。

    num_proxy_rgb = len(cluster_features_rgb)
    num_proxy_ir = len(cluster_features_ir)

    i2r = [[] for _ in range(num_proxy_ir)]
    r2i = [[] for _ in range(num_proxy_rgb)]

    for cam_rgb in unique_cameras_rgb:
        proxy_rgb = cam2proxy_rgb[cam_rgb]
        num_rgb = len(proxy_rgb)

        features_rgb = F.normalize(cluster_features_rgb[proxy_rgb], dim=1) # (r,f)

        for cam_ir in unique_cameras_ir:
            proxy_ir = cam2proxy_ir[cam_ir]
            num_ir = len(proxy_ir)

            features_ir = F.normalize(cluster_features_ir[proxy_ir], dim=1) # (i,f)

            cost = 1 / torch.from_numpy(
                gnn_reranking(features_rgb, features_ir, k1=args.gnn_k1_cross, k2=args.gnn_k2_cross)
            ).exp()

            row_ind, col_ind = linear_sum_assignment(cost) # (m,m)
            for row, col in zip(row_ind, col_ind):
                r2i[proxy_rgb[row]].append(proxy_ir[col].cuda())
                i2r[proxy_ir[col]].append(proxy_rgb[row].cuda())

            if num_rgb > num_ir:
                # 找出所有被匹配的行
                matched_rows = np.unique(row_ind)
                # 找出所有没有被匹配的行
                all_rows = np.arange(cost.shape[0])
                unmatched_rows = np.setdiff1d(all_rows, matched_rows)

                r2i = cross_modal_matching_by_cam_stage2(r2i, cost, unmatched_rows, 'r2i', proxy_rgb, proxy_ir)

            elif num_rgb < num_ir:
                # 找出所有被匹配的列
                matched_rows = np.unique(col_ind)
                # 找出所有没有被匹配的列
                all_cols = np.arange(cost.shape[1])
                unmatched_cols = np.setdiff1d(all_cols, matched_rows)

                i2r = cross_modal_matching_by_cam_stage2(i2r, cost, unmatched_cols, 'i2r', proxy_rgb, proxy_ir)

    r2i = [torch.stack(tensor, dim=0) if len(tensor) != 0 else torch.Tensor() for tensor in r2i]
    i2r = [torch.stack(tensor, dim=0) if len(tensor) != 0 else torch.Tensor() for tensor in i2r]




    return r2i, i2r

def cross_modal_matching_by_cam_stage2(dictionary, cost, unmatched, kind, proxy_rgb, proxy_ir):
    if kind == 'r2i':
        while unmatched:
            unmatched_cost = cost[unmatched, :]
            unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
            for row, col in zip(unmatched_row_ind, unmatched_col_ind):
                dictionary[proxy_rgb[unmatched[row]]].append(proxy_ir[col].cuda())

            unmatched = np.delete(unmatched, unmatched_row_ind)

    else: # 'i2r'
        while unmatched:
            unmatched_cost = cost[:, unmatched]
            unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
            for row, col in zip(unmatched_row_ind, unmatched_col_ind):
                dictionary[proxy_ir[unmatched[col]]].append(proxy_rgb[row].cuda())

            unmatched = np.delete(unmatched, unmatched_col_ind)

    return dictionary
