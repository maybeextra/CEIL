"""Eval for GCR."""
import numpy as np
import torch


def ComputeEuclid(array1,array2,fg_sqrt=True,fg_norm=False):
    #array1:[m1,n],array2:[m2,n]
    assert array1.shape[1]==array2.shape[1];
    # norm
    if fg_norm:
        array1 = array1/np.linalg.norm(array1,ord=2,axis=1,keepdims=True)
        array2 = array2/np.linalg.norm(array2,ord=2,axis=1,keepdims=True)
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    #print 'array1,array2 shape:',array1.shape,array2.shape
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    #shape [m1,m2]
    if fg_sqrt:
        dist = np.sqrt(squared_dist)
        #print('[test] using sqrt for distance')
    else:
        dist = squared_dist
        #print('[test] not using sqrt for distance')
    sim = 1-dist**2/2.
    return 1-sim


def mergesetfeat4(X, labels):
    """Run GCR for one iteration."""

    labels_cam = labels
    unique_labels_cam = np.unique(labels_cam)
    index_dic = {item: [] for item in unique_labels_cam}
    for labels_index, item in enumerate(labels_cam):
        index_dic[item].append(labels_index)

    beta1 = 0.08
    beta2 = 0.08
    k1 = 45
    k2 = 90
    scale = 0.7

    # compute global feat
    sim = X.dot(X.T)

    if scale != 1.0:
        rank = gpu_argsort(-sim)

        S = np.zeros(sim.shape)
        for i in range(0, X.shape[0]):
            S[i, rank[i, :k1]] = np.exp(sim[i, rank[i, :k1]]/beta1)
            S[i, i] = np.exp(sim[i, i]/beta1)    # this is duplicated???

        D_row = np.sqrt(1. / np.sum(S, axis=1))
        D_col = np.sqrt(1. / np.sum(S, axis=0))
        L = np.outer(D_row, D_col) * S
        global_X = L.dot(X)
    else:
        global_X = 0.0

    if scale != 0.0:
        # compute cross camera feat
        for i in range(0, X.shape[0]):
            tmp = sim[i, i]
            sim[i, index_dic[labels[i]]] = -2
            sim[i, i] = tmp

        rank = gpu_argsort(-sim)

        S = np.zeros(sim.shape)
        for i in range(0, X.shape[0]):
            S[i, rank[i, :k2]] = np.exp(sim[i, rank[i, :k2]] / beta2)
            S[i, i] = np.exp(sim[i, i]/beta2)    # this should not be ommited

        D_row = np.sqrt(1. / np.sum(S, axis=1))
        D_col = np.sqrt(1. / np.sum(S, axis=0))
        L = np.outer(D_row, D_col) * S
        cross_X = L.dot(X)
    else:
        cross_X = 0.0

    X = scale*cross_X+(1-scale)*global_X
    X /= np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    return X

def mergesetfeat1_notrk(P, neg_vector, in_feats, in_labels):
    """Mergesetfeat1 notrk"""

    out_feats = []
    for i in range(in_feats.shape[0]):
        camera_id = in_labels[i]
        feat = in_feats[i] - neg_vector[camera_id]
        feat = P[camera_id].dot(feat)
        # random neg vec
        # rand_vec = np.random.random((512,))
        # rand_vec = rand_vec / np.linalg.norm(rand_vec, ord=2) * 0.1
        # feat = in_feats[i] - rand_vec

        feat = feat/np.linalg.norm(feat, ord=2)
        out_feats.append(feat)
    out_feats = np.vstack(out_feats)
    return out_feats


def mergesetfeat1(_cfg, P, neg_vector, in_feats, in_labels, in_tracks):
    """Run PVG."""

    trackset = np.unique(in_tracks)
    # trackset = list(set(list(in_tracks)))
    out_feats = []
    out_labels = []
    track_index_dic = {item: [] for item in trackset}
    for track_index, item in enumerate(in_tracks):
        track_index_dic[item].append(track_index)

    for track in trackset:
        indexes = track_index_dic[track]
        camera_id = in_labels[indexes, 1][0]

        feat = np.mean(in_feats[indexes], axis=0) - neg_vector[camera_id]
        feat = P[camera_id].dot(feat)

        feat = feat/np.linalg.norm(feat, ord=2)
        label = in_labels[indexes][0]
        out_feats.append(feat)
        out_labels.append(label)
        # if len(out_feats) % 100 == 0:
        #     print('%d/%d' %(len(out_feats), len(trackset)))
    out_feats = np.vstack(out_feats)
    out_labels = np.vstack(out_labels)
    return out_feats, out_labels


def compute_P_all(gal_feats, gal_labels, la):
    """Compute P and neg for all data(global cameras)."""

    X = gal_feats
    neg_vector_all = np.mean(X, axis=0).astype('float32')
    # la = 0.04
    P_all = np.linalg.inv(X.T.dot(X)+X.shape[0]*la*np.eye(X.shape[1])).astype('float32')
    neg_vector = {}
    u_labels = np.unique(gal_labels[:, 1])
    P = {}
    for label in u_labels:
        P[label] = P_all
        neg_vector[label] = neg_vector_all
    return P, neg_vector

def compute_P2(gal_feats, gal_labels, la):
    """Compute P2 on gal data????"""

    X = gal_feats
    neg_vector = {}
    u_labels = np.unique(gal_labels)
    P = {}
    for label in u_labels:
        curX = gal_feats[gal_labels == label, :]
        neg_vector[label] = np.mean(curX, axis=0).astype('float32')
        P[label] = np.linalg.inv(curX.T.dot(curX)+curX.shape[0]*la*np.eye(X.shape[1])).astype('float32')
    return P, neg_vector


def run_pvg(prb_feats, prb_labels, gal_feats, gal_labels):
    """Run pvg."""
    P, neg_vector = compute_P2(prb_feats, prb_labels, la=0.06)
    prb_feats = mergesetfeat1_notrk(P, neg_vector, prb_feats, prb_labels)
    gal_feats = mergesetfeat1_notrk(P, neg_vector, gal_feats, gal_labels)
    return prb_feats, gal_feats


def run_gcr(prb_feats, prb_labels, gal_feats, gal_labels):
    """Run GCR."""
    prb_n = len(prb_labels)
    data = np.vstack((prb_feats, gal_feats))
    labels = np.concatenate((prb_labels, gal_labels))

    for gal_round in range(3):
        data = mergesetfeat4(data, labels)

    prb_feats_new = data[:prb_n, :]
    gal_feats_new = data[prb_n:, :]
    return prb_feats_new, gal_feats_new


def GCRV(prb_feats, gal_feats, prb_labels, gal_labels, k1=None, k2=None):
    """GCRV image port."""
    prb_feats = prb_feats.cpu().numpy()
    gal_feats = gal_feats.cpu().numpy()

    # prb_feats, gal_feats = run_pvg(prb_feats, prb_labels, gal_feats, gal_labels)
    prb_feats, gal_feats = run_gcr(prb_feats, prb_labels, gal_feats, gal_labels)
    sims = ComputeEuclid(prb_feats, gal_feats, 1)
    return -sims


def gpu_argsort(temp):
    """Use torch for faster argsort."""
    temp = torch.from_numpy(temp).to('cuda').half()
    rank = torch.argsort(temp, dim=1).cpu().numpy()
    return rank
