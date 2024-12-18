"""
Created on Mon Jul 16 12:41:38 2018

@author: ssarfraz
 Expanded Cross Neighbourhood distance based Re-ranking (ECN)
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics, preprocessing


# ECN Re-ranking
def ECN(testset, queryset, query_cam = None, gall_cam = None, k1=None, k2=None, k=25, t=3, q=8, method='rankdist'):
    """ Expanded Cross Neighbourhood distance based Re-ranking (ECN)
    :param queryset: probe matrix (#_of_probes x featdim) feature vectors in rows
    :param testset:  Gallery matrix (feature vectors in rows)
    :param k, t, q: ECN parmaters (defaults k=25, t=3, q=8)
    :param method:  rankdist (default- based on rank list comparison) or origdist (based on orignol euclidean dist)
    :return:
            re-ranked distance matrix [shape: #test x #query]

    The code implements the ECN re-ranking algorithm described in our CVPR 2018 paper
     M. Saquib Sarfraz, Arne Schumann, Andreas Eberle, Ranier Stiefelhagen, " A
     Pose Sensitive Embedding for Person Re-Identification with Exapanded Cross
     Neighborhood Re-Ranking", http://openaccess.thecvf.com/content_cvpr_2018/papers/Sarfraz_A_Pose-Sensitive_Embedding_CVPR_2018_paper.pdf. CVPR 2018.

     Copyright
     M. Saquib Sarfraz (Karlsruhe Institute of Technology (KIT)), 2018
     For acedemic purpose only
    """
    queryset = queryset.cpu().numpy()
    testset = testset.cpu().numpy()
    nQuery = queryset.shape[0]
    ntest = testset.shape[0]
    mat = np.concatenate((queryset.astype(np.float32), testset.astype(np.float32)), axis=0)
    orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric='cosine', n_jobs=-1)
    initial_rank = orig_dist.argsort().astype(np.int32)

    if method == 'rankdist':
        r_dist = rankdist(initial_rank, k)
    else:
        r_dist = orig_dist

    top_t_nb = initial_rank[:, 1:t + 1]
    t_ind = top_t_nb[nQuery:, :].T
    next_2_tnbr = np.transpose(initial_rank[t_ind, 1:q + 1], [0, 2, 1])
    next_2_tnbr = np.reshape(next_2_tnbr, (t * q, ntest))
    t_ind = np.concatenate((t_ind, next_2_tnbr), axis=0)

    q_ind = top_t_nb[:nQuery, :].T
    next_2_qnbr = np.transpose(initial_rank[q_ind, 1:q + 1], [0, 2, 1])
    next_2_qnbr = np.reshape(next_2_qnbr, (t * q, nQuery))

    q_ind = np.concatenate((q_ind, next_2_qnbr), axis=0)

    t_nbr_dist = r_dist[t_ind, :nQuery]

    q_nbr_dist = r_dist[q_ind, nQuery:]
    q_nbr_dist = np.transpose(q_nbr_dist, [0, 2, 1])

    ecn_dist = np.mean(np.concatenate((q_nbr_dist, t_nbr_dist), axis=0), axis=0)

    # Use orig_dist values where rank-list based similarities are zero -- fixes behaviour in large scale open-ended retrievals
    ecn_dist = merge_dist(ecn_dist, orig_dist, nQuery)
    return -ecn_dist


def rankdist(initial_rank, k):
    pos_L1 = initial_rank.argsort().astype(np.int32)
    fac_1 = csr_matrix(np.maximum(0, k - pos_L1))
    rankdist = fac_1 @ fac_1.T
    return -rankdist.toarray()


def normalise_dist(dist):  ## OPTIONAL -- normalise values 0-1
    min_max_scaler = preprocessing.MinMaxScaler()
    dist = min_max_scaler.fit_transform(dist.T)
    return 1 - dist.T

# FIX: use orig_dist values where rank-list based similarities are zero -- fixes behaviour in large scale open.ended retrievals
def merge_dist(ecn_dist, orig_dist, nQuery):
    orig_dist = orig_dist[nQuery:, :nQuery]
    ecn_dist = np.where(ecn_dist != 0, ecn_dist, orig_dist)
    return ecn_dist
