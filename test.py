import numpy as np
import numpy as np 
import torch.nn as nn
from data_loader import VideoDataset
import torch

def extract_feature(dataloader, feature_extractor,use_gpu=True):
    features = []
    pids = []
    camids = []
    tracklets = []
    paths = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):            
            imgs = batch[0]
            if use_gpu:
                imgs = imgs.cuda(3)
            pid = batch[1]
            camid = batch[2]
            path = list(batch[3])
            if len(batch) > 4:
                tracklet = batch[4]
                tracklets.append(tracklet)

            feature = feature_extractor(imgs)

            features += [feature[:, :, 0].cpu()]
            pids += [pid]
            camids += [camid]
            paths += path
            #print(batch_idx)
        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy().astype('int')
        camids = torch.cat(camids, dim=0).numpy().astype('int')
        if len(tracklets) > 0:
            tracklets = torch.cat(tracklets, dim=0).numpy().astype('int')
    return features, pids, camids, paths, tracklets

def compute_distmat(a, b):
    m = a.size(0)
    n = b.size(0)
    distmat = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, a, b.t())
    distmat = distmat.clamp(min=1e-12).sqrt()
    return distmat.numpy()

def test(query, gallery, model,re_rank=False, pool='mean',use_gpu = True):

    feat_list = []
    pids_list = []
    camids_list = []
    for dataloader in [query,gallery]:
        features, pids, camids,paths, tracklets = extract_feature(dataloader,model,use_gpu)
        if isinstance(dataloader,VideoDataset):
            if dataloader.sampler == 'dense':
                seq_len = dataloader.dataset.seq_len
                m,n = feature.shape
                if pool == 'mean':
                    pool = nn.AdaptiveAvgPool2d(m//seq_len,n)
                elif pool == 'max':
                    pool = nn.AdaptiveMaxPool2d(m//seq_len, n)
                features = pool(features)
                tmp = list(range(0,len(pids),seq_len))
                pids = pids[tmp]
                camids = pids[tmp]
        feat_list.append(features)
        pids_list.append(pids)
        camids_list.append(camids)

    distmat = compute_distmat(feat_list[0],feat_list[1])

    if re_rank:
        q_q_dist = compute_distmat(feat_list[0], feat_list[0])
        g_g_dist = compute_distmat(feat_list[1],feat_list[1])
        distmat = re_ranking(distmat,q_q_dist,q_g_dist)
    max_rank = 50
    q_pids, g_pids = pids_list
    q_camids, g_camids = camids_list
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22. 
- This version accepts distance matrix instead of raw features. 
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1+1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    # change the cosine similarity metric to euclidean similarity metric
    original_dist = 2. - 2 * original_dist
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(
        1. * original_dist/np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition(original_dist, range(1, k1+1))

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(
                initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + \
                np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


