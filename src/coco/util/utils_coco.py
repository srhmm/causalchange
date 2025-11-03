import itertools
import math
from collections import defaultdict

import numpy as np

from src.coco.util.co_test_causal import test_causal
from src.coco.util.co_test_confounding import test_confounded
from src.coco.util.co_test_types import CoCoTestType
from src.coco.gen.dag_confounded import clus_sim_spectral
from src.coco.util.mi import mutual_info_scores
from src.coco.util.mi_sampling import Sampler


def to_affinity_mat(nodes, sim_mi, sim_01, sim_pval):

    confounded_lookup = []
    node_lookup = [i for i in nodes]
    for i, j in itertools.combinations(range(len(node_lookup)), 2):
        n_i, n_j = node_lookup[i], node_lookup[j]
        if (n_i !=n_j) and sim_01[n_i][n_j]:
            if (n_i not in confounded_lookup):
                confounded_lookup.append(n_i)
            if (n_j not in confounded_lookup):
                confounded_lookup.append(n_j)
    mat_01_sub =  [[0 for _ in  confounded_lookup] for _ in confounded_lookup]
    mat_mi_sub =  [[0 for _ in  confounded_lookup] for _ in confounded_lookup]
    mat_pval_sub =  [[0 for _ in  confounded_lookup] for _ in confounded_lookup]


    mat_mi = [[0 for _ in  nodes] for _ in nodes]
    mat_pval = [[0 for _ in nodes] for _ in nodes]
    mat_01 = [[0 for _ in nodes] for _ in nodes]

    for i, j in itertools.combinations(range(len(confounded_lookup)), 2):
        n_i, n_j = confounded_lookup[i], confounded_lookup[j]
        if n_j in sim_01[n_i]:
            if sim_01[n_i][n_j]:
                mi = sim_mi[n_i][n_j]
                pv = sim_pval[n_i][n_j]
                mat_01_sub[i][j], mat_01_sub[j][i] = 1, 1
                mat_mi_sub[i][j], mat_mi_sub[j][i] = mi, mi
                mat_pval_sub[i][j], mat_pval_sub[j][i] = pv, pv
        elif n_i in sim_01[n_j]:
            if sim_01[n_j][n_i]:
                mi = sim_mi[n_j][n_i]
                pv = sim_pval[n_j][n_i]
                mat_01_sub[i][j], mat_01_sub[j][i] = 1, 1
                mat_mi_sub[i][j], mat_mi_sub[j][i] = mi, mi
                mat_pval_sub[i][j], mat_pval_sub[j][i] = pv, pv

    for i, j in itertools.combinations(range(len(node_lookup)), 2):
        n_i, n_j = node_lookup[i], node_lookup[j]

        mi = sim_mi[n_i][n_j]  # this is mi-emi
        pval = sim_pval[n_i][n_j]
        i = 0
        if sim_01[n_i][n_j] :
            i = 1
        mat_mi[i][j], mat_mi[j][i] = mi, mi
        mat_pval[i][j], mat_pval[j][i] = pval, pval
        mat_01[i][j], mat_01[j][i] = i, i
        mat_mi[i][i], mat_mi[j][j] = 1 ,1
        mat_01[i][i], mat_01[j][j]  = 1, 1

    for i, j in itertools.combinations(range(len(confounded_lookup)), 2):
        n_i, n_j = confounded_lookup[i], confounded_lookup[j]

        if n_j in sim_01[n_i]:
            assert ((sim_01[n_i][n_j] and mat_01_sub[i][j] == 1) or ((not sim_01[n_i][n_j]) and mat_01_sub[i][j] == 0))
        elif n_i in sim_01[n_j]:
            assert ((sim_01[n_j][n_i] and mat_01_sub[i][j] == 1) or ((not sim_01[n_j][n_i]) and mat_01_sub[i][j] == 0))


    return mat_mi, mat_01, mat_pval, node_lookup, \
           mat_mi_sub, mat_01_sub, mat_pval_sub,  confounded_lookup


def node_similarities(maps_nodes, nodes, test: CoCoTestType, sampler:Sampler):

    similarity_01 = defaultdict(defaultdict)
    similarity_mi = defaultdict(defaultdict)
    similarity_pval = defaultdict(defaultdict)

    similarity_causal_01 = defaultdict(defaultdict)
    similarity_cent = defaultdict(defaultdict)
    similarity_causal_pval = defaultdict(defaultdict)

    for n_i, n_j in itertools.combinations(nodes, 2):
        map_i = maps_nodes[n_i]
        map_j = maps_nodes[n_j]

        decision_mi, mi, pval_mi, stdev_cent_sampling = test_confounded(map_i, map_j, test, sampler)

        mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)
        vi = h1 + h2 - 2*mi

        decision_causal, pval_causal = test_causal(h1, mi, stdev_cent_sampling) #sampler

        similarity_mi[n_i][n_j] = mi-emi
        similarity_01[n_i][n_j] = decision_mi
        similarity_pval[n_i][n_j] = pval_mi

        similarity_causal_01[n_i][n_j] = decision_causal
        similarity_cent[n_i][n_j] = mi - h1
        similarity_causal_pval[n_i][n_j] = pval_causal

    return similarity_mi, similarity_01, similarity_pval,\
           similarity_cent, similarity_causal_01, similarity_causal_pval
'''
def graph_cuts_n(n_components, nodes, sim_mi, sim_01, sim_pval, soft=True):


    mat_mi, mat_01, mat_pval, node_lookup, \
    mat_mi_sub, mat_01_sub, mat_pval_sub, confounded_lookup = to_affinity_mat(nodes, sim_mi, sim_01, sim_pval)

    if len(confounded_lookup) == 0:
        return []
    if soft:
        mat = mat_mi_sub #want clusters of nodes with high pairwise mutual information
    else:
        mat = mat_01_sub #clustering using 0/1 indicators (confounded or not) also works

    mat = mat + np.ones((len(confounded_lookup), len(confounded_lookup))) * 0.01 #avoid 0 entries for spectral clustering
    try:
        labels = clus_sim_spectral(mat, n_components)
    except:  # kmeans: ValueError( f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}.")
        labels = [0 for _ in range(len(confounded_lookup))]

    def disagreement(n_i, n_j, labels, sim):
        if labels[n_i] == labels[n_j]:
            return sim[n_i][n_j]
        else:
            return -sim[n_i][n_j]

    eval_n = sum(
        [disagreement(n_i, n_j, labels, mat_mi_sub) for n_i, n_j in itertools.combinations(range(len(labels)), 2)])

    confounder_list = []
    for g in np.unique(labels):
        confounder_list.append([ci for i, ci in enumerate(confounded_lookup) if i in np.where(labels == g)[0]])
    return confounder_list, eval_n 
'''

def graph_cuts(nodes, sim_mi, sim_01, sim_pval, n_components=None, soft=False,verbosity=0):

    mat_mi, mat_01, mat_pval, node_lookup, \
    mat_mi_sub, mat_01_sub, mat_pval_sub, confounded_lookup = to_affinity_mat(nodes, sim_mi, sim_01, sim_pval)

    if len(confounded_lookup) == 0:
        return []
    if soft:
        mat = mat_mi_sub  # want clusters of nodes with high pairwise mutual information
    else:
        mat = mat_01_sub  # clustering using 0/1 indicators (confounded or not) also works

    mat = mat + np.ones(
        (len(confounded_lookup), len(confounded_lookup))) * 0.01  # avoid 0 entries for spectral clustering

    labels = [0 for _ in range(len(confounded_lookup))]
    if n_components is not None:
        try:
            labels = clus_sim_spectral(mat, n_components)
        except:  #empty mat(?) kmeans: ValueError( f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}.")
            pass

        if verbosity > 0:
            print(f"(COCO-confounding) Known components: {len(np.unique(labels))} confounded sets")
    else:
        def graph_cuts_sqsum(n_i, n_j, labels, sim):
            if labels[n_i] == labels[n_j]:
                return sim[n_i][n_j] ** 2
            else:
                return 0
        max_n = int(np.floor(len(mat)/2))
        labels_N =  [[] for _ in range(max(max_n,1))]
        eval_N = [0 for _ in range(max(max_n,1))]
        labels_N[0] = labels
        eval_N[0] = sum(
                [graph_cuts_sqsum(n_i, n_j, labels, mat_mi_sub) for n_i, n_j in itertools.combinations(range(len(labels)), 2)])
        for n_components in range(1, max(max_n,1)):
            labels_n = clus_sim_spectral(mat, n_components)
            labels_N[n_components] = labels_n
            #each node has to be confounded
            valid_n = not (False in [len(np.where(labels_n==i)[0]) > 1 for i in np.unique(labels_n)])
            #goodness of clustering
            if valid_n:


                eval_N[n_components] = sum(
                [graph_cuts_sqsum(n_i, n_j, labels_n, mat_mi_sub) for n_i, n_j in itertools.combinations(range(len(labels_n)), 2)])
            else:
                eval_N[n_components] = math.inf
        labels = labels_N[0]
        #if np.min(eval_N) > 0:  # oterwise all clusterings are invalid (incl case <0 or not?)
        labels = labels_N[np.argmin(eval_N)]

        if verbosity > 0:
            print(f"(COCO-confounding) *** Discovered {len(np.unique(labels))} confounded sets ({[e for e in eval_N]})")
    confounder_list = []

    for g in np.unique(labels):
        confounder_list.append([ci for i, ci in enumerate(confounded_lookup) if i in np.where(labels == g)[0]])
    return confounder_list

