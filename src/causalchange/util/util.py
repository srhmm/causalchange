import itertools
import statistics
from collections import defaultdict

import causaldag
import causallearn.graph.GeneralGraph
import networkx as nx
import sklearn
from sklearn import preprocessing
import numpy as np

from src.exp.exp_change.gen.generate import GSType
from src.causalchange.util.utils_idl import _get_true_idl_Z

from src.causalchange.util.eval import rates, matthews_correlation, positives_negatives
from src.baselines.sep_distances.codebase import mixed_graph as graph_lmg
from src.baselines.sep_distances.codebase import metrics as graph_metrics


# %% Evaluation of mixing assignments and targeted node sets
def compare_Z(n_samp, t_A, e_A, t_Z, t_n_Z, e_Z, Z_pairs, e_n_Z, e_Z_n, e_Z_n_p=None, gs:GSType=GSType.GRAPH):
    """ for easier logging, no underscores in dict names """
    # %% eval per node
    mets = defaultdict(list)
    for nodei in range(len(t_A)):
        estim_idl = e_Z_n[nodei]
        #estim_pproba = e_Z_n_p[nodei]
        true_parents_target = np.where(t_A[:, nodei] != 0)[0]
        true_idl = _get_true_idl_Z(true_parents_target, nodei, t_A, t_Z, t_n_Z, n_samp)

        mets['node-aris'].append(sklearn.metrics.adjusted_rand_score(true_idl, estim_idl))
        mets['node-amis'].append(sklearn.metrics.adjusted_mutual_info_score(true_idl, estim_idl))
        mets['node-vmeas'].append(sklearn.metrics.v_measure_score(true_idl, estim_idl))
        mets['node-fms'].append(sklearn.metrics.fowlkes_mallows_score(true_idl, estim_idl))

        # debug, larger graphs: distinguish between confounded cases (and those where the given node has a parent (-pa))
        if not gs.is_bivariate():
            for nodei in range(len(t_A)):
                ami = sklearn.metrics.adjusted_mutual_info_score(true_idl, estim_idl)
                if len(np.unique(true_idl)) == 1:
                    mets['node-amis-nocf'].append(ami)
                else:
                    mets['node-amis-cf'].append(ami)
                    mets['node-amis-pa'].append(ami) if len(true_parents_target) > 0 else mets['node-amis-nopa'].append(
                        ami)

    # debug, bivariate graph structures: specifically evaluate the parent X and effect Y
    if gs.is_bivariate():
        for nodei, nm in [(1, 'X'), (2, 'Y')]:
            estim_idl = e_Z_n[nodei]
            #estim_pproba = e_Z_n_p[nodei]
            true_parents_target = np.where(t_A[:, nodei] != 0)[0]
            true_idl = _get_true_idl_Z(true_parents_target, nodei, t_A, t_Z, t_n_Z, n_samp)

            mets[f'node-amis-{nm}'].append(sklearn.metrics.adjusted_mutual_info_score(true_idl, estim_idl))
            mets[f'node-aris-{nm}'].append(sklearn.metrics.adjusted_rand_score(true_idl, estim_idl))
            mets[f'node-vmeas-{nm}'].append(sklearn.metrics.v_measure_score(true_idl, estim_idl))
            mets[f'node-fms-{nm}'].append(sklearn.metrics.fowlkes_mallows_score(true_idl, estim_idl))
            mets[f'diff-k-{nm}'].append(np.abs(len(np.unique(true_idl))-len(np.unique(estim_idl))))

    res = dict()
    for ky in mets: # debug, avg over the cases
        res[ky] = -1 if len(mets[ky]) == 0 else mets[ky][0] if len(mets[ky]) < 2 else statistics.mean(
            mets[ky]) #  after all runs might be better, setting to -1s can distort statistics (but used only for debug)

    # %% eval node set
    true_set = {node for node in range(len(t_A)) if any([node in st for st in t_n_Z])}
    e_set = {node for node in range(len(t_A)) if any([node in st for st in e_n_Z])}
    jacc = 1 if len(true_set.union(e_set))==0 else len(true_set.intersection(e_set)) / len(true_set.union(e_set))

    # %% eval node pairs
    tp_pair, fp_pair, fn_pair, tn_pair = 0, 0, 0, 0
    for n1, n2 in itertools.combinations(range(len(t_A)), 2):
        is_cf = any([n1 in nset and n2 in nset for nset in t_n_Z])
        est_cf = any([n1 in nset and n2 in nset for nset in e_n_Z]) # or Z_pairs??
        if is_cf:
            if est_cf: tp_pair += 1
            else: fp_pair += 1
        else:
            if est_cf: fn_pair += 1
            else: tn_pair += 1

    # %% eval per node: intervened or not
    tp_z, fp_z, fn_z, tn_z = 0, 0, 0, 0

    for n1 in range(len(t_A)):
        is_cf = any([n1 in nset for nset in t_n_Z])
        est_cf = any([n1 in nset for nset in e_n_Z])
        if is_cf:
            if est_cf: tp_z += 1
            else: fp_z += 1
        else:
            if est_cf: fn_z += 1
            else: tn_z += 1
    den = tp_z + 1 / 2 * (fp_z + fn_z)
    f1_z = tp_z / den if den > 0 else 1
    den = tp_pair + 1 / 2 * (fp_pair + fn_pair)
    f1_pair = tp_pair / den if den > 0 else 1

    tpr_z, fpr_z, tnr_z, fnr_z = rates(tp_z, tn_z, fp_z, fn_z)
    res.update({
        'jacc': jacc,
        'tp-pair': tp_pair, 'fp-pair': fp_pair, 'fn-pair': fn_pair, 'tn-pair': tn_pair, 'f1-pair': f1_pair,
        'tp-iv': tp_z, 'fp-iv': fp_z, 'fn-iv': fn_z, 'tn-iv': tn_z,
        'tpr-iv': tpr_z, 'fpr-iv': fpr_z, 'fnr-iv': fnr_z, 'tnr-iv': tnr_z, 'f1-iv': f1_z,
    })
    return res


# %% Evaluation of causal graphs
def compare_lmg_CPDAG(true_lmg, est_lmg):
    metrics = dict()
    metrics['sc'] = graph_metrics.metric_CPDAGs(true_lmg, est_lmg)
    metrics['shd'] = graph_metrics.SHD_CPDAGs(true_lmg, est_lmg)
    metrics['sd'] = graph_metrics.SD_CPDAGs(true_lmg, est_lmg)
    metrics['anc-aid'] = graph_metrics.ancestor_AID_CPDAGs(true_lmg, est_lmg)
    metrics['parent-aid'] = graph_metrics.parent_AID_CPDAGs(true_lmg, est_lmg)
    metrics['sym-sd'] = graph_metrics.sym_SD_CPDAGs(true_lmg, est_lmg)
    metrics['sym-anc-aid'] = graph_metrics.sym_ancestor_AID_CPDAGs(true_lmg, est_lmg)
    metrics['sym-pa-aid'] = graph_metrics.sym_parent_AID_CPDAGs(true_lmg, est_lmg)

    #metrics['wgt-sd'] = ...

    n_n = len(true_lmg.directed)
    shd_nm = metrics['shd'] / ((n_n ** 2) if n_n > 0 else 1)
    metrics.update({'shd-nm': shd_nm})
    metrics.update(compare_adj_directed_edges(true_lmg, est_lmg))
    return metrics

def compare_lmg_DAG(true_lmg, est_lmg):
    # for easier logging, no underscores in dict names
    metrics = dict()
    metrics['sc'] = graph_metrics.metric_DAGs(true_lmg, est_lmg)
    metrics['shd'] = graph_metrics.SHD_DAGs(true_lmg, est_lmg)
    metrics['sd'] = graph_metrics.SD_DAGs(true_lmg, est_lmg)
    metrics['anc-aid'] = graph_metrics.ancestor_AID_DAGs(true_lmg, est_lmg)
    metrics['parent-aid'] = graph_metrics.parent_AID_DAGs(true_lmg, est_lmg)
    metrics['sym-sd'] = graph_metrics.sym_SD_DAGs(true_lmg, est_lmg)
    metrics['sym-anc-aid'] = graph_metrics.sym_ancestor_AID_DAGs(true_lmg, est_lmg)
    metrics['sym-pa-aid'] = graph_metrics.sym_parent_AID_DAGs(true_lmg, est_lmg)
    #metrics['wgt-sd'] = graph_metrics.weighted_SD_DAGs(true_lmg, est_lmg) #no cpdag equivalent?

    n_n = len(true_lmg.directed)
    shd_nm = metrics['shd'] / ((n_n ** 2) if n_n > 0 else 1)
    metrics.update({'shd-nm': shd_nm})
    metrics.update(compare_adj_directed_edges(true_lmg, est_lmg))
    return metrics

def compare_adj_directed_edges(true_lmg, est_lmg):
    # for easier logging, no underscores in dict names
    true_adj = lmg_to_directed_edge_adj(true_lmg)
    est_adj = lmg_to_directed_edge_adj(est_lmg)
    tp, tn, fp, fn = positives_negatives(true_adj, est_adj)

    #tp = sum([sum([1 if (est_adj[i][j] != 0 and true_adj[i][j] != 0) else 0
    #               for j in range(len(true_adj[i]))]) for i in range(len(true_adj))])
    #tn = sum([sum([1 if (est_adj[i][j] == 0 and true_adj[i][j] == 0) else 0
    #               for j in range(len(true_adj[i]))]) for i in range(len(true_adj))])
    #fp = sum([sum([1 if (est_adj[i][j] != 0 and true_adj[i][j] == 0) else 0
    #               for j in range(len(true_adj[i]))]) for i in range(len(true_adj))])
    #fn = sum([sum([1 if (est_adj[i][j] == 0 and true_adj[i][j] != 0) else 0
    #               for j in range(len(true_adj[i]))]) for i in range(len(true_adj))])

    assert (fp==sum([sum([1 if (est_adj[i,j] == 1 and true_adj[i,j] == 0) else 0 for j in range(len(true_adj))]) for i in range(len(true_adj))]))
    assert (fn==sum([sum([1 if (est_adj[i,j] == 0 and true_adj[i,j] == 1) else 0  for j in range(len(true_adj))]) for i in range(len(true_adj))]))
    assert (tp==sum([sum([1 if (est_adj[i,j] == 1 and true_adj[i,j] == 1) else 0 for j in range(len(true_adj))]) for i in range(len(true_adj))]))
    assert ( tp + tn + fn + fp == est_adj.shape[0] * est_adj.shape[1])

    den = tp + 1 / 2 * (fp + fn)
    f1 = tp / den if den > 0 else 1

    tpr, fpr, tnr, fnr = rates(tp, tn, fp, fn)
    mcc = matthews_correlation(tp, tn, fp, fn)

    return dict(f1=f1, tp=tp, fp=fp, fn=fn, tn=tn, tpr=tpr, fpr=fpr, pr=(fp+tp)/(fp+tp+tn+fn), tnr=tnr, fnr=fnr, mcc=mcc)

# %% Evaluation of context partitions

def _pairwise_f1(y_true, y_pred):
    """F1 over 'same-cluster' for all unordered pairs."""
    n = len(y_true)
    # pairwise same/diff (upper triangle)
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i+1, n):
            same_t = (y_true[i] == y_true[j])
            same_p = (y_pred[i] == y_pred[j])
            if same_p and same_t:
                tp += 1
            elif same_p and not same_t:
                fp += 1
            elif (not same_p) and same_t:
                fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
    return {"pairwise_precision": prec, "pairwise_recall": rec, "pairwise_f1": f1}

def _contingency(y_true, y_pred):
    """Contingency table as a dict-of-dicts and arrays for ARI/NMI."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes, class_idx = np.unique(y_true, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred, return_inverse=True)
    C = np.zeros((classes.size, clusters.size), dtype=int)
    for i in range(y_true.size):
        C[class_idx[i], cluster_idx[i]] += 1
    return C

def _comb2(x):
    x = np.asarray(x, dtype=np.int64)
    return x*(x-1)//2

def adjusted_rand_index(y_true, y_pred):
    """ARI (Hubert & Arabie, 1985) — pure NumPy implementation."""
    C = _contingency(y_true, y_pred)
    sum_comb_c = _comb2(C).sum()
    a = C.sum(axis=1)
    b = C.sum(axis=0)
    sum_comb_a = _comb2(a).sum()
    sum_comb_b = _comb2(b).sum()
    n = C.sum()
    total_comb = _comb2(n)
    expected = (sum_comb_a * sum_comb_b) / total_comb if total_comb > 0 else 0.0
    max_ix = 0.5*(sum_comb_a + sum_comb_b)
    denom = max_ix - expected
    return (sum_comb_c - expected) / denom if denom != 0 else 1.0

def normalized_mutual_info(y_true, y_pred, eps=1e-12):
    """NMI (symmetric, sqrt normalization)."""
    C = _contingency(y_true, y_pred).astype(float)
    n = C.sum()
    pi = C.sum(axis=1, keepdims=True) / n
    pj = C.sum(axis=0, keepdims=True) / n
    P = C / n
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.log((P + eps) / (pi @ pj + eps))
    mi = float((P * log_term).sum())
    hi = float(-(pi * np.log(pi + eps)).sum())
    hj = float(-(pj * np.log(pj + eps)).sum())
    return mi / np.sqrt(hi*hj + eps)

def try_ami(y_true, y_pred):
    """Try sklearn's AMI; fall back to NMI if sklearn not available."""
    try:
        from sklearn.metrics import adjusted_mutual_info_score
        return adjusted_mutual_info_score(y_true, y_pred), True
    except Exception:
        return normalized_mutual_info(y_true, y_pred), False

def evaluate_context_partition(node_id, fit_result, mechanism_partitions):
    contexts = fit_result["contexts"]                     # e.g. [0,1,...,C-1]
    y_pred = np.asarray(fit_result["labels_pred"], int)   # shape (C_used,)

    y_true_all = np.asarray(mechanism_partitions[node_id], int)
    y_true = y_true_all[np.asarray(contexts, int)]

    pair = _pairwise_f1(y_true, y_pred)
    ari  = adjusted_rand_index(y_true, y_pred)
    ami_value, is_ami = try_ami(y_true, y_pred)

    C_used = len(y_true)
    n_pairs = C_used * (C_used - 1) // 2
    is_variant_true = (len(np.unique(y_true)) > 1)

    out = {
        "contexts": contexts,
        "y_true": y_true,
        "y_pred": y_pred,
        "pairwise_precision": pair["pairwise_precision"],
        "pairwise_recall":    pair["pairwise_recall"],
        "pairwise_f1":        pair["pairwise_f1"],
        "ari": ari,
        "n_pairs": int(n_pairs),
        "is_variant_true": bool(is_variant_true),
    }
    if is_ami:
        out["ami"] = ami_value
    else:
        out["nmi_fallback"] = ami_value
    return out


def _agg_mean(vals):
    return float(np.mean(vals)) if len(vals) else float("nan")

def _agg_weighted(vals, weights):
    vals = np.asarray(vals, float)
    weights = np.asarray(weights, float)
    wsum = weights.sum()
    return float((vals * weights).sum() / wsum) if wsum > 0 else float("nan")

def _pick_ami(met):
    return met.get("ami", met.get("nmi_fallback", float("nan")))

# %% Misc utils
def is_insignificant(gain, alpha=0.05):
    if gain < 0: return True
    if gain > 100: return False
    return gain < 0 or 2 ** (-gain) > alpha

def cantor_pairing(x, y):
    return int((x + y) * (x + y + 1) / 2 + y)

def data_scale(y):
    scaler = preprocessing.StandardScaler().fit(y)
    return (scaler.transform(y))


def augmented_graph_to_lmg(gg : causallearn.graph.GeneralGraph.GeneralGraph) -> graph_lmg.LabelledMixedGraph:
    """ Converts a causallearn GeneralGraph to a LabelledMixedGraph if the GG contains an additional context node

    :param gg: a CausalGraph object, where gg.graph[j,i]=1 and gg.graph[i,j]=-1 indicates  i --> j ,
                    gg.graph[i,j] = gg.graph[j,i] = -1 indicates i --- j,
                    gg.graph[i,j] = G.graph[j,i] = 1 indicates i <-> j.
    """

    lmg = graph_lmg.LabelledMixedGraph(nodes=set(range(len(gg.graph)-1)))
    for i in  range(len(gg.graph)-1):
        for j in range(len(gg.graph)-1):
            if gg.graph[j][i] == 1 and gg.graph[i][j] == -1:  # causal
                lmg.add_directed(i,j)
            elif gg.graph[j][i] == -1 and gg.graph[i][j] == -1:  # undirected
                lmg.add_undirected(i,j)
    return lmg

# %% Graph conversions: nx.directed (w only directed edges), np.array (w only directed edges), GeneralGraph (causallearn), and LabelledMixedGraph (sepdistances)
#def general_graph_to_lmg(gg : causallearn.graph.GeneralGraph.GeneralGraph) -> graph_lmg.LabelledMixedGraph:
#    """ Converts a causallearn GeneralGraph to a LabelledMixedGraph#
#
#    :param gg: a CausalGraph object, where gg.graph[j,i]=1 and gg.graph[i,j]=-1 indicates  i --> j ,
#                    gg.graph[i,j] = gg.graph[j,i] = -1 indicates i --- j,
#                    gg.graph[i,j] = G.graph[j,i] = 1 indicates i <-> j.
#    """#
#
#    lmg = graph_lmg.LabelledMixedGraph(nodes=set(range(len(gg.graph))))
#    for i in  range(len(gg.graph)):
#        for j in range(len(gg.graph)):
#            if gg.graph[j][i] == 1 and gg.graph[i][j] == -1:  # causal
#                lmg.add_directed(i,j)
#            elif gg.graph[j][i] == -1 and gg.graph[i][j] == -1:  # undirected
#                lmg.add_undirected(i,j)
#    return lmg


def general_graph_to_lmg(gg, add_bidirected=False) -> graph_lmg.LabelledMixedGraph:
    """ Converts a causallearn GeneralGraph to a LabelledMixedGraph if the GG contains an additional context node

    :param gg: a CausalGraph object, where gg.graph[j,i]=1 and gg.graph[i,j]=-1 indicates  i --> j ,
                    gg.graph[i,j] = gg.graph[j,i] = -1 indicates i --- j,
                    gg.graph[i,j] = G.graph[j,i] = 1 indicates i <-> j.
    Encoding (M = gg.graph):
      - i → j        : M[j,i] =  1 and M[i,j] = -1
      - i — j        : M[i,j] = -1 and M[j,i] = -1
      - i ↔ j        : M[i,j] =  1 and M[j,i] =  1
      - i o→ j       : M[j,i] =  1 and M[i,j] =  2
      - j o→ i       : M[i,j] =  1 and M[j,i] =  2
      - i o–o j      : M[i,j] =  2 and M[j,i] =  2
    """
    M = np.asarray(gg.graph)
    n = M.shape[0]
    lmg = graph_lmg.LabelledMixedGraph(nodes=set(range(n)))

    for i in range(n):
        for j in range(i+1, n):
            a = M[i, j]
            b = M[j, i]

            # fully directed
            if b == 1 and a == -1:      # i -> j
                lmg.add_directed(i, j)
                continue
            if a == 1 and b == -1:      # j -> i
                lmg.add_directed(j, i)
                continue

            # undirected
            if a == -1 and b == -1:     # i - j
                lmg.add_undirected(i, j)
                continue

            if add_bidirected:
                # bidirected
                if a == 1 and b == 1:       # i <-> j
                    lmg.add_bidirected(i, j)
                    continue

                # semidirected (circle-tail)
                if b == 1 and a == 2:       # i o-> j
                    lmg.add_semidirected(i, j)
                    continue
                if a == 1 and b == 2:       # j o-> i
                    lmg.add_semidirected(j, i)
                    continue

                # circle-circle: treat as undirected
                if a == 2 and b == 2:       # i o-o j
                    lmg.add_undirected(i, j)
                    continue # else: no edge
    return lmg


def nxdigraph_to_lmg(nxg):
    lmg = graph_lmg.LabelledMixedGraph(nodes=nxg.nodes)
    for (i, j) in nxg.edges: lmg.add_directed(i,j)
    return lmg

def nxgraph_to_lmg(nxg):
    lmg = graph_lmg.LabelledMixedGraph(nodes=nxg.nodes)
    for (i, j) in nxg.edges:
        if not (j, i) in nxg.edges: lmg.add_directed(i,j)
        else: lmg.add_bidirected(i,j)
    return lmg

def causaldag_to_lmg(cdag: causaldag.DAG):
    #cpdag: causaldag.PDAG = cdag.cpdag()
    amat, node_list = cdag.to_amat()#cpdag.to_amat()
    lmg = graph_lmg.LabelledMixedGraph(nodes=set(range(len(amat))))
    for i in  range(len(amat)):
        for j in range(len(amat)):
            if amat[j][i] == 1 and not amat[i][j] == 1:  # causal
                lmg.add_directed(i,j)
            elif amat[j][i] == 1 and amat[i][j] == 1:
                lmg.add_undirected(i,j)
    return lmg
def lmg_to_directed_edge_adj(lmg: graph_lmg.LabelledMixedGraph) -> np.ndarray:
    adj = np.zeros((len(lmg.nodes), len(lmg.nodes)))
    for (ni, nmi), (nj, nmj) in itertools.combinations(enumerate(lmg.node_list), 2):
        if (nmi, nmj) in lmg.directed:
            adj[ni, nj] = 1
        if (nmj, nmi) in lmg.directed:
            adj[nj, ni] = 1
    return adj

def lmg_to_undirected_nx(lmg: graph_lmg.LabelledMixedGraph) -> nx.Graph:
    adj = np.zeros((len(lmg.nodes), len(lmg.nodes)))
    for (ni, nmi), (nj, nmj) in itertools.combinations(enumerate(lmg.node_list), 2):
        if (nmi, nmj) in lmg.directed:
            adj[ni, nj] = 1
        elif (nmj, nmi) in lmg.directed:
            adj[nj, ni] = 1
        elif  frozenset({nmi, nmj})  in lmg.undirected:
            adj[ni, nj] = 1
    G = nx.Graph(adj)
    return G

def general_graph_to_directed_edge_adj(gg):
    adj = np.zeros((len(gg.nodes), len(gg.nodes)))
    for i in  range(len(gg.graph)):
        for j in range(len(gg.graph)):
            if gg.graph[j][i] == 1 and gg.graph[i][j] == -1:  # causal
                adj[i, j] = 1
    return adj
def general_graph_to_undirected_edge_adj(gg):
    adj = np.zeros((len(gg.nodes), len(gg.nodes)))
    for i in  range(len(gg.graph)):
        for j in range(len(gg.graph)):
            if gg.graph[j][i] == 1 and gg.graph[i][j] == -1 or gg.graph[j][i] == -1 and gg.graph[i][j] == -1:
                adj[i, j] = 1
    return adj