import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import jaccard_score

#from mixtures.old.causal_mixture import EdgeState
from src.causalchange.util.utils_idl import pi_xor_pessimistic


def precision(s0, s1):
    s = set(s0).intersection(s1)
    return 1 if not len(s1) else len(s) / len(s1)


def recall(s0, s1):
    s = set(s0).intersection(s1)
    return 1 if not len(s0) else len(s) / len(s0)


def preci_recall(s1, s2):
    return precision(s1, s2), recall(s1, s2)

def rates( tp, tn, fp, fn):
    pp, nn = tp + fn, tn + fp
    tpr_power_recall = 1 if pp == 0 else tp / pp
    fnr_type2 = 1 - tpr_power_recall
    fpr_type1 = 0 if nn == 0 else fp / nn
    tnr_specificity = 1 - fpr_type1
    return tpr_power_recall, fpr_type1, tnr_specificity, fnr_type2

def matthews_correlation(tp, tn, fp, fn):
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return  1 if denom==0 else ((tp*tn)-(fp*fn))/denom
def eval_confounders(true_conf_ix_set, conf_ix_set):
    true_conf_ix = {node for nset in true_conf_ix_set for node in nset}
    conf_ix = {node for nset in conf_ix_set for node in nset}

    prec_cf, rec_cf = preci_recall(true_conf_ix, conf_ix)
    f1_cf = 1 if prec_cf + rec_cf == 0 else 2 * ((prec_cf * rec_cf) / (prec_cf + rec_cf))
    overlap_score = set_overlap_score(true_conf_ix_set, conf_ix_set)
    grouping_score = grouping_consistency(true_conf_ix_set, conf_ix_set)

    return {
        'prec-cf': prec_cf,
        'rec-cf': rec_cf,
        'f1-cf': f1_cf,
        'ovl-cfs': overlap_score,
        'grp-cfs': grouping_score
    }


# Topological Ordering Divergence (as in SCORE paper, Rolland et al.)

def positives_negatives(A_true, A):
    tp = sum([sum([1 if (A[i][j] != 0 and A_true[i][j] != 0) else 0
                   for j in range(len(A_true[i]))]) for i in range(len(A_true))])
    tn = sum([sum([1 if (A[i][j] == 0 and A_true[i][j] == 0) else 0
                   for j in range(len(A_true[i]))]) for i in range(len(A_true))])
    fp = sum([sum([1 if (A[i][j] != 0 and A_true[i][j] == 0) else 0
                   for j in range(len(A_true[i]))]) for i in range(len(A_true))])
    fn = sum([sum([1 if (A[i][j] == 0 and A_true[i][j] != 0) else 0
                   for j in range(len(A_true[i]))]) for i in range(len(A_true))])
    return  tp, tn, fp, fn

def dtop_measure(A_true, A, pred_order):
    dtop = sum([sum([A_true[pa][i] for pa in range(len(A_true))
                     if pred_order.index(pa) > pred_order.index(i)])
                for i in range(len(A_true))])
    tp, tn, fp, fn = positives_negatives(A_true, A)

    n_edges = tp + fn
    dtop_norm = 0 if n_edges == 0 else dtop / n_edges
    return dtop, dtop_norm


def kendalls_tau(pred_order, true_order):
    n = len(pred_order)
    pred_idx = {node: i for i, node in enumerate(pred_order)}
    true_idx = {node: i for i, node in enumerate(true_order)}

    concordant, discordant = 0, 0
    for (u, v) in permutations(pred_order, 2):
        pred_sign = np.sign(pred_idx[u] - pred_idx[v])
        true_sign = np.sign(true_idx[u] - true_idx[v])
        if pred_sign == true_sign:
            concordant += 1
        else:
            discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def pairwise_accuracy(pred_order, true_order, true_dag):
    pred_idx = {node: i for i, node in enumerate(pred_order)}
    correct = 0
    total = 0

    for u, v in true_dag.edges:
        total += 1
        if pred_idx[u] < pred_idx[v]:  # Correctly ordered
            correct += 1

    return correct / total


def topological_sort_distance(pred_order, true_order):
    # distance between predicted and true topological orders (L1 norm of ranks).
    true_idx = {node: i for i, node in enumerate(true_order)}
    pred_idx = {node: i for i, node in enumerate(pred_order)}

    return sum(abs(pred_idx[node] - true_idx[node]) for node in true_order)


def normalized_topological_sort_distance(pred_order, true_order):
    n = len(true_order)
    max_distance = n * (n - 1) // 2  # Maximum distance for reverse order
    raw_distance = topological_sort_distance(pred_order, true_order)
    return 1 - raw_distance / max_distance


def node_level_rank_error(pred_order, true_order):
    # compute node-level rank errors (per-node absolute rank difference).
    true_idx = {node: i for i, node in enumerate(true_order)}
    pred_idx = {node: i for i, node in enumerate(pred_order)}

    return {node: abs(pred_idx[node] - true_idx[node]) for node in true_order}


def eval_graph(A_true: np.array, A_estim: np.array, sid_call: bool) -> dict:
    """ eval causal graph over observed nodes

    :param A_true: true DAG, only truly causal edges, no confounding
    :param A_estim: estimated PAG, causal edge i->j if A[i][j]!=0, v[j][i]==0; indep or bidirected otherwise
    :param sid_call: R call
    :return:
    """
    A_directed = np.zeros(A_estim.shape)
    A_true_one = np.zeros(A_estim.shape)
    NN = len(A_directed)

    # remove undirected/bidirected edges
    for i, j in set(product(set(range(NN)), set(range(NN)))):
        if A_estim[i][j] != 0 and not A_estim[j][i] != 0:
            A_directed[i][j] = A_estim[i][j]
        if A_true[i][j] != 0:
            A_true_one[i][j] = 1

    G_true = nx.from_numpy_array(A_true, create_using=nx.DiGraph)
    G = nx.from_numpy_array(A_directed, create_using=nx.DiGraph)
    e0 = G_true.edges()
    e1 = G.edges()
    prec, rec = preci_recall(e0, e1)  # precision_recall(e1, e0)
    f1 = 0 if prec + rec == 0 else 2 * ((prec * rec) / (prec + rec))
    from cdt.metrics import SID, SHD

    sid = -1 if not sid_call else SID(G, G_true)
    sid_norm = -1 if not sid_call else sid / (NN * (NN - 1))
    shd = SHD(G, G_true)
    shd_norm = shd / (NN ** 2)

    import cdt
    aupr, curve = cdt.metrics.precision_recall(A_true_one, A_directed)

    # remove cycles to eval top. orderings
    try:
        while True:
            cycle = nx.find_cycle(G)
            e = cycle.pop()
            G.remove_edge(*e)
    except nx.NetworkXNoCycle:
        pass
    try:
        while True:
            cycle = nx.find_cycle(G_true)
            e = cycle.pop()
            G_true.remove_edge(*e)
    except nx.NetworkXNoCycle:
        pass
    pred_order, true_order = list(nx.topological_sort(G)), list(nx.topological_sort(G_true))

    kt = kendalls_tau(pred_order, true_order)
    ktn = (kt + 1) / 2
    tpa = pairwise_accuracy(pred_order, true_order, G_true)
    tsd = normalized_topological_sort_distance(pred_order, true_order)

    tp, tn, fp, fn = positives_negatives(A_true, A_directed)
    dtop, dtop_norm, = dtop_measure(A_true, A_directed, pred_order)

    dtn = 1 - dtop_norm
    top_ave = (ktn + dtn) / 2

    tpr, fpr, tnr, fnr = rates(tp, tn, fp, fn)
    mcc = matthews_correlation(tp, tn, fp, fn)

    return {
        'prec-g': prec,
        'rec-g': rec,
        'f1-g': f1,
        'mcc-g': mcc,
        'tp-g': tp,
        'fp-g': fp,
        'tn-g': tn,
        'fn-g': fn,
        'tpr-g': tpr,
        'fpr-g': fpr,
        'tnr-g': tnr,
        'fnr-g': fnr,
        'shd-g': shd,
        'sid-g': sid,
        'shdnm-g': shd_norm,
        'sidnm-g': sid_norm,
        'top-dtop': dtn,
        'top-ave': top_ave,
        'aupr-g': aupr
    }


def set_overlap_score(true_conf_ix_set, conf_ix_set):
    """Calculate an overlap score based on Jaccard index between each true set and each predicted set."""
    overlap_scores = []

    for true_set in true_conf_ix_set:
        best_score = 0
        for pred_set in conf_ix_set:
            # Jaccard similarity
            intersection = len(set(true_set) & set(pred_set))
            union = len(set(true_set) | set(pred_set))
            score = intersection / union if union > 0 else 0
            best_score = max(best_score, score)

        overlap_scores.append(best_score)

    # avg overlap score across all true sets
    avg_overlap_score = np.mean(overlap_scores)
    return avg_overlap_score


def grouping_consistency(true_conf_ix_set, conf_ix_set):
    """Evaluate grouping consistency using the Jaccard Index for node groupings."""
    all_nodes = sorted(set(node for group in true_conf_ix_set + conf_ix_set for node in group))
    node_to_index = {node: i for i, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    true_labels = np.full(num_nodes, -1, dtype=int)
    pred_labels = np.full(num_nodes, -1, dtype=int)

    for label, group in enumerate(true_conf_ix_set):
        for node in group:
            true_labels[node_to_index[node]] = label

    for label, group in enumerate(conf_ix_set):
        for node in group:
            pred_labels[node_to_index[node]] = label

    grouping_score = jaccard_score(true_labels == -1, pred_labels == -1, average='micro')
    return grouping_score


def match_node_sets(true_sets, pred_sets):
    """Match true and predicted sets based on Jaccard overlap"""
    n, m = len(true_sets), len(pred_sets)
    overlap_matrix = np.zeros((n, m))

    for i, true_set in enumerate(true_sets):
        for j, pred_set in enumerate(pred_sets):
            intersection = len(set(true_set) & set(pred_set))
            union = len(set(true_set) | set(pred_set))
            overlap_matrix[i, j] = intersection / union if union > 0 else 0

    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)  # maximize overlap
    matches = list(zip(row_ind, col_ind))
    return matches, overlap_matrix


def eval_edge_states(true_confd_targets, true_A, edge_states):
    # F1 directed edge or not
    tp_dir, fp_dir, tn_dir, fn_dir = 0, 0, 0, 0
    # F1 causal or not
    tp_cdir, fp_cdir, tn_cdir, fn_cdir = 0, 0, 0, 0
    # F1 confounded pair or not (CONF label must be present, CAUS_CONF included)
    tp_cfd, fn_cfd, fp_cfd, tn_cfd = 0, 0, 0, 0
    # F1 indep or not (INDEP label counts)
    tp_indep, fn_indep, fp_indep, tn_indep = 0, 0, 0, 0
    # F1 indep-Z or not (CONF label must be present, CAUS_CONF wrong)
    tp_coo, fn_coo, fp_coo, tn_coo = 0, 0, 0, 0
    # F1 conf and caus (DEP_CONF label must be present, regardless the direction)
    tp_coca, fn_coca, fp_coca, tn_coca = 0, 0, 0, 0

    any_direction = [
        EdgeState.UNDIR, EdgeState.UNDIR_CONFD, EdgeState.CAUS_CONFD, EdgeState.CAUS, EdgeState.REV_CONFD,
        EdgeState.REV]
    any_confounding = [EdgeState.CONFD, EdgeState.CAUS_CONFD, EdgeState.REV_CONFD, EdgeState.UNDIR_CONFD]
    directed_confounding = [EdgeState.CAUS_CONFD, EdgeState.REV_CONFD, EdgeState.UNDIR_CONFD]
    for id_i_j in edge_states:
        (i, j, edge_state) = edge_states[id_i_j]
        has_any_edge = (true_A[i][j] != 0 or true_A[j][i] != 0)

        # directed or not
        if has_any_edge:
            tp_dir, fn_dir = (1, 0) if edge_state in any_direction else (0, 1)
        else:
            fp_dir, tn_dir = (1, 0) if edge_state in any_direction else (0, 1)

            # direction correct
        if true_A[i][j] != 0:
            tp_cdir, fn_cdir = (1, 0) if edge_state in [EdgeState.CAUS_CONFD, EdgeState.CAUS] else (0, 1)
        elif true_A[j][i] != 0:
            tp_cdir, fn_cdir = (1, 0) if edge_state in [EdgeState.REV_CONFD, EdgeState.REV] else (0, 1)
        else:
            fp_cdir, tn_cdir = (1, 0) if edge_state in [EdgeState.REV_CONFD, EdgeState.REV, EdgeState.CAUS_CONFD,
                                                        EdgeState.CAUS] else (0, 1)

        # confounded or not
        is_confd = any([i in st and j in st for st in true_confd_targets])
        if is_confd:
            tp_cfd, fn_cfd = (1, 0) if edge_state in any_confounding else (0, 1)
        else:
            fp_cfd, tn_cfd = (1, 0) if edge_state in any_confounding else (0, 1)

            # decision "only confounded" correct
        if is_confd and not has_any_edge:
            tp_coo, fn_coo = (1, 0) if edge_state in [EdgeState.CONFD] else (0, 1)
        else:
            fp_coo, tn_coo = (1, 0) if edge_state in [EdgeState.CONFD] else (0, 1)

            # decision not only confounded correct
        if is_confd and has_any_edge:
            tp_coca, fn_coca = (1, 0) if edge_state in directed_confounding else (0, 1)
        else:
            fp_coca, tn_coca = (1, 0) if edge_state in directed_confounding else (0, 1)

            # independent
        is_indep = true_A[i][j] == 0 and true_A[j][i] == 0 and not is_confd
        if is_indep:
            tp_indep, fn_indep = (1, 0) if edge_state in [EdgeState.INDEP] else (0, 1)
        else:
            fp_indep, tn_indep = (1, 0) if edge_state in [EdgeState.INDEP] else (0, 1)

    metrics = {}
    for (quadruplet, nm) in [
        ([tp_dir, tn_dir, fp_dir, fn_dir], "dir_pair"),
        ([tp_cdir, tn_cdir, fp_cdir, fn_cdir], "cdir_pair"),
        ([tp_cfd, tn_cfd, fp_cfd, fn_cfd], "cfd_pair"),
        ([tp_coo, tn_coo, fp_coo, fn_coo], "coo_pair"),
        ([tp_coca, tn_coca, fp_coca, fn_coca], "coca_pair"),
        ([tp_indep, tn_indep, fp_indep, fn_indep], "indep_pair"),
    ]:
        metrics[f"f1_{nm}"], metrics[f"tpr_{nm}"], metrics[f"fpr_{nm}"], metrics[f"fdr_{nm}"] = _f1(*quadruplet)

    return metrics


def _f1(tp, tn, fp, fn):
    den = tp + 1 / 2 * (fp + fn)
    f1 = 0 if (tp + tn + fp + fn == 0) else 1 if den == 0 else tp / den
    tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
    fpr = 0 if (tn + fp == 0) else fp / (tn + fp)
    fdr = 0 if (tp + fp == 0) else fp / (tp + fp)
    return f1, tpr, fpr, fdr


def eval_labels_per_node(true_sets, true_labels, pred_labels_per_node):
    n_nodes = len(pred_labels_per_node)
    ami_scores = []
    true_k_counts = []
    pred_k_counts = []
    err_k_counts = []

    for i in range(n_nodes):
        pred_label = pred_labels_per_node[i]
        true_label = np.zeros(len(pred_label))
        for nset, cf_label in zip(true_sets, true_labels):
            if i in nset:
                true_label = pi_xor_pessimistic(true_label, cf_label)

        ami = adjusted_mutual_info_score(true_label, pred_label)
        ami_scores.append(ami)
        print(ami)
        true_k_counts.append(len(set(true_label)))
        pred_k_counts.append(len(set(pred_label)))
        err_k_counts.append(abs(len(set(pred_label)) - len(set(true_label))))

    # Aggregate AMI scores
    avg_ami = np.mean(ami_scores)
    avg_k_err = np.mean(err_k_counts)

    # Evaluate number of categories
    true_k = sum(true_k_counts)
    pred_k = sum(pred_k_counts)
    exact_match_k = int(true_k == pred_k)
    rel_error_k = abs(true_k - pred_k) / true_k if true_k > 0 else float('inf')

    return {
        "avami-node-cfs": avg_ami,
        "avgerr-node-k": avg_k_err,
        "relerr-node-k": rel_error_k
    }


def eval_labels_global(true_labels, pred_label):
    if len(pred_label) == 0:
        ami, rel_error_k = 0, 0
    else:
        true_label = np.zeros(len(pred_label))

        for i, lbl in enumerate(true_labels):
            true_label = pi_xor_pessimistic(true_label, lbl)

        ami = adjusted_mutual_info_score(true_label, pred_label)

        true_k = len(np.unique(true_label))
        pred_k = len(np.unique(pred_label))
        exact_match_k = int(true_k == pred_k)
        rel_error_k = abs(true_k - pred_k) / true_k if true_k > 0 else float('inf')

    return {
        "ami-global-cfs": ami,
        "relerr-global-k": rel_error_k
    }

def eval_categorical_labels(true_sets, true_labels, pred_sets, pred_labels):
    matches, overlap_matrix = match_node_sets(true_sets, pred_sets)

    ami_scores = []
    true_k_counts = []
    pred_k_counts = []
    err_k_counts = []

    for i, j in matches:
        true_label = true_labels[i]
        pred_label = pred_labels[j]

        ami = adjusted_mutual_info_score(true_label, pred_label)
        ami_scores.append(ami)

        true_k_counts.append(len(set(true_label)))
        pred_k_counts.append(len(set(pred_label)))
        err_k_counts.append(abs(len(set(pred_label)) - len(set(true_label))))

    avg_ami = np.mean(ami_scores)
    avg_k_err = np.mean(err_k_counts)

    true_k = sum(true_k_counts)
    pred_k = sum(pred_k_counts)
    exact_match_k = int(true_k == pred_k)  # Binary: 1 if k_true == k_pred
    rel_error_k = abs(true_k - pred_k) / true_k if true_k > 0 else float('inf')
    if not len(
            ami_scores):  # for example, no confounded nodes found -> dont report metrics. todo -1 instead for tex plots?
        return {}

    return {
        "avami-cfs": avg_ami,
        "relerr-k": rel_error_k,
        "avgerr-k": avg_k_err
    }


import numpy as np
from itertools import permutations, product
