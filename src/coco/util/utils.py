import itertools
import math
import warnings
import numpy as np
from sklearn import preprocessing


def logg(x):
    if x == 0:
        return 0
    return math.log(x)


def divv(x, y):
    if y == 0:
        return 0
    return x/y


def f1_score(tp, fp, fn):
    if tp + fp + fn == 0 :
        return 1
    return divv(tp, tp + 1 / 2 * (fp + fn))


def fpr(fp, tn):
    return divv(fp, fp+tn)


def tpr(tp, fn):
    return divv(tp, tp+fn)


# Naming and Formats for representing partitions/clusterings of a set
def partition_to_vector(partition):
    assignment = dict()
    for k, L in enumerate(partition):
        for x in L:
            assignment[x] = k
    _, y = zip(*sorted(assignment.items()))
    return y


def partition_to_map(part):
    return partition_to_vector(part)  # pi_group_map(part) # or partition to vec


def map_to_partition(mp):
    num_groups = max(mp) + 1
    pi = [[] for _ in range(num_groups)]
    for i in range(len(mp)):
        pi[mp[i]].append(i)
    return pi


def map_to_shifts(mp):
    return [1 if x != y else 0 for k, x in enumerate(mp) for y in mp[k + 1:]]


def shifts_to_map(shifts, n_c):
    mp = [0 for _ in range(n_c)]
    for ci in range(n_c):
        cur_idx = mp[ci]
        # assign all pairs (ci, c2) without a mechanism shift to the same group
        for ind, (c1, c2) in enumerate(itertools.combinations(range(n_c), 2)):
            if c1 != ci:
                continue
            if shifts[ind] == 0:
                mp[c2] = cur_idx
            else:
                mp[c2] = cur_idx + 1
    return mp


def pval_to_map(pval_mat, alpha=0.05):
    n_c = pval_mat.shape[0]

    cur_idx = 0
    mp = [0 for _ in range(n_c)]
    n_pairs_changing = [0 for _ in range(n_c)]
    for c1 in range(n_c):
        for c2 in range(n_c):
            if (not (c1 == c2)) and pval_mat[c1][c2] < alpha:
                n_pairs_changing[c1] += 1
    contexts_reordered = np.argsort(n_pairs_changing)
    for i, c1 in enumerate(contexts_reordered):

        # find the largest group of contexts seen so far that c1 can be added to
        cur_idx = mp[c1]
        # assign all pairs (ci, c2) without a mechanism shift to the same group
        for j in range(i, len(contexts_reordered)):  # range(c1, n_c):
            c2 = contexts_reordered[j]
            if c2 == c1:
                continue
            if pval_mat[c1][c2] > alpha:
                mp[c2] = mp[c1]  # cur_idx
            else:
                mp[c2] = cur_idx + 1
        # cur_idx = cur_idx + 1
    return pi_decrease_naming(mp)


def pi_decrease_naming(map):
    nms = np.unique(map)
    renms = [i for i in range(len(nms))]

    nmap = [renms[np.min(np.where(nms == elem))] for elem in map]

    assert (len(np.unique(nmap)) == len(np.unique(map)))
    return nmap


### Combination Operators on Partitions
def pi_join(map_1, map_2):
    # All mechanism changes in map_1 and map_2
    map = [map_1[ci] + map_2[ci] * (max(map_1) + 1) for ci in range(len(map_1))]
    return pi_decrease_naming(map)


def confound_partition(map:tuple, map_confounder:tuple, N:int):
    assert(len(map) == len(map_confounder))
    assert(len(map) == N)
    map_confounded = pi_join(map, map_confounder)
    assert(len(map_confounded) == N)
    assert(max(map_confounded) < len(map_confounded))

    return map_confounded


### From mechanism shifts between context pairs (that do not necessarily agree in practice) to partitions
def shifts_to_singleton_map(shifts, n_c):
    mp = [0 for _ in range(n_c)]
    c0 = 0
    cur_idx = 0
    for ind, (c1, c2) in enumerate(itertools.combinations(range(n_c), 2)):
            ci = c0
            if c1 == c0:
                ci = c2
            elif c2 == c0:
                ci = c1
            else:
                continue
                # assign all contexts different from c1 to their own new group
            if shifts[ind] == 0:
                mp[ci] = 0
            else:
                mp[ci] = cur_idx + 1
                cur_idx = cur_idx + 1
    return mp


# and map_to_partition(shifts_to_map(map_to_shifts(partition_to_map([[0],[1],[2],[3]])), 4))
# and when not starting from an agreeing partition

# Disagreement/ transitivity condition: if C1==C2 and C1!=C3 but also C2==C3, or if C1==C2 and C2!=C3 but also C1==C3
def shifts_resolve_disagreements(shifts, n_c):
    new_shifts = shifts.copy()
    for ind, (c1, c2) in enumerate(itertools.combinations(range(n_c), 2)):
        if shifts[ind] == 1:
            continue

        # For each third context, find the pair (c1;c3)
        for sub1_ind, (ca, cb) in enumerate(itertools.combinations(range(n_c), 2)):
            if not ((ca == c1) or (cb == c1)):
                continue
            c3 = ca
            if ca == c1:
                c3 = cb

            # Find the pair (c2;c3)
            for sub2_ind, (cc, cd) in enumerate(itertools.combinations(range(n_c), 2)):

                is_c2_c3 = ((cc == c2) and (cd == c3)) or ((cd == c2) and (cc == c3))
                disagrees = (shifts[sub1_ind] == 1 and shifts[sub2_ind] == 0) \
                            or (shifts[sub1_ind] == 0 and shifts[sub2_ind] == 1)

                if is_c2_c3 and disagrees:
                    new_shifts[sub1_ind] = 1
                    new_shifts[sub2_ind] = 1
    return new_shifts


def G_to_adj(G, G_true, n_c_vars):
    index = [i for i in G.nodes]
    index_true = [i for i in G_true.nodes]

    A = np.zeros((len(index), len(index)))
    A_gaps = np.full((len(index)+n_c_vars, len(index)+n_c_vars), True)
    A_edges = np.full((len(index)+n_c_vars, len(index)+n_c_vars), False)
    A_true = np.zeros((len(index_true), len(index_true)))

    for i in G_true.nodes:
        ind_i_true = index_true.index(i)
        for j in G_true.nodes:
            ind_j_true = index_true.index(j)
            if (i,j) in G_true.edges():
                A_true[ind_i_true][ind_j_true] = 1

            if i in G.nodes and j in G.nodes:
                ind_i = index.index(i)
                ind_j = index.index(j)
                if (i,j) in G.edges():
                    A[ind_i][ind_j] = 1
                    A_gaps[ind_i][ind_j] = False
                    A_edges[ind_i][ind_j] = True
                    A_gaps[ind_j][ind_i] = False
                    A_edges[ind_j][ind_i] = True
    return A, A_gaps, A_edges, A_true


def data_to_jci(D, seed, n_confounders):
    n_nodes = D.shape[2]
    n_context_vars = D.shape[0]-1
    n_contexts = D.shape[0]
    n_samp = D.shape[1]

    Djci = np.zeros((n_samp * n_contexts, n_nodes + n_context_vars))
    Dpooled = np.zeros((n_samp * n_contexts, n_nodes))
    offset = 0
    for ci in range(n_contexts):
        rows = range(offset, n_samp * (ci + 1))
        offset += n_samp
        #Djci[rows, 0:n_nodes] = D[ci]

        Djci[rows, 0:n_nodes] = D[ci]
        Dpooled[rows, 0:n_nodes] = D[ci]
        if ci > 0:
            Djci[rows, [n_nodes + ci - 1]] = [1 for _ in range(n_samp)]

    json_args = {}
    json_args["SystemVars"] = [i for i in range(1, D.shape[2]+ 1)]
    json_args["ContextVars"] = [i for i in range(D.shape[2]+ 1,D.shape[2]+ D.shape[0] +1-1 )]
    json_args["basefilename"] = ''
    json_args["pSysObs"] = [n_nodes]
    json_args["pSysConf"] = [n_confounders]
    json_args["pContext"] = [n_contexts]
    json_args["N"] = [len(Djci)]
    json_args["sufficient"] = [0]
    json_args["acyclic"] = [1]
    json_args["Itargets"] = []
    json_args["targets"] = []
    json_args["seed"] =  seed

    return Djci, Dpooled, json_args


def data_scale(X):
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X)


def data_sub(Dc, node_set):
    return np.array(
        [Dc[c_i][:, node_set] for c_i in range(len(Dc))])  # np.array([Dc[c_i][k, :] for c_i in range(len(Dc))])


def data_group(Xc, yc, partition):
    """ Given a partition such as [[0,1,2],[3,4]], pool data within each group """
    n_pi = len(partition)
    n_c = Xc.shape[0]

    assert ((len(Xc.shape) == 3) & (len(yc.shape) == 2))
    assert ((yc.shape[0] == n_c) & (yc.shape[1] == Xc.shape[1]))

    Xpi = [np.concatenate([Xc[c_i] for c_i in partition[pi_k]]) for pi_k in range(n_pi)]
    ypi = [np.concatenate([yc[c_i] for c_i in partition[pi_k]]) for pi_k in range(n_pi)]

    return Xpi, ypi


def data_check(X, allow_nan=False, allow_inf=False):
    assert (len(X.shape) == 3)
    for c_i in range(len(X)):
        assert allow_nan or not np.isnan(X[c_i]).any(), "Data contains NaN."
        assert allow_inf or not np.isinf(X[c_i]).any(), "Data contains Inf."
        if X[c_i].shape[0] < X[c_i].shape[1]:
            warnings.warn("Num variables greater than num samples, p>n.")


def data_check_soft(X, allow_nan=False, allow_inf=False):
    check = True
    assert (len(X.shape) == 3)
    for c_i in range(len(X)):
        check = check and (allow_nan or not np.isnan(X[c_i]).any())
        check = check and (allow_inf or not np.isinf(X[c_i]).any())
        if X[c_i].shape[0] < X[c_i].shape[1]:
            warnings.warn("Num variables greater than num samples, p>n.")
    return check
