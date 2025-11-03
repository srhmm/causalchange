import statistics

import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.metrics.cluster import expected_mutual_information
from sklearn.metrics.cluster._supervised import check_clusterings, contingency_matrix, mutual_info_score, entropy, \
    _generalized_average


#%% partitions/assignments
def pi_join(map_1, map_2):
    # All mechanism changes in map_1 and map_2
    map = [map_1[ci] + map_2[ci] * (max(map_1) + 1) for ci in range(len(map_1))]
    return pi_decrease_naming(map)


def pi_decrease_naming(map):
    nms = np.unique(map)
    renms = [i for i in range(len(nms))]
    nmap = [renms[np.min(np.where(nms == elem))] for elem in map]
    assert (len(np.unique(nmap)) == len(np.unique(map)))
    return nmap

def pi_xor_pessimistic(idl_1, idl_2):
    assert len(idl_1) == len(idl_2)
    idl_ave = [cantor_pairing(idl_1[i], idl_2[i]) for i in range(len(idl_1))]
    idl_ave = pi_decrease_naming(idl_ave)
    return idl_ave


def _get_true_idl(true_idls, parents, target, t_A):
    true_pa_i = [k for k in np.where(t_A[:, target] != 0)[0]]
    true_Z_i = true_idls[target]
    observed_Z_i = true_Z_i.copy()
    for parent in true_pa_i:
        if parent in parents: continue
        observed_Z_i = pi_join(observed_Z_i, true_idls[parent])
    return observed_Z_i

def _get_true_idl_Z(pa_i, node_i,  t_A, t_Z, t_n_Z, n_samp):
    true_pa_i = [k for k in np.where(t_A[:, node_i] != 0)[0]]

    # Changes of node_i as combination of all confounders
    true_Z_i = np.zeros(n_samp)
    for zi, nodeset in enumerate(t_n_Z):
        if node_i in nodeset:
            true_Z_i = pi_join(true_Z_i, t_Z[zi])
    observed_Z_i = true_Z_i.copy()
    # Observed changes of node_i as combination of all confounders and missing parents
    for node_j in true_pa_i:
        if node_j in pa_i:
            continue
        for zi, nodeset in enumerate(t_n_Z):
            if node_j in nodeset:
                observed_Z_i = pi_join(observed_Z_i, t_Z[zi])
    return observed_Z_i

#%% check usage
def _get_true_idl_Z_graph(adjacency, t_A, t_Z, t_n_Z, X):
    true_Z = np.zeros(X.shape[0])
    observed_Z = np.zeros(X.shape[0])
    for node_i in range(len(adjacency)):
        pa_i = [k for k in np.where(adjacency[:, node_i] != 0)[0]]
        true_pa_i = [k for k in np.where(t_A[:, node_i] != 0)[0]]

        # true changes
        true_Z_i = np.zeros(X.shape[0])
        for zi, nodeset in enumerate(t_n_Z):
            if node_i in nodeset:
                true_Z_i = pi_join(true_Z, t_Z[zi])
        true_Z = pi_join(true_Z_i, true_Z)

        # missing parents
        observed_Z_i = np.zeros(X.shape[0])
        observed_Z_i = pi_join(observed_Z_i, true_Z_i)
        for node_j in true_pa_i:
            if node_j in pa_i:
                continue
            for zi, nodeset in enumerate(t_n_Z):
                if node_j in nodeset:
                    observed_Z_i = pi_join(observed_Z_i, t_Z[zi])

        observed_Z = pi_join(observed_Z, observed_Z_i)
    return observed_Z

def _get_true_idl_graph(adjacency, true_idls,  t_A, X):
    true_Z = np.zeros(X.shape[0])
    observed_Z = np.zeros(X.shape[0])

    for node_i in range(len(adjacency)):
        parents = [k for k in np.where(adjacency[:, node_i] != 0)[0]]
        true_pa_i = [k for k in np.where(t_A[:, node_i] != 0)[0]]

        true_Z_i = true_idls[node_i]
        observed_Z_i = true_Z_i.copy()
        true_Z = pi_join(true_Z, true_Z_i)
        for parent in true_pa_i:
            if parent in parents:
                continue
            observed_Z_i = pi_join(observed_Z_i, true_idls[parent])
        observed_Z = pi_join(observed_Z, observed_Z_i)
    return observed_Z


#%% Graphs
def to_confounded_adjacency(A: np.array, n_Z):
    e_A = A.copy()

    import itertools as itt
    for n_set in n_Z:
        for (i, j) in itt.combinations(n_set, 2):
            e_A[i][j] = 1
            e_A[j][i] = 1
    return e_A


### Misc
def is_insignificant(gain, alpha=0.05):
    return gain < 0 or 2 ** (-gain) > alpha


def cantor_pairing(x, y):
    return int((x + y) * (x + y + 1) / 2 + y)


def data_scale(y):
    scaler = preprocessing.StandardScaler().fit(y)
    return (scaler.transform(y))


#%% MI
def expected_sampled_mutual_info_score(Za, Zb):
    ka, kb = len(np.unique(Za)), len(np.unique(Zb))
    mis = [0 for _ in range(30)]
    for ii, it in enumerate(range(30)):
        Za_samp = np.random.choice(ka, size=len(Za))
        Zb_samp = np.random.choice(kb, size=len(Zb))
        mis[ii] = sklearn.metrics.mutual_info_score(Za_samp, Zb_samp)
    emi = statistics.mean(mis)
    #print("sample EMI: ",emi, "comp. to", expected_adjusted_mutual_info_score(Za, Zb)[0])
    return emi

def expected_sampled_adjusted_mutual_info_score(Za, Zb):
    ka, kb = len(np.unique(Za)), len(np.unique(Zb))
    mis = [0 for _ in range(30)]
    for ii, it in enumerate(range(30)):
        Za_samp = np.random.choice(ka, size=len(Za))
        Zb_samp = np.random.choice(kb, size=len(Zb))
        mis[ii] = sklearn.metrics.adjusted_mutual_info_score(Za_samp, Zb_samp)
    emi = statistics.mean(mis)
    #print("sample EMI: ",emi, "comp. to", expected_adjusted_mutual_info_score(Za, Zb)[0])
    return emi

def exp_mutual_info_score(
        labels_true, labels_pred, *, average_method="arithmetic"
):
    """Adjusted Mutual Information between two clusterings.

    Adjusted Mutual Information (AMI) is an adjustment of the Mutual
    Information (MI) score to account for chance. It accounts for the fact that
    the MI is generally higher for two clusterings with a larger number of
    clusters, regardless of whether there is actually more information shared.
    For two clusterings :math:`U` and :math:`V`, the AMI is given as::

        AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching :math:`U` (``label_true``)
    with :math:`V` (``labels_pred``) will return the same score value. This can
    be useful to measure the agreement of two independent label assignments
    strategies on the same dataset when the real ground truth is not known.

    Be mindful that this function is an order of magnitude slower than other
    metrics, such as the Adjusted Rand Index.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.

    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.

    average_method : {'min', 'geometric', 'arithmetic', 'max'}, default='arithmetic'
        How to compute the normalizer in the denominator.

        .. versionadded:: 0.20

        .. versionchanged:: 0.22
           The default value of ``average_method`` changed from 'max' to
           'arithmetic'.

    Returns
    -------
    ami: float (upperlimited by 1.0)
       The AMI returns a value of 1 when the two partitions are identical
       (ie perfectly matched). Random partitions (independent labellings) have
       an expected AMI around 0 on average hence can be negative. The value is
       in adjusted nats (based on the natural logarithm).

    See Also
    --------
    adjusted_rand_score : Adjusted Rand Index.
    mutual_info_score : Mutual Information (not adjusted for chance).

    References
    ----------
    .. [1] `Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for
       Clusterings Comparison: Variants, Properties, Normalization and
       Correction for Chance, JMLR
       <http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf>`_

    .. [2] `Wikipedia entry for the Adjusted Mutual Information
       <https://en.wikipedia.org/wiki/Adjusted_Mutual_Information>`_

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import adjusted_mutual_info_score
      >>> adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      ... # doctest: +SKIP
      1.0
      >>> adjusted_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      ... # doctest: +SKIP
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the AMI is null::

      >>> adjusted_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      ... # doctest: +SKIP
      0.0
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if (
            classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0, 1.0

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    # Calculate the expected value for the mutual information
    emi = expected_mutual_information(contingency, n_samples)
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    normalizer = _generalized_average(h_true, h_pred, average_method)
    denominator = normalizer - emi
    # Avoid 0.0 / 0.0 when expectation equals maximum, i.e. a perfect match.
    # normalizer should always be >= emi, but because of floating-point
    # representation, sometimes emi is slightly larger. Correct this
    # by preserving the sign.
    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        denominator = max(denominator, np.finfo("float64").eps)
    ami = (mi - emi) / denominator
    return emi, mi


#### OTHER


import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.mixture import GaussianMixture

def noisy_labels(Z, num_classes, confidence=1.0):
    N = len(Z)
    base_probs = np.eye(num_classes)[Z]
    noisy = base_probs * confidence + (1 - confidence) / num_classes
    return noisy / noisy.sum(axis=1, keepdims=True)

#%%
def sample_noisy_labels(N, k, confidence):
    Z = np.random.choice(k, size=N)
    return noisy_labels(Z, k, confidence)
def estimate_confidence(p):
    return np.mean(np.max(p, axis=1))

def expected_smi(pa, pb, N, trials=30):
    ka = pa.shape[1]
    kb = pb.shape[1]
    ca = estimate_confidence(pa)
    cb = estimate_confidence(pb)
    smis = []
    for _ in range(trials):
        pa_i = sample_noisy_labels(N, ka, ca)
        pb_i = sample_noisy_labels(N, kb, cb)
        smi = soft_mutual_information(pa_i, pb_i)
        smis.append(smi)
    return np.mean(smis), np.std(smis)

def se(pproba, eps=1e-12): #mean entropy
    clip_pproba = np.clip(pproba, eps, 1.0)
    return np.mean(-np.sum(clip_pproba * np.log(clip_pproba), axis=1))

def soft_mutual_information(P_probs, Q_probs, eps=1e-12):
    N = P_probs.shape[0]

    joint = np.einsum('ni,nj->ij', P_probs, Q_probs) / N
    P_marg = joint.sum(axis=1, keepdims=True)
    Q_marg = joint.sum(axis=0, keepdims=True)

    # Prevent log(0)
    joint_safe = np.maximum(joint, eps)
    P_marg_safe = np.maximum(P_marg, eps)
    Q_marg_safe = np.maximum(Q_marg, eps)
    MI = np.sum(joint_safe * np.log(joint_safe / (P_marg_safe * Q_marg_safe)))

    if P_probs.shape[1] == 1 and Q_probs.shape[1] == 1:
        return 1.0

    return MI


def soft_adjusted_mutual_information(pa, pb, N, trials=50):
    ka = pa.shape[1]
    kb = pb.shape[1]
    ca = estimate_confidence(pa)
    cb = estimate_confidence(pb)
    smi = soft_mutual_information(pa, pb)
    smi_exp, _ = expected_smi(pa, pb, N, trials=trials)
    Ha = -np.sum(pa * np.log(pa + 1e-12)) / pa.shape[0]
    Hb = -np.sum(pb * np.log(pb + 1e-12)) / pb.shape[0]
    denom = max(Ha, Hb) - smi_exp
    return (smi - smi_exp) / denom if denom > 0 else 0.0