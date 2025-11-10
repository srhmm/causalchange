import math

import numpy as np
from scipy import sparse as sp
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import (
    contingency_matrix,
    entropy,
    expected_mutual_information,
)
from sklearn.metrics.cluster._supervised import check_clusterings, _generalized_average


def hard_to_soft_one_hot(labels):
    n_clusters = len(np.unique(labels))
    n_samples = len(labels)
    soft_labels = np.zeros((n_samples, n_clusters))
    soft_labels[np.arange(n_samples), labels] = 1
    return soft_labels


def mi_dependency_test(map_i, map_j):
    mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)
    dep = mi > emi and not math.isclose(mi, emi)
    return dep


def p_entropy(p_labels):
    """Compute the entropy of a distribution for given labels."""
    return -np.sum(p_labels * np.log(p_labels))


def m_entropy(labels):
    """Compute the entropy of a distribution for given labels."""
    value, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log(probabilities))


def joint_entropy(*labels):
    """Compute the joint entropy for multiple label vectors."""
    joint_distribution = np.vstack(labels).T
    joint_labels, joint_counts = np.unique(
        joint_distribution, axis=0, return_counts=True
    )
    joint_probabilities = joint_counts / len(joint_distribution)
    return -np.sum(joint_probabilities * np.log(joint_probabilities))


def total_correlation(*clusterings):
    """Compute the total correlation over multiple clusterings."""
    entropies = [m_entropy(labels) for labels in clusterings]
    joint_ent = joint_entropy(*clusterings)
    return np.sum(entropies) - joint_ent


def mutual_info_soft(U, V):
    """
    Compute the mutual information between two soft clusterings.

    Parameters:
    U (np.ndarray): r x n matrix of soft clustering assignments for the first clustering.
    V (np.ndarray): c x n matrix of soft clustering assignments for the second clustering.

    Returns:
    float: The mutual information between the two clusterings.
    """
    # Contingency table
    N = np.dot(U, V.T)
    total_samples = U.shape[1]

    # Row and column sums
    row_sums = np.sum(N, axis=1, keepdims=True)
    col_sums = np.sum(N, axis=0, keepdims=True)

    # Normalized contingency table
    N_normalized = N / total_samples

    # Avoid division by zero and log(0) issues
    epsilon = 1e-10
    expected_N = np.dot(row_sums, col_sums) / total_samples
    non_zero_entries = N > epsilon

    # Mutual Information calculation
    mutual_info = np.sum(
        N_normalized[non_zero_entries]
        * np.log(N[non_zero_entries] / (expected_N[non_zero_entries] + epsilon))
    )

    return mutual_info


def mutual_info_scores(labels_true, labels_pred):
    """Mutual information, adjusted mutual information, expected mutual information, and entropy over clusterings. (as in sklearn)

    :param labels_true: Cluster labels 1.
    :param labels_pred: Cluster labels 2.
    :return: MI, AMI, EMI, entropy(labels_true), entropy(labels_pred)
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
        return 1.0, 1.0, 0, 0, 1.0

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    # Calculate the expected value for the mutual information
    emi = expected_mutual_information(contingency, n_samples)

    h1, h2 = entropy(labels_true), entropy(labels_pred)

    normalizer = _generalized_average(h1, h2, "arithmetic")
    denominator = normalizer - emi
    # Avoid 0.0 / 0.0 when expectation equals maximum, i.e a perfect match.
    # normalizer should always be >= emi, but because of floating-point
    # representation, sometimes emi is slightly larger. Correct this
    # by preserving the sign.
    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        denominator = max(denominator, np.finfo("float64").eps)
    ami = (mi - emi) / denominator

    return mi, ami, emi, h1, h2


def entropy_score_2(labels):
    contingency = contingency_matrix(labels, labels, sparse=True)
    if isinstance(contingency, np.ndarray):
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    else:
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)

    contingency_sum = contingency.sum()

    # per cluster, element count
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    # labelling with zero entropy, i.e. containing a single cluster
    if pi.size == 1:
        return 0.0

    N = sum(nz_val)

    # count of elem per cluster
    # ai = [sum(nz_val[(np.where(nzx==i))]) for i in range(max(nzx)+1) => pi
    return -sum(pi / N * np.log(pi / N))



# targets = {0: [3], 1: [0], 2: [4], 3: [2]}
# pa, node = 3, 2
# idl_pa = to_node_idl(targets, pa, _hidden_truths.true_contexts, data)
# idl_combo = oracle_partition(
#    node, [], [pa], targets, _hidden_truths.true_contexts, data
# )
# adjusted_mutual_info_score(relabel_partitions(idl_pa, idl_combo)[0],relabel_partitions(idl_pa, idl_combo)[1] )
# adjusted_mutual_info_score(idl_pa, idl_combo)

from scipy.optimize import linear_sum_assignment


def relabel_partitions(target_labels, result_labels):
    max_label = int(max(target_labels.max(), result_labels.max()) + 1)
    cost_matrix = np.zeros((max_label, max_label))

    for i in range(max_label):
        for j in range(max_label):
            cost_matrix[i, j] = np.sum((target_labels == i) & (result_labels == j))

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    new_result_labels = np.zeros_like(result_labels)

    for i, j in zip(row_ind, col_ind):
        new_result_labels[result_labels == j] = i

    return target_labels, new_result_labels