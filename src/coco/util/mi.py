import numpy as np
from scipy import sparse as sp
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import contingency_matrix, entropy, expected_mutual_information
from sklearn.metrics.cluster._supervised import check_clusterings, _generalized_average


def mutual_info_scores(labels_true, labels_pred):
    ''' Mutual information, adjusted mutual information, expected mutual information, and entropy over clusterings. (as in sklearn)

    :param labels_true: Cluster labels 1.
    :param labels_pred: Cluster labels 2.
    :return: MI, AMI, EMI, entropy(labels_true), entropy(labels_pred)
    '''
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


def entropy_score(labels):
    return entropy(labels)


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