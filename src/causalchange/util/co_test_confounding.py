import math

import numpy as np
import scipy

from src.causalchange.util.co_test_types import CoCoTestType
from src.causalchange.util.mi_sampling import Sampler
from src.causalchange.util.mi import mutual_info_scores


def test_confounded(map_i, map_j, test: CoCoTestType, sampler: Sampler):
    if test.value == CoCoTestType.SKIP.value:
        confounded = False
        return confounded, 0, pv(confounded), 0
    if test.value == CoCoTestType.MI_ZTEST.value:
        return test_confounded_Z_mi_sampling(map_i, map_j, sampler)
    elif test.value == CoCoTestType.AMI_ZTEST.value:
        return test_confounded_Z_ami_sampling(map_i, map_j, sampler)
    elif test.value == CoCoTestType.VI_ZTEST.value:
        return test_confounded_Z_vi_sampling(map_i, map_j, sampler)
    if test.value == CoCoTestType.MI.value:
        return test_confounded_exact_mi_sampling(map_i, map_j, sampler)
    elif test.value == CoCoTestType.VI.value:
        return test_confounded_exact_vi(map_i, map_j)
    else:
        raise ValueError()


def test_confounded_ami(map_i, map_j):
    mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)
    confounded = (ami > 0 and not math.isclose(ami, 0))
    return confounded, mi, pv(confounded), 0


def test_confounded_mi(map_i, map_j):
    mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)
    confounded = (mi > emi and not math.isclose(mi, emi))
    return confounded, mi, pv(confounded), 0


def test_confounded_exact_vi(map_i, map_j):
    mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)
    vi = h1 + h2 - 2 * mi
    confounded = (vi == 0 or math.isclose(vi, 0))

    return confounded, mi, pv(confounded), 0

def pv(confounded):
    if confounded:
        return 1
    else:
        return 0

def test_confounded_exact_mi_sampling(map_i, map_j, sampler: Sampler):
    N = len(map_i)
    assert(len(map_j)==N)
    n_shifts_i, n_shifts_j = len(np.unique(map_i))-1, len(np.unique(map_j))-1
    top_mi_sampling, _, _, _, _, mean_emi_sampling, \
    var_mi_sampling, var_ami_sampling, stdev_mi_sampling, stdev_ami_sampling, stdev_cent_sampling, stdev_vi_sampling = sampler.sample(N, n_shifts_i, n_shifts_j)

    mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)
    confounded = (mi > top_mi_sampling and not math.isclose(mi, top_mi_sampling))
    pval = 1
    if confounded:
        pval = 0
    return confounded, mi, pval, stdev_cent_sampling


def test_confounded_Z_mi_sampling(map_i, map_j, sampler: Sampler, t = 1.96):
    N = len(map_i)
    assert(len(map_j)==N)
    n_shifts_i, n_shifts_j = len(np.unique(map_i))-1, len(np.unique(map_j))-1
    top_mi_sampling, _, _, _, _, mean_emi_sampling, \
    var_mi_sampling, var_ami_sampling, stdev_mi_sampling, stdev_ami_sampling, stdev_cent_sampling, stdev_vi_sampling = sampler.sample(N, n_shifts_i, n_shifts_j)

    mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)

    if stdev_mi_sampling == 0:
        z = mi - emi
    else:
        z = (mi - emi) / stdev_mi_sampling #TODO emi or mean_emi_sampling?

    pval = scipy.stats.norm.sf(abs(z))

    confounded  = (z > t or z < -t)
    return confounded, mi, pval, stdev_cent_sampling


def test_confounded_Z_vi_sampling(map_i, map_j, sampler: Sampler,  t = 1.96):
    N = len(map_i)
    assert(len(map_j)==N)
    n_shifts_i, n_shifts_j = len(np.unique(map_i))-1, len(np.unique(map_j))-1
    top_mi_sampling, top_ami_sampling, top_vi_sampling, _, _, \
    mean_emi_sampling, \
    var_mi_sampling, var_ami_sampling, stdev_mi_sampling, stdev_ami_sampling, stdev_cent_sampling, stdev_vi_sampling = sampler.sample(N, n_shifts_i, n_shifts_j)

    mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)
    vi = h1 + h2 - 2 * mi

    if stdev_vi_sampling ==0:
        z = vi - top_vi_sampling
    else:
        z = (vi- top_vi_sampling) / stdev_vi_sampling #TODO emi or mean_emi_sampling?

    pval = scipy.stats.norm.sf(abs(z))

    confounded  = (z > t or z < -t)
    return confounded, mi, pval, stdev_cent_sampling


def test_confounded_Z_ami_sampling(map_i, map_j, sampler: Sampler, t = 1.96):
    N = len(map_i)
    assert(len(map_j)==N)
    n_shifts_i, n_shifts_j = len(np.unique(map_i))-1, len(np.unique(map_j))-1
    _, top_ami_sampling, _, _, _, mean_emi_sampling, \
    var_mi_sampling, var_ami_sampling, stdev_mi_sampling, stdev_ami_sampling, stdev_cent_sampling, stdev_vi_sampling = sampler.sample(N, n_shifts_i, n_shifts_j)

    mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)
    if stdev_ami_sampling == 0:
        z = ami - emi
    else:
        z = (ami - emi) / stdev_ami_sampling #TODO emi or mean_emi_sampling?
    pval = scipy.stats.norm.sf(abs(z))

    confounded = (z > t or z < t)
    return confounded, mi, pval, 0