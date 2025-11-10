import math
from statistics import mean, variance, stdev

import numpy as np

from src.coco.gen.dag_gen import gen_partition
from src.causalchange.util.mi import mutual_info_scores
from src.coco.util.utils import partition_to_map


class Sampler:
    def __init__(self, reps=100, alpha=0.05, sampler=gen_partition):
        self.sampler = sampler
        self.reps = reps
        self.alpha = alpha
        self.table = {}

    def sample(self, n_c, n_shifts_i, n_shifts_j):
        key = str(n_c)+'_'+str(n_shifts_i)+'_'+str(n_shifts_j)
        if not self.table.__contains__(key):
            self.table[key] = sampling_mi_entropy(n_c, n_shifts_i, n_shifts_i, self.reps, self.alpha, self.sampler)

        return self.table[key]


def sampling_mi_entropy(n_c, n_shifts_i, n_shifts_j, reps=1000, alpha=0.05,
                        sampler=gen_partition):
    ''' Samples partitions with a given number of groups at random to find sig. thresholds for MI and entropy.
    :param n_c: Number of samples in a partitions
    :param n_shifts_i: Number of groups in clustering i
    :param n_shifts_j: Number of groups in clustering j
    :param reps: Number partitions that will be sampled with n_shifts_i and n_shifts_j
    :param alpha: Sign level, top (rep*alpha) partitions will be considered exceptional
    :param sampler: Model of randomness we assume for partitions, e.g. gen_partition for permutation model
    :return:
    '''
    ents1 = [0 for _ in range(reps)]
    ents2 = [0 for _ in range(reps)]
    cond_ents = [0 for _ in range(reps)]
    mis = [0 for _ in range(reps)]
    amis = [0 for _ in range(reps)]
    emis = [0 for _ in range(reps)]
    vis = [0 for _ in range(reps)]

    seed = 0

    for rep in range(reps):
        seed = seed + 2
        pi1 = sampler(seed, n_c, n_shifts_i)
        pi2 = sampler(seed+1, n_c, n_shifts_j)
        labels1 = partition_to_map(pi1)
        labels2 = partition_to_map(pi2)
        mi, ami, emi, h1, h2 = mutual_info_scores(labels1, labels2)

        vi = h1 + h2 - 2 * mi
        ents1[rep] = h1
        ents2[rep] = h2
        cond_ents[rep] = emi - h1
        mis[rep] = mi
        vis[rep] = vi
        amis[rep] = ami
        emis[rep] = emi

    mi_sampling = np.sort(mis)[np.int64(np.floor(alpha * reps))]
    ami_sampling = np.sort(amis)[np.int64(np.floor(alpha * reps))]
    vi_sampling = np.sort(vis)[np.int64(np.floor(alpha * reps))]
    ent1_sampling = np.sort(ents1)[np.int64(np.floor(alpha * reps))]
    ent2_sampling = np.sort(ents2)[np.int64(np.floor(alpha * reps))]
    cents_sampling = np.sort(cond_ents)[np.int64(np.floor(alpha * reps))]

    return mi_sampling, ami_sampling, vi_sampling, \
           ent1_sampling, ent2_sampling, \
           mean(emis), variance(mis), variance(amis), \
           stdev(mis), stdev(amis), stdev(cond_ents), stdev(vis) # mean(emis) should roughly corresp to mi_sampling. these are not very meaningful: mean(mis), mean(ents1), mean(ents2), emi_sampling(?)


def sampling_mi_entropy_group_sizes():
    pass

def sample_partition_interventions(seed, n_c, n_shifts):
    pass  # does this random model make sense? n_shifts corresponds to a single partition (up to permutations).


def sampling_mi_entropy_any(n_c, reps=1000, alpha=0.05):
    sub_reps = math.floor(reps / (n_c - 1))
    reps = sub_reps * (n_c - 1)

    ents = [0 for _ in range(reps)]
    mis = [0 for _ in range(reps)]
    emis = [0 for _ in range(reps)]

    for partition_size in range(n_c):
        for rep in range(sub_reps):
            seed = rep
            np.random.seed(seed)
            pi1 = gen_partition(seed, n_c, partition_size)
            pi2 = gen_partition(seed, n_c, partition_size)
            labels1 = partition_to_map(pi1)
            labels2 = partition_to_map(pi2)
            mi, _, emi, h1, _ = mutual_info_scores(labels1, labels2)
            ents[rep] = h1
            # cond_ents[rep] = emi - h1
            mis[rep] = mi
            emis[rep] = emi

    mi_sampling = np.sort(mis)[np.int64(np.floor(alpha * reps))]
    emi_sampling = np.sort(emis)[np.int64(np.floor(alpha * reps))]

    return mi_sampling, emi_sampling, mean(ents), mean(mis), mean(emis)

