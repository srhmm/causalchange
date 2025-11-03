import itertools

import numpy as np

from src.coco.util.co_test_types import CoShiftTestType
from src.coco.util.utils import pval_to_map
from src.coco.util.sparse_shift.testing import test_mechanism


def co_shift_test(Dc: np.array, node_i: int, pa_i: list, shift_test: CoShiftTestType, alpha=0.5):
    """ Discovers a grouping of contexts

    :param Dc: data
    :param node_i: target node
    :param pa_i: causal parents of target
    :param shift_test: test type
    :param alpha
    :return:
    """

    n_c = Dc.shape[0]

    n_pairs = len([i for i in itertools.combinations(range(n_c), 2)])
    pval_mat = np.ones((n_pairs, n_pairs))

    if shift_test.value == CoShiftTestType.SKIP.value:
        mp = [0 for _ in range(n_c)]

    elif shift_test.value == CoShiftTestType.VARIO.value:
        # D_up, pa_up_i = Dc, pa_i
        raise ValueError('Vario currently not supported')
        # pistar, _, _ = causal_mechanism_search(D_up, node_i, pa_up_i, greedy=False)
        # map = partition_to_map(pistar)

    elif shift_test.value == CoShiftTestType.VARIO_GREEDY.value:
        raise ValueError('Vario currently not supported')
        # D_up, pa_up_i = Dc, pa_i
        # pistar, _, _ = causal_mechanism_search(D_up, node_i, pa_up_i, greedy=True)
        # map = partition_to_map(pistar)
        
    elif shift_test.value == CoShiftTestType.LINC.value:
        raise ValueError('Linc currently not supported')
        # vsn = Vsn(rff=True, ilp=True, clus=False)
        # pistar, _ = pi_partition_search(D_up, node_i, pa_up_i, vsn)
        # map = partition_to_map(pistar)
        
    elif shift_test.value == CoShiftTestType.PI_KCI.value:
        parents = [1 if node_i in pa_i else 0 for node_i in range(Dc.shape[2])]
        pval_mat = test_mechanism(Dc, node_i, parents, 'kci', {})
        mp = pval_to_map(pval_mat, alpha=alpha)
    else:
        raise ValueError()
    return mp, pval_mat


def _augment(Dc):
    n = Dc.shape[2]+1
    D = np.random.normal(size=(Dc.shape[0], Dc.shape[1], n))
    D[:, :, range(n-1)] = Dc
    return D, [n]
