from typing import List, Tuple

import numpy as np

from src.causalchange.scoring.fit_cond_mixture import conditional_mixture_known_assgn, MixingType, fit_conditional_mixture
from src.causalchange.scoring.fit_score import fit_fun_IID, fit_score_gp, fit_score_rff, fit_score_krr
from src.causalchange.cc_types import ScoreType, GPType, DataMode
from src.causalchange.scoring.fit_score import fit_score_ln, fit_score_gam, fit_score_spln, fit_resid_CONTEXTS, fit_resid_TIME, fit_resid_TIME_CONTEXTS
from src.causalchange.scoring.test_ci import test_fun_kci
from src.causalchange.search.partition_search import fit_fun_CONTEXTS, fit_citest_CONTEXTS, fit_fun_MIXED
from src.causalchange.util.utils_idl import get_true_idl, get_true_idl_Z


class EdgeMemoized:

    def __init__(self, X, data_mode: DataMode, score_type: ScoreType, mixing_type: MixingType, **scoring_params):
        self.X = X
        self.data_mode = data_mode
        self.score_type = score_type
        self.mixing_type = mixing_type
        self.scoring_params = scoring_params
        self.lg = scoring_params.get("lg", None)
        self.vb = scoring_params.get("vb", 0)
        self._info = lambda st: (self.lg.info(st) if self.lg is not None else print(st)) if self.vb > 0 else None

        # Memoized info
        self.score_cache = {}
        self.res_cache = {}
        self.discrep_cache = {}
        self.res_discrep_cache = {}
        self.resid_cache = {}

        def __eq__(self, other):
            return self.value == other.value

    def get_score_fun(self):
        #regression model
        score_fun = fit_score_krr if self.score_type == ScoreType.KRR \
            else fit_score_gp if self.score_type.value == GPType.EXACT.value \
            else fit_score_rff if self.score_type.value == GPType.FOURIER.value \
            else fit_score_gam if self.score_type == ScoreType.GAM \
            else fit_score_spln if self.score_type == ScoreType.SPLINE \
            else fit_score_ln if self.score_type == ScoreType.LIN \
            else None
        return score_fun


    def discrepancy(self, j, pa) -> (float,dict):
        """
        Evaluates discrepancy of conditional distributions P(Xj | pa(Xj), C=c) across different values of c (usually c are context labels, but they can also be temporal regimes separated by changepoints, or unknown labels).

        :param j: Xj
        :param pa: pa(Xj)
        :return: discrepancy, usually sth like \sum_{pairs c1, c2} d(residuals of the regression Xpa->Xj in c1; same residuals in c2)
        """
        hash_key = f'j_{str(j)}_pa_{str(pa)}'

        if self.discrep_cache.__contains__(hash_key):
            assert hash_key in self.res_discrep_cache
            return self.discrep_cache[hash_key], self.res_discrep_cache[hash_key]

        score_fun = self.get_score_fun()
        assert score_fun is not None, f"no scoring function for {self.score_type}"
        discrep_fun = mmd_rbf # could change based on a discrep_type if we want

        if self.data_mode == DataMode.CONTEXTS:
            resid_sets = fit_resid_CONTEXTS(self.X, j, pa, score_fun)

        elif self.data_mode == DataMode.TIME:
            resid_sets = fit_resid_TIME(self.X, j, pa, score_fun, self.scoring_params.get("changepoints"))
        elif self.data_mode == DataMode.TIME_CONTEXTS:
            resid_sets = fit_resid_TIME_CONTEXTS(self.X, j, pa, score_fun,
                                                 self.scoring_params.get("changepoints_per_context"))
        elif self.data_mode == DataMode.MIXED:
            raise NotImplementedError
           # resid_sets = fit_resid_MIXED(self.X, j, pa, score_fun)
        else: raise ValueError(self.data_mode)

        discrep, res = discrepancy_from_resid(resid_sets, mmd_fn=discrep_fun, aggregate="sum")

        self.discrep_cache[hash_key] = discrep
        self.res_discrep_cache[hash_key] = res
        return discrep, res

    def score_edge(self, j, pa) -> (float,dict):
        """
        Evaluates a score (regularized likelihood, MDL) for a (functional) relationship pa(Xj)->Xj.

        :param j: Xj
        :param pa: pa(Xj)
        :return: score(Xpa->Xj)
        """
        hash_key = f'j_{str(j)}_pa_{str(pa)}'

        if self.score_cache.__contains__(hash_key):
            assert hash_key in self.res_cache
            return self.score_cache[hash_key], self.res_cache[hash_key]

        score_fun = self.get_score_fun()
        test_fun = test_fun_kci
        if self.score_type.is_scorebased(): assert score_fun is not None, f"no scoring function for {self.score_type}"
        if self.score_type.is_constraintbased(): assert test_fun is not None, f"no ci test for {self.score_type}"

        if self.data_mode == DataMode.IID:
            assert self.mixing_type == MixingType.SKIP
            score, res = fit_fun_IID(self.X, pa=pa, target=j, score_fun=score_fun, **self.scoring_params)
        elif self.data_mode == DataMode.CONTEXTS:
            assert self.mixing_type == MixingType.SKIP
            assert isinstance(self.X, dict)
            if self.score_type.is_scorebased():
                score, res = fit_fun_CONTEXTS(
                    self.X, pa=pa, target=j, score_fun=score_fun)
            elif self.score_type.is_constraintbased():
                score, res = fit_citest_CONTEXTS(
                    self.X, pa=pa, target=j )
        elif self.data_mode == DataMode.TIME:
            raise NotImplementedError #fit_fun_partition_TIME
        elif self.data_mode == DataMode.TIME_CONTEXTS:
            raise NotImplementedError #fit_fun_partition_TIME_CONTEXTS
        elif self.data_mode == DataMode.CONFOUNDED:
            raise NotImplementedError#? something w coco?
        elif self.data_mode == DataMode.MIXED:
            assert self.mixing_type != MixingType.SKIP
            score, res = fit_fun_MIXED(self.mixing_type,
                                       self.X, covariates=pa, target=j, resid=None, **self.scoring_params)
        else:
            raise ValueError(self.data_mode)

        self.score_cache[hash_key] = score
        self.res_cache[hash_key] = res
        return score, res



def local_score_latent_bic(
    Data: Tuple[np.ndarray, int], i: int, PAi: List[int], parameters=None
) -> float:
    """ compute the latent-aware BIC, to do so we fit a MLR using EM.
    for use within causallearn.score.LocalScoreFunctionClass.LocalScoreClass """

    if parameters is None: kmax = 5
    else: kmax = parameters.get("k_max", 5)
    params = {"k_max" : kmax, "oracle_K": False, "oracle_Z": False}
    score, _ = fit_fun_MIXED(MixingType.MIX_LIN, Data, PAi, i, None, **params)
    return score



def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """ Unbiased empirical MMD^2 w RBF
    :param x: shape (n_x, d) or (n_x,)
    :param y: shape (n_y, d) or (n_y,)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # ensure shape (n_samples, n_features)
    if x.ndim == 1: x = x[:, None]
    if y.ndim == 1: y = y[:, None]

    n = x.shape[0]
    m = y.shape[0]

    # if too few samples, return 0
    if n < 2 or m < 2:
        return 0.0

    if gamma is None:
        # heuristic: use median pairwise distance in pooled sample
        z = np.vstack([x, y])
        dists = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
        if z.shape[0] > 1:
            median_sqdist = np.median(dists)
        else:
            median_sqdist = 1.0
        gamma = 1.0 / (2.0 * median_sqdist + 1e-8)

    def k(a, b):
        sqdist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * sqdist)

    K_xx = k(x, x)
    K_yy = k(y, y)
    K_xy = k(x, y)

    mmd2 = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1)) \
         + (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1)) \
         - 2.0 * K_xy.mean()

    return float(mmd2)


from itertools import combinations

def discrepancy_from_resid(
    residual_sets: list[np.ndarray],
    mmd_fn = mmd_rbf,
    aggregate: str = "sum"
) -> tuple[float, dict]:
    """  residual_sets: list of arrays, one per regime (context or segment)
    :param mmd_fn: function (x, y) -> float (MMD)
    :param  aggregate: "sum" or "avg"

    :return: ``discrep`` - aggregated discrepancy across all regime pairs,
       ``details`` -  dict with pairwise MMDs, keyed by (i, j)
    """
    n_sets = len(residual_sets)
    pairwise = {}
    values = []

    for (i, j) in combinations(range(n_sets), 2):
        mmd_ij = mmd_fn(residual_sets[i], residual_sets[j])
        pairwise[(i, j)] = mmd_ij
        values.append(mmd_ij)

    if not values:
        return 0.0, pairwise

    values = np.array(values, dtype=float)

    if aggregate == "sum": discrep = float(values.sum())
    elif aggregate == "avg": discrep = float(values.mean())
    else:  raise ValueError(f"'{aggregate}', use 'sum' or 'avg'")

    return discrep, pairwise
