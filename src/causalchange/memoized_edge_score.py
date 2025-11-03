from typing import List, Tuple

import numpy as np

from src.causalchange.mixing.mixing import conditional_mixture_known_assgn, MixingType
from src.causalchange.mixing.mixing import fit_conditional_mixture
from src.causalchange.mixing.regression import ScoreType, GPType, fit_functional_model_context, \
    fit_functional_model, fit_score_gp, fit_score_rff, DataMode
from src.causalchange.mixing.regression import fit_fun, RegressorType
from src.causalchange.util.utils_idl import _get_true_idl, _get_true_idl_Z


class MemoizedEdgeScore:

    def __init__(self, X, data_mode: DataMode, score_type: ScoreType, mixing_type: MixingType, lg=None, vb=0):
        self.X = X
        self.data_mode = data_mode
        self.score_type = score_type
        self.mixing_type = mixing_type
        self.lg = lg
        self.vb = vb
        self._info = lambda st: (self.lg.info(st) if self.lg is not None else print(st)) if self.vb > 0 else None

        # Memoized info
        self.score_cache = {}
        self.res_cache = {}
        self.resid_cache = {}

    def score_edge(self, j, pa, **scoring_params) -> (float,dict):
        """
        Evaluates score for a causal relationship pa(Xj)->Xj.

        :param j: Xj
        :param pa: pa(Xj)
        :return: score_up=score(Xpa->Xj)
        """
        hash_key = f'j_{str(j)}_pa_{str(pa)}'

        if self.score_cache.__contains__(hash_key):
            assert hash_key in self.res_cache
            return self.score_cache[hash_key], self.res_cache[hash_key]

        if self.data_mode == DataMode.IID:
            assert self.mixing_type == MixingType.SKIP
            score, res = fit_functional_model(self.X, pa=pa, target=j, **scoring_params)
        elif self.data_mode == DataMode.CONTEXTS:
            assert self.mixing_type == MixingType.SKIP
            assert isinstance(self.X, dict)
            fun_gp = fit_score_gp if self.score_type == GPType.EXACT \
                else fit_score_rff if self.score_type == GPType.FOURIER else None
            if fun_gp is None: raise ValueError("use either GPs or RFFs here!")
            alpha_gp_mdl = scoring_params.get("alpha_gp_mdl", 0.05)
            score, res = fit_functional_model_context(
                self.X, pa=pa, target=j, fun_gp=fun_gp, alpha=alpha_gp_mdl)
        elif self.data_mode.value == DataMode.MIXED.value:
            assert self.mixing_type != MixingType.SKIP
            assert self.score_type == ScoreType.LIN
            resi = None if scoring_params.get("oracle_Z") else self.resid_edge(j,  pa)  # remove? resids not used in final version

            score, res = idl_and_latent_bic_score(self.mixing_type,
                                     self.X, covariates=pa, target=j, resid=resi, **scoring_params)
        else: raise ValueError(self.data_mode)

        self.score_cache[hash_key] = score
        self.res_cache[hash_key] = res
        return score, res



    def resid_edge(self, j: int, pa: [int]) -> np.array:
        """Residual for average functional model for edge pa to node j."""
        hash_key = f"j_{str(j)}_pa_{str(pa)}"

        if self.resid_cache.__contains__(hash_key):
            return self.resid_cache[hash_key]

        resids, strength = fit_fun(self.X[:, pa], self.X[:, j], RegressorType.LN, 42)

        self.resid_cache[hash_key] = resids
        return resids



def local_score_latent_bic(
    Data: Tuple[np.ndarray, int], i: int, PAi: List[int], parameters=None
) -> float:
    """ compute the latent-aware BIC, to do so we fit a MLR using EM.
    for use within causallearn.score.LocalScoreFunctionClass.LocalScoreClass """

    if parameters is None: kmax = 5
    else: kmax = parameters.get("k_max", 5)
    params = {"k_max" : kmax, "oracle_K": False, "oracle_Z": False}
    score, _ = idl_and_latent_bic_score(MixingType.MIX_LIN, Data, PAi, i, None, **params)
    return score


def idl_and_latent_bic_score(
        mixing_type,
        X,
        covariates: list,
        target: int,
        resid=None,
        **params) -> [List, List, dict]:
    if params.get("true_idls") is not None:
        true_idl = _get_true_idl(params["true_idls"], covariates, target, params["t_A"])
    elif params.get("t_Z") is not None:
        true_idl = _get_true_idl_Z(
            covariates, target, params["t_A"], params["t_Z"], params["t_n_Z"], X.shape[0])
    else: true_idl = None

    if params["oracle_Z"]:
        assert true_idl is not None
        true_idl, true_pproba, true_dict = conditional_mixture_known_assgn(
            X=X, node_i=target, pa_i=covariates, true_idl=true_idl, **params)
        return true_idl, true_pproba, true_dict

    range_k = range(1, params["k_max"] + 1) if not params["oracle_K"] else [len(np.unique(true_idl))]
    res_dict = fit_conditional_mixture(
        mty=mixing_type, X=X, node_i=target, pa_i=covariates, range_k=range_k, resid=resid, true_idl=true_idl,
        lg=params.get("lg", None))
    score = res_dict["bic"] #idl_dict.get("bic", 0)
    return score, res_dict


