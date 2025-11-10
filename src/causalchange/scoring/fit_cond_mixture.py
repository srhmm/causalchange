import os
from enum import Enum

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture

from src.causalchange.scoring.util_mixture import to_params, mix_regression_bic, mix_regression_params_kn_assgn


class MixingType(Enum):
    # mixtures of regressions
    MIX_LIN = 'mixLin'
    MIX_QUAD = 'mixQuad'
    MIX_CUB = 'mixCub'
    MIX_NS = 'mixNS'
    MIX_BS = 'mixBS'
    # simple baselines
    _BASE_BEST = 'clusBest'  # this selects per node the best MM which is not really fair; select for the whole graph the best one
    BASE_GMM = 'clusGMM'
    BASE_KMEANS = 'clusKmeans'
    BASE_SPECTRAL = 'clusSpectral'
    BASE_DBSCAN = 'clusDBSCAN'
    BASE_HDBSCAN = 'clusHDBSCAN'
    BASE_GMM_GLOB = 'clusGMMglobal'
    BASE_RANDOM_SPLIT = 'clusRandSplit'

    SKIP = '' #'skip'

    def __eq__(self, other): return self.value == other.value
    def __str__(self): return str(self.value)
    def search_each_node(self): return not self.value.endswith('global')

    def is_unconditional_mixture(self): return self.value.startswith('clus')


def fit_conditional_mixture(mty: MixingType, **kwargs):
    assert mty.value != MixingType.SKIP.value

    if mty.value.startswith('mix'):
        method= 'quad' if mty.value=='mixQuad' else 'cub' if mty.value=='mixCub' else 'ns' if mty.value=='mixNS' else \
            'bs' if mty.value=='mixBS' else 'lin'
        return fit_functional_mixture(**kwargs, method=method)
    elif mty.value.startswith('resid'):
        return fit_resid_mixture(mty, **kwargs)
    elif mty.value.startswith('clus'):
        return fit_marginal_mixture(mty, **kwargs)
    else:
        raise ValueError(mty)


def _fit_best_mixture(X, range_k, true_idl, sim_score=adjusted_mutual_info_score, sim_min=-np.inf):
    best_ami = sim_min
    best_arg = None
    for mty in [MixingType.BASE_GMM, MixingType.BASE_KMEANS, MixingType.BASE_SPECTRAL, MixingType.BASE_DBSCAN]:
        idl, pproba, div = fit_mixture_model(mty, X, range_k, None)
        ami = sim_score(true_idl, idl)
        if ami > best_ami:
            best_ami = ami
            best_arg = idl, pproba, div
    res_dict = dict(
        bic=0,
        idl=best_arg[0],
        pproba=best_arg[1],
    )
    return res_dict


def fit_mixture_model(mty, X, range_k, true_idl=None, kchoice_score=silhouette_score, kchoice_threshold=0.5,
                      kchoice_min=-1):
    if mty == MixingType.BASE_RANDOM_SPLIT:
        assert true_idl is not None
        true_k = len(np.unique(true_idl))
        # sample random labels with true k
        rand_split = np.random.choice(true_k, size=len(true_idl))
        res_dict = dict( bic=0, idl=rand_split )
        return rand_split, None, dict()

    elif mty == MixingType._BASE_BEST:
        assert true_idl is not None
        return _fit_best_mixture(X, range_k, true_idl)

    elif mty in [MixingType.BASE_GMM, MixingType.BASE_GMM_GLOB]:
        mm = GaussianMixture
        best_bic, best_k, best_m = np.inf, 0, None
        for k in range_k:
            gm = mm(k)
            gm.fit(X)
            bic_k = gm.bic(X)
            if bic_k < best_bic: best_bic, best_k, best_m = bic_k, k, gm

        res_dict = dict( bic=best_bic, idl=best_m.predict(X), pproba = best_m.predict_proba(X))
        return res_dict

    elif mty == MixingType.BASE_DBSCAN:
        mm = DBSCAN().fit(X)
        res_dict = dict(idl=mm.labels_)
        return res_dict
    elif mty == MixingType.BASE_HDBSCAN:

        from sklearn.cluster import  HDBSCAN
        mm = HDBSCAN().fit(X)
        res_dict = dict(idl=mm.labels_)
        return res_dict
    else:
        model = KMeans if mty == MixingType.BASE_KMEANS \
            else SpectralClustering if mty == MixingType.BASE_SPECTRAL else None
        if model is None: raise ValueError(mty)
        best_s, best_k, best_idl = kchoice_min, 1, None
        for k in range_k:
            if k == 1: continue
            mm = model(n_clusters=k, random_state=42)
            idl = mm.fit_predict(X)
            s = kchoice_score(X, idl)
            if s > best_s: best_s, best_k, best_idl = s, k, idl
        if best_s < kchoice_threshold:  best_idl = model(n_clusters=1, random_state=42).fit_predict(X)
        res_dict = dict(idl=best_idl)
        return res_dict


def fit_marginal_mixture(mty, X, node_i, pa_i, range_k, resid, true_idl, **kwargs):
    X = np.hstack([X[:, pa_i], X[:, node_i].reshape(-1, 1)]) if len(pa_i) > 0 else X[:, node_i].reshape(-1, 1)
    return fit_mixture_model(mty, X, range_k, true_idl)


def fit_resid_mixture(mty, X, node_i, pa_i, range_k, resid, true_idl):
    return fit_mixture_model(mty, resid, range_k)


def fit_functional_mixture(X, node_i, pa_i, range_k, resid, true_idl, lg=None, vb=0, degree=3, method="lin"):
    if not len(pa_i):
        return fit_marginal_mixture(MixingType.BASE_GMM, X, node_i, pa_i, range_k, resid, true_idl)
    if lg is not None and vb > 0: lg.info(f"Fitting mixture ({method})")

    import rpy2.robjects as robjects
    import numpy as np
    from rpy2.robjects import Formula
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    flexmix = importr('flexmix')
    splines = importr('splines')

    y = X[:, node_i].reshape(-1, 1)
    X_pa = X[:, pa_i]

    data_np = np.hstack([y, X_pa])
    data_r = robjects.r.matrix(data_np, nrow=data_np.shape[0], ncol=data_np.shape[1])
    robjects.r.assign("data_r", data_r)
    r_df = robjects.r['data.frame'](x=data_r)

    rhs_terms = []
    for i in range(X_pa.shape[1]):
        xi = f"x.{i + 2}"
        if method == "quad": rhs_terms.append(f"poly({xi}, {2})")
        elif method == "cub":
            degree = 3
            rhs_terms.append(f"poly({xi}, {degree})")
        #deg >3 useful? mixed?
        elif method == "ns":
            rhs_terms.append(f"ns({xi}, df={degree})")
        elif method == "bs":
            rhs_terms.append(f"bs({xi}, df={degree})")
        else:
            rhs_terms.append(xi)#linear

    formula_str = f"x.1 ~ " + " + ".join(rhs_terms)
    formula = Formula(formula_str)

    if lg is not None and vb > 0: lg.info(f"Formula: {formula}")

    best_bic = np.inf
    best_model = None
    best_k = None

    for k in range_k:
        m = flexmix.flexmix(formula, data=r_df, k=k)
        bic = robjects.r['BIC'](m)[0]
        if vb:
            print(f"k={k}, BIC={bic}")
        if bic < best_bic:
            best_bic = bic
            best_model = m
            best_k = k

    post_probs = np.array(robjects.r['posterior'](best_model))
    hard_assign = post_probs.argmax(axis=1)

    def post_entropy(p_proba, eps=1e-12):
        p_safe = np.clip(p_proba, eps, 1.0)
        return -np.sum(p_safe * np.log(p_safe), axis=1)

    ent_idl = post_entropy(post_probs.reshape(1, -1))
    res_dict = dict(
        bic=best_bic,
        idl=hard_assign,
        pproba=post_probs,
        entropy=ent_idl,
        best_k=best_k
    )
    return res_dict


def conditional_mixture_known_assgn(X, node_i, pa_i, true_idl, **scoring_params):
    """ fit regresssions for a known mix assignment, pproba from log liks of those regressions (todo or degen?) """
    if len(pa_i) > 0:
        (Xx, y) = (X[:, pa_i], X[:, node_i])
        beta_l, sig_l = mix_regression_params_kn_assgn(Xx, y, true_idl)
        bic = mix_regression_bic(Xx, y, true_idl, beta_l, sig_l)
        pproba = None
        ent_idl = 0
    else:
        pproba = None
        bic = 0
        ent_idl = 0
    res_dict = dict(
        bic=bic,
        idl=true_idl,
        pproba=pproba,
        ent_idl=ent_idl
    )
    return res_dict
