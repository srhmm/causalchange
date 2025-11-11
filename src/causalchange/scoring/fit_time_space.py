import logging

import numpy as np
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.causalchange.scoring.test_cond_discrepancy import DiscrepancyTestType
from src.causalchange.search.cps_search import mdl_error_cp_search, eval_fit_partition
from src.causalchange.util.sttypes import TimeseriesScoringFunction
from src.causalchange.util.util_regimes import r_partition_to_windows_T
from src.causalchange.util.utils import data_scale


def hybrid_score_edge_time_space(
        data_C: dict, covariates: list, target: int, max_lag: int, min_dur: int,
        scoring_function: TimeseriesScoringFunction, discrepancy_test_type: DiscrepancyTestType,
        lg: logging = None, verbosity: int = 0, edge_info=""):
    """First find the CPS and regime partition for node given its parents, then compute the score"""

    n_n = data_C[0].shape[0]
    target_links = _lagged_target_links(covariates, target)
    # Consider only the target and its parents
    links = {0: {i: target_links if i == target else [((i, -1), None, None)] for i in range(n_n)}}

    found_r_partition, _, _, _ = mdl_error_cp_search(data_C, links, max_lag, discrepancy_test_type, min_dur)

    windows_T = r_partition_to_windows_T(found_r_partition, max_lag)
    mdl_scores = score_edge_time_space(
        data_C, covariates, target, windows_T, found_r_partition, max_lag, min_dur,
        scoring_function, discrepancy_test_type,
        lg, verbosity, edge_info)
    if verbosity > 1:
        lg.info(f"\t\t\t-edge {target} <- {covariates}\ts: {[score for score in mdl_scores]}"
                f"\tcps: {windows_T}\ttruth: {edge_info}")

    return mdl_scores, windows_T


def score_edge_time_space(
        data_C: dict, covariates: list, target: int, windows_T: list, regimes_R: list, max_lag: int, min_dur: int,
        scoring_function: TimeseriesScoringFunction, discrepancy_test_type: DiscrepancyTestType,
        lg: logging = None, verbosity: int = 0, edge_info="", ) -> np.array:
    """Compute the score for a node given its parents, using the provided CPS and regime partition"""
    target_links = _lagged_target_links(covariates, target)
    C = len(data_C)
    T = len(windows_T)
    only_one_regime_exists = len(np.unique([regime for _, _, regime in regimes_R])) == 1

    # Fit a different GP in each context and each time window/chunk, disregard the regime labels
    if only_one_regime_exists or discrepancy_test_type == DiscrepancyTestType.SKIP:
        # Fit a GP for each edge in each context and each time window
        scores_time_space = fit_functional_model_time_space(
            scoring_function, data_C, windows_T, target_links, target, return_models=False)

        scores = [[scores_time_space[c_i][w_i] for w_i in range(T)] for c_i in range(C)]
        #if verbosity > 1:
            #lg.info(
            #    f'\tEval Edge {covariates} -> {target}: {np.round(np.flatten(sum(sum(np.array(scores_time_space)))), 2)}\t{edge_info}')
        return scores
    # Fit a different GP in each context and same regime
    else:
        # only consider this target
        links = {0: {0: target_links}}
        # Given the current CPS, find the regime partition and sum up scores
        score, r_partition, contexts_per_node, regimes_per_node, gps, hist, scores = \
            eval_fit_partition(data_C, links, windows_T, max_lag, discrepancy_test_type, hist=None)
        return scores


def score_edge_continuous(
        data_C: dict, covariates: list, target: int,
        scoring_function, lg: logging = None, verbosity: int = 0, info: str = "") -> int:
    raise NotImplementedError("Potential support for non-time series scoring functions.")


def _lagged_target_links(covariates, i):
    """Generates the links for a single effect, with given time lags"""
    links = []
    links.append(((i, -1), 1, None))
    # Add causal parents
    for j, lag in covariates:
        links.append(((j, -lag), 1, None))
    return links


def fit_gp_time_space(**kwargs):
    """ fit we use in CPS search (needs gp residuals) """
    return fit_functional_model_time_space(
        TimeseriesScoringFunction.GP,
        return_models=True, **kwargs)


def fit_functional_model_time_space(
        scoring_function: TimeseriesScoringFunction,
        data_C, windows_T, target_links, target, contexts=None, regimes=None, return_models=False):
    """ Fit a functional model over time (regimes) and space (context)

    :param scoring_function: MDL score
    :param data_C: data
    :param windows_T: time windows segmented by cutpoints
    :param target_links: causal links towards target
    :param target: target node
    :param contexts: if known, contexts (dataset groups)
    :param regimes: if known, regimes (time window groups)
    :param return_models: return regression models
    :return:
    """

    gp_time_space = dict()
    contexts = list(range(len(data_C))) if contexts is None else contexts
    regimes = list(range(len(windows_T))) if regimes is None else regimes

    for context in set(contexts):
        gp_time_space[context] = dict()
        for regime in set(regimes):
            gp_time_space[context][regime]= fit_functional_model_r_c(context, regime, scoring_function, #subsampling_approach,
                                                                     data_C, windows_T, target_links, target, contexts, regimes, return_models)

    return gp_time_space

def fit_functional_model_r_c( context, regime, scoring_function: TimeseriesScoringFunction,
        #subsampling_approach: SubsamplingApproach,
        data_C, windows_T, target_links, target, contexts=None, regimes=None, return_models=False):
        pa_i = [(var, lag) for (var, lag), _, _ in target_links]
        M = len(pa_i)
        data_pa_i, data_all, data_node_i = collect_subsample(#subsampling_approach,
                                                             data_C=data_C, target_links=target_links, context=context,
                                                             regime=regime, contexts=contexts,
                                                             regimes=regimes, target=target, windows_T=windows_T)
        if M == 0:
            data_pa_i = data_node_i.reshape(-1, 1)
        score, model = fit_functional_model(scoring_function, data_pa_i, data_node_i, pa_i, data_C)
        return (score, model, data_pa_i, data_node_i, data_all) if return_models else score

def fit_functional_model(scoring_function: TimeseriesScoringFunction, data_pa_i, data_node_i, pa_i, data_C):
    """ Fits a functional model with corresponding MDL score. """
    if scoring_function.value == TimeseriesScoringFunction.GLOBE.value:
        score, spline = fit_MARS_regression_spline(data_C, pa_i, data_pa_i, data_node_i)
        return score, spline
    else:
        assert scoring_function.value in [
            TimeseriesScoringFunction.GP.value,
            TimeseriesScoringFunction.GP_QFF.value
        ]
        gp = fit_gaussian_process(
            data_pa_i, data_node_i,
            scoring_function=scoring_function,
            check_fit=False)
        score, lik, model, pen = gp.mdl_score_ytrain()
        return score, gp




def collect_subsample(data_C, target_links, context, regime, contexts, regimes, target, windows_T):
    """ Subsample the time series. """
    pa_i = [(var, -np.abs(lag)) for (var, lag), _, _ in target_links]  # convention neg lags
    M = len(pa_i)
    for k in data_C:
        N = data_C[k].shape[1]
    data_pa_i = np.zeros((1, M), dtype='float32')
    data_all = np.zeros((1, N + 1), dtype='float32')
    data_node_i = np.zeros((1), dtype='float32')

    for dataset in [d for d in range(len(data_C)) if contexts[d] == context]:
        data = data_C[dataset]

        for window, (t0, tn) in enumerate(windows_T):
            if regimes[window] != regime: continue
            T = tn - t0
            data_pa_i_w = np.zeros((T, M), dtype='float32')
            data_all_w = np.zeros((T, N + 1), dtype='float32')

            for j, (var, lag) in enumerate(pa_i):
                if var == target:
                    data_all_w[:, N] = data[t0 + lag:tn + lag, var]
                data_pa_i_w[:, j] = data[t0 + lag:tn + lag, var]
                data_all_w[:, var] = data[t0 + lag:tn + lag, var]

            data_pa_i = np.concatenate([data_pa_i, data_pa_i_w])
            data_all = np.concatenate([data_all, data_all_w])
            data_node_i = np.concatenate([data_node_i, data[t0:tn, target]])

    data_pa_i = data_pa_i[1:]
    data_all = data_all[1:]
    data_node_i = data_node_i[1:]

    return data_pa_i, data_all, data_node_i



def fit_MARS_regression_spline(data_C, pa_i, data_pa_i, data_node_i):
    """ Spline Regression. Mini GLOBE implementation (Mian et al. 2021)
    :param data_C:
    :param pa_i:
    :param data_pa_i:
    :param data_node_i:
    :return:
    """
    from src.causalchange.scoring.spline_mdl import Slope

    def _min_diff(tgt):
        sorted_v = np.copy(tgt)
        sorted_v.sort(axis=0)
        diff = np.abs(sorted_v[1] - sorted_v[0])
        if diff == 0: diff = np.array([10.01])
        for i in range(1, len(sorted_v) - 1):
            curr_diff = np.abs(sorted_v[i + 1] - sorted_v[i])
            if curr_diff != 0 and curr_diff < diff:
                diff = curr_diff
        return diff

    def _combinator(M, k):
        from scipy.special import comb
        sum = comb(M + k - 1, M)
        if sum == 0:
            return 0
        return np.log2(sum)

    def _aggregate_hinges(interactions, k, slope_, F):
        cost = 0
        for M in hinges:
            cost += slope_.logN(M) + _combinator(M, k) + M * np.log2(F)
        return cost

    source_g = data_pa_i
    target_g = data_node_i
    slope_ = Slope()
    globe_F = 9
    k, dim, M, rows, mindiff = np.array([len(pa_i)]), data_C[0].shape[1], 3, data_C[0].shape[0], _min_diff(target_g)
    base_cost = slope_.model_score(k) + k * np.log2(dim)
    sse, model, coeffs, hinges, interactions = slope_.FitSpline(source_g, target_g, M, False)
    base_cost = base_cost + slope_.model_score(hinges) + _aggregate_hinges(interactions, k, slope_, globe_F)
    cost = slope_.gaussian_score_emp_sse(sse, rows, mindiff) + model + base_cost
    return cost, slope_


def fit_gaussian_process(
        X, y, scoring_function=TimeseriesScoringFunction.GP,
        alpha=1.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2),
        scale=True, grid_search=False,
        show_plt=False, check_fit=False):
    """GP regression.

    :param X: parents
    :param y: target
    :param scoring_function: gp or qff
    :param alpha: rbf kernel param
    :param length_scale: rbf kernel param
    :param length_scale_bounds: rbf kernel param
    :param scale: scale data
    :param grid_search: kernel parameter tuning
    :param show_plt: plot
    :param check_fit:
    :return: GP_k, gaussian process per context/group
    """
    kernel = 1 * RBF(length_scale=length_scale,
                     length_scale_bounds=length_scale_bounds)
    size_tr_local = len(X)
    tr_indices = np.sort(np.random.RandomState().choice(len(X), size=size_tr_local, replace=False))
    Xtr = X[tr_indices]
    ytr = y[tr_indices]

    if scale:
        Xtr = data_scale(Xtr)
        ytr = data_scale(ytr.reshape(-1, 1))

    # Optional: Grid search for kernel parameter tuning
    param_grid = [{
        "alpha": [1e-2, 1e-3],
        "kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]
    }, {
        "alpha": [1e-2, 1e-3],
        "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]}]
    score = "r2"

    if grid_search:
        gaussianProcess = scoring_function.get_model(kernel, alpha, n_restarts_optimizer=9)
        gaussianProcessGrid = GridSearchCV(estimator=gaussianProcess, param_grid=param_grid, cv=4,
                                           scoring='%s' % score)
        gaussianProcessGrid.fit(Xtr, ytr)
        gp = gaussianProcessGrid.best_estimator_
    else:
        gaussianProcess = scoring_function.get_model(kernel, alpha, n_restarts_optimizer=9)
        gaussianProcess.fit(Xtr, ytr)

        gp = gaussianProcess
        if check_fit:
            y_pred = gaussianProcess.predict(Xtr, ytr, return_mdl=False)
            score = r2_score(ytr, y_pred)
            assert score >= .5

    if show_plt:
        predictions = gp.predict(Xtr, ytr, return_mdl=False)
        plt.scatter(Xtr, ytr, label=" Values", linewidth=.2, marker=".")
        plt.scatter(Xtr, predictions, label=" Predict", linewidth=.4, marker="+")
    if show_plt: plt.legend()
    return gp
