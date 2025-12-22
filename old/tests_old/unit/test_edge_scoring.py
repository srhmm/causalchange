import numpy as np
import pytest

from causalchange._cc_types import DataMode, ScoreType, GPType, CIType
from causalchange.old.dag.edge_memoized import EdgeMemoized, discrepancy_from_resid
from causalchange.old.scoring.fit_cond_mixture import MixingType
from causalchange.old.scoring.fit_score import (
    fit_score_functional_model,
    fit_score_ln,
    fit_resid_CONTEXTS,
    fit_resid_TIME,
    fit_resid_TIME_CONTEXTS,
)
from causalchange.old.search import (
    fit_fun_CONTEXTS,
    fit_citest_CONTEXTS,
    fit_fun_MIXED,
)
from tests.utils.sample import (
    sample_linear_sem,
    sample_linear_sem_mixed,
    sample_time_series,
)



def _get_score_fun_for(X, data_mode: DataMode, score_type):
    edge = EdgeMemoized(
        X=X,
        data_mode=data_mode,
        score_type=score_type,
        mixing_type=MixingType.SKIP,
        lambda_mix=1.0,
        oracle_Z=False,
        oracle_K=False,
    )
    score_fun = edge.get_score_fun()
    if hasattr(score_type, "is_scorebased") and score_type.is_scorebased():
        assert score_fun is not None, f"Expected a score_fun for {score_type}"
    return score_fun


SCOREBASED_TYPES = [
    ScoreType.LIN,
    GPType.EXACT,
]

CONSTRAINT_TYPES = [CIType.KCI]



@pytest.mark.parametrize("score_type", SCOREBASED_TYPES)
def test_fit_fun_iid_basic(score_type):
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    X = sample_linear_sem(adj, n_samples=300, noise_std=0.2, seed=1)
    D, N = X.shape

    score_fun = _get_score_fun_for(X, DataMode.IID, score_type)

    j = 1
    pa = [0]

    score, info = fit_score_functional_model(X, pa=pa, target=j, score_fun=score_fun)
    assert np.isscalar(score)
    assert np.isfinite(score)
    assert isinstance(info, dict)
    assert "model" in info


@pytest.mark.parametrize("score_type", SCOREBASED_TYPES)
def test_fit_fun_iid_with_residuals(score_type):
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    X = sample_linear_sem(adj, n_samples=200, noise_std=0.2, seed=2)
    D, N = X.shape

    score_fun = _get_score_fun_for(X, DataMode.IID, score_type)

    j = 1
    pa = [0]

    resid, info = fit_score_functional_model(X, pa=pa, target=j, score_fun=score_fun, ret_residuals=True)
    resid = np.asarray(resid)
    assert resid.shape[0] == D
    assert isinstance(info, dict)
    assert "model" in info


def test_fit_fun_iid_empty_parents():
    D, N = 150, 3
    X = np.random.randn(D, N)

    score_fun = _get_score_fun_for(X, DataMode.IID, ScoreType.LIN)

    j = 1
    pa = []

    score, info = fit_score_functional_model(X, pa=pa, target=j, score_fun=score_fun)
    assert np.isscalar(score)
    assert np.isfinite(score)
    assert isinstance(info, dict)
    assert "model" in info




@pytest.mark.parametrize("score_type", SCOREBASED_TYPES)
def test_fit_fun_contexts_basic(score_type):
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    X0 = sample_linear_sem(adj, n_samples=200, noise_std=0.2, seed=1)
    X1 = sample_linear_sem(adj, n_samples=180, noise_std=0.2, seed=2)
    X_contexts = {0: X0, 1: X1}

    score_fun = _get_score_fun_for(X0, DataMode.CONTEXTS, score_type)

    j = 1
    pa = [0]

    score, info = fit_fun_CONTEXTS(X_contexts, pa=pa, target=j, score_fun=score_fun)
    assert np.isscalar(score)
    assert np.isfinite(score)
    assert isinstance(info, dict)



def test_fit_fun_mixed_basic():
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    X_mixed, _ = sample_linear_sem_mixed(adj, n_samples=200, noise_std=0.2, seed=5)

    j = 1
    pa = [0]
    params = dict(k_max=3, oracle_Z=False, oracle_K=False)

    score, info = fit_fun_MIXED(
        MixingType.MIX_LIN,
        X_mixed,
        covariates=pa,
        target=j,
        resid=None,
        **params,
    )

    assert np.isscalar(score)
    assert np.isfinite(score)
    assert isinstance(info, dict)



@pytest.mark.parametrize("score_type", CONSTRAINT_TYPES)
def test_fit_citest_contexts_basic(score_type):
    rng = np.random.default_rng(0)
    D = 150

    X0_0 = rng.normal(size=D)
    X1_0 = X0_0 + rng.normal(scale=0.3, size=D)
    X_ctx0 = np.column_stack([X0_0, X1_0])

    X0_1 = rng.normal(size=D)
    X1_1 = X0_1 + rng.normal(scale=0.3, size=D)
    X_ctx1 = np.column_stack([X0_1, X1_1])

    X_contexts = {0: X_ctx0, 1: X_ctx1}

    j = 1
    pa = [0]

    score_bits, results = fit_citest_CONTEXTS(X_contexts, pa=pa, target=j)

    assert np.isscalar(score_bits)
    assert np.isfinite(score_bits)
    assert isinstance(results, dict)

    for key in ["contexts", "y_grid", "dy", "cond_density", "entropy_bits_per_context", "labels_pred", "groups"]:
        assert key in results


@pytest.mark.parametrize("with_parents", [True, False])
def test_fit_resid_contexts_basic(with_parents):
    rng = np.random.default_rng(0)
    D0, D1 = 100, 120
    N = 3

    X0 = rng.normal(size=(D0, N))
    X1 = rng.normal(size=(D1, N))
    X_contexts = {0: X0, 1: X1}

    score_fun = fit_score_ln  # simple & fast

    target = 1
    parents = [0] if with_parents else []

    residual_sets = fit_resid_CONTEXTS(X_contexts, target=target, parents=parents, score_fun=score_fun)

    assert isinstance(residual_sets, list)
    assert len(residual_sets) == len(X_contexts)

    for ctx_id, resid in zip(sorted(X_contexts.keys()), residual_sets):
        resid = np.asarray(resid)
        assert resid.shape[0] == X_contexts[ctx_id].shape[0]


def test_fit_resid_time_basic():
    adj = np.array([
        [0, 1],
        [0, 0],
    ])
    X_time = sample_time_series(adj, n_timepoints=200, noise_std=0.2, seed=1)

    score_fun = fit_score_ln
    target = 1
    parents = [0]

    residual_sets = fit_resid_TIME(X_time, target, parents, score_fun, changepoints=[50, 200])

    assert isinstance(residual_sets, list)
    assert len(residual_sets) >= 1

    for resid in residual_sets:
        resid = np.asarray(resid)
        assert resid.ndim == 1
        assert resid.shape[0] > 0


def test_fit_resid_time_contexts_basic():
    adj = np.array([
        [0, 1],
        [0, 0],
    ])
    X0 = sample_time_series(adj, n_timepoints=150, noise_std=0.2, seed=2)
    X1 = sample_time_series(adj, n_timepoints=160, noise_std=0.2, seed=3)
    X_dict = {0: X0, 1: X1}

    score_fun = fit_score_ln
    target = 1
    parents = [0]

    residual_sets = fit_resid_TIME_CONTEXTS(
        X_dict,
        target,
        parents,
        score_fun,
        cp_per_ctx={0: [], 1: []},
    )

    assert isinstance(residual_sets, list)
    assert len(residual_sets) >= 1

    for resid in residual_sets:
        resid = np.asarray(resid)
        assert resid.ndim == 1
        assert resid.shape[0] > 0



def test_discrepancy_from_resid_single_set():
    resid = np.random.randn(100)
    discrep, details = discrepancy_from_resid([resid])

    assert discrep == 0.0
    assert details == {}


def test_discrepancy_from_resid_multiple_sets():
    rng = np.random.default_rng(0)
    r0 = rng.normal(size=100)
    r1 = rng.normal(loc=1.0, scale=1.0, size=100)
    r2 = rng.normal(loc=-1.0, scale=1.0, size=100)

    residual_sets = [r0, r1, r2]

    discrep_sum, details_sum = discrepancy_from_resid(residual_sets, aggregate="sum")
    discrep_avg, details_avg = discrepancy_from_resid(residual_sets, aggregate="avg")


    expected_keys = {(0, 1), (0, 2), (1, 2)}
    assert set(details_sum.keys()) == expected_keys
    assert set(details_avg.keys()) == expected_keys


    assert discrep_sum >= -1e-6
    assert discrep_avg >= -1e-6
    assert np.isfinite(discrep_sum)
    assert np.isfinite(discrep_avg)


    assert discrep_sum >= discrep_avg - 1e-8
    assert discrep_sum > 0 or discrep_avg > 0


def test_discrepancy_from_resid_invalid_aggregate():
    residual_sets = [np.random.randn(50), np.random.randn(50)]
    with pytest.raises(ValueError):
        discrepancy_from_resid(residual_sets, aggregate="invalid")
