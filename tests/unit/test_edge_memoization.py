import numpy as np
import pytest

from src.causalchange.dag.edge_memoized import EdgeMemoized
from src.causalchange.cc_types import DataMode, ScoreType, GPType, CIType
from src.causalchange.scoring.fit_cond_mixture import MixingType
from tests.utils.sample import sample_linear_sem, sample_linear_sem_mixed


# -------------------------------------------------------------------
# Helpers to construct EdgeMemoized in different regimes
# -------------------------------------------------------------------


def _make_edge_iid(X, score_type):
    return EdgeMemoized(
        X=X,
        data_mode=DataMode.IID,
        score_type=score_type,
        mixing_type=MixingType.SKIP,
        lambda_mix=1.0,
        oracle_Z=False,
        oracle_K=False,
    )


def _make_edge_contexts(X_contexts, score_type):
    return EdgeMemoized(
        X=X_contexts,
        data_mode=DataMode.CONTEXTS,
        score_type=score_type,
        mixing_type=MixingType.SKIP,
        lambda_mix=1.0,
        oracle_Z=False,
        oracle_K=False,
    )


def _make_edge_mixed(X, mixing_type, score_type):
    return EdgeMemoized(
        X=X,
        data_mode=DataMode.MIXED,
        score_type=score_type,
        mixing_type=mixing_type,
        lambda_mix=1.0,
        oracle_Z=False,
        oracle_K=False,
        k_max=3,
    )


# -------------------------------------------------------------------
# Score type groups
# -------------------------------------------------------------------

# Score-based types usable with IID / CONTEXTS
SCOREBASED_TYPES = [
    ScoreType.LIN,
    ScoreType.GAM,
    ScoreType.SPLINE,
    ScoreType.KRR,
    GPType.EXACT,
    GPType.FOURIER,
]

# Constraint-based types (currently only CIType.KCI)
CONSTRAINT_TYPES = [
    CIType.KCI,
]


# -------------------------------------------------------------------
# A. score_edge tests
# -------------------------------------------------------------------


@pytest.mark.parametrize("score_type", SCOREBASED_TYPES)
def test_score_edge_iid_true_parent_improves_or_equals_score(score_type):
    """
    For IID data, adding the true parent should not worsen the score.

    We use a simple linear SEM: 0 -> 1 -> 2.
    We check that score(j | pa=[0]) <= score(j | pa=[]).
    """
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    X = sample_linear_sem(adj, n_samples=400, noise_std=0.2, seed=1)

    edge = _make_edge_iid(X, score_type)

    j = 1  # child
    score_empty, _ = edge.score_edge(j, pa=[])
    score_with_parent, _ = edge.score_edge(j, pa=[0])

    assert np.isfinite(score_empty)
    assert np.isfinite(score_with_parent)

    # Lower score is better (e.g., negative log-likelihood / BIC)
    assert score_with_parent <= score_empty + 1e-6


@pytest.mark.parametrize("score_type", SCOREBASED_TYPES)
def test_score_edge_contexts_runs_and_finite(score_type):
    """
    For CONTEXTS data, score_edge should run and give finite scores.

    We use two contexts drawn from the same SEM for simplicity.
    """
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    X0 = sample_linear_sem(adj, n_samples=300, noise_std=0.2, seed=1)
    X1 = sample_linear_sem(adj, n_samples=300, noise_std=0.2, seed=2)
    X_contexts = {0: X0, 1: X1}

    edge = _make_edge_contexts(X_contexts, score_type)

    j = 1
    score_empty, _ = edge.score_edge(j, pa=[])
    score_with_parent, _ = edge.score_edge(j, pa=[0])

    assert np.isfinite(score_empty)
    assert np.isfinite(score_with_parent)


@pytest.mark.parametrize("score_type", CONSTRAINT_TYPES)
def test_score_edge_contexts_constraint_based_runs(score_type):
    """
    For CONTEXTS data with a constraint-based type (e.g. KCI),
    score_edge should call fit_citest_CONTEXTS and return a finite score.
    """
    # simple 2-node system where X0 -> X1
    rng = np.random.default_rng(0)
    D = 200
    X0_0 = rng.normal(size=D)
    X1_0 = X0_0 + rng.normal(scale=0.3, size=D)
    X_ctx0 = np.column_stack([X0_0, X1_0])

    X0_1 = rng.normal(size=D)
    X1_1 = X0_1 + rng.normal(scale=0.3, size=D)
    X_ctx1 = np.column_stack([X0_1, X1_1])

    X_contexts = {0: X_ctx0, 1: X_ctx1}

    edge = _make_edge_contexts(X_contexts, score_type)

    j = 1
    score, res = edge.score_edge(j, pa=[0])

    assert np.isfinite(score)
    assert isinstance(res, dict)


def test_score_edge_caching_iid():
    """
    Calling score_edge with the same (j, pa) twice should hit the cache:
      - only one entry in score_cache
      - returned score/res are identical and same objects
    """
    D, N = 100, 3
    X = np.random.randn(D, N)
    score_type = GPType.EXACT

    edge = _make_edge_iid(X, score_type)

    j, pa = 1, [0]
    score1, res1 = edge.score_edge(j, pa)
    score2, res2 = edge.score_edge(j, pa)

    assert len(edge.score_cache) == 1
    assert score1 == score2
    assert res1 is res2  # same cached object


def test_score_edge_mixed_runs_and_finite():
    """
    For MIXED data, score_edge should run and return finite values.

    We don't assert monotonicity because mixture inference is complex; we just
    ensure basic sanity and routing to fit_fun_MIXED.
    """
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    X_mixed, _ = sample_linear_sem_mixed(adj, n_samples=200, noise_std=0.2, seed=5)

    edge = _make_edge_mixed(
        X=X_mixed,
        mixing_type=MixingType.MIX_LIN,
        score_type=GPType.EXACT,  # score-based
    )

    j = 1
    score_empty, res_empty = edge.score_edge(j, pa=[])
    score_with_parent, res_parent = edge.score_edge(j, pa=[0])

    assert np.isfinite(score_empty)
    assert np.isfinite(score_with_parent)
    assert isinstance(res_empty, dict)
    assert isinstance(res_parent, dict)


# -------------------------------------------------------------------
# B. discrepancy tests
# -------------------------------------------------------------------


def test_discrepancy_contexts_basic():
    """
    For CONTEXTS, discrepancy() should be >= 0 and reflect changes in
    conditional distributions across contexts.

    We construct two contexts with different linear mechanisms:
      ctx0: X1 = X0 + noise
      ctx1: X1 = 2 * X0 + noise
    """
    rng = np.random.default_rng(0)
    D = 300
    N = 2

    # Context 0
    X0_0 = rng.normal(size=D)
    noise0 = rng.normal(scale=0.2, size=D)
    X1_0 = X0_0 + noise0
    X_ctx0 = np.column_stack([X0_0, X1_0])

    # Context 1 (different slope)
    X0_1 = rng.normal(size=D)
    noise1 = rng.normal(scale=0.2, size=D)
    X1_1 = 2.0 * X0_1 + noise1
    X_ctx1 = np.column_stack([X0_1, X1_1])

    X_contexts = {0: X_ctx0, 1: X_ctx1}

    score_type = GPType.EXACT  # score-based, needed for residuals
    edge = EdgeMemoized(
        X=X_contexts,
        data_mode=DataMode.CONTEXTS,
        score_type=score_type,
        mixing_type=MixingType.SKIP,
    )

    j = 1
    pa = [0]

    discrep1, res1 = edge.discrepancy(j, pa)
    discrep2, res2 = edge.discrepancy(j, pa)  # cache hit

    # Allow small negative values from unbiased MMD estimator
    assert discrep1 >= -1e-1
    assert np.isfinite(discrep1)

    # Cache correctness
    assert np.isclose(discrep1, discrep2, atol=1e-12)
    assert res1 is res2
    assert len(edge.discrep_cache) == 1
