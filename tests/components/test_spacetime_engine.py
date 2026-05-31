import pandas as pd
import pytest

from causalchange.causal_change import CausalChange
from causalchange.config.cc_config import ChangepointMode, SpaceTimeConfig
from causalchange.config.cc_types import ContextAggregation, DataMode, GPType, GraphSearch, ScoreType
from causalchange.discovery.search_time.base import TimePanel
from causalchange.discovery.search_time.mechanism_tests import KCIMechanismEqualityTest
from causalchange.discovery.search_time.partitioning import SpaceTimePartitioning


def test_spacetime_globe_stationary():
    n = 30
    X = pd.DataFrame(
        {
            "x0": [float(i) for i in range(n)],
            "x1": [float(i) + 0.1 for i in range(n)],
            "x2": [float(i % 2) for i in range(n)],
        }
    )

    cc = CausalChange(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
        tau_max=2,
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert cc.result_ is not None
    assert cc.graph_.is_directed()
    assert cc.result_.changepoints == []


def test_spacetime_fixed_changepoints():
    n = 40
    X = pd.DataFrame(
        {
            "x0": [float(i) for i in range(n)],
            "x1": [float(i) + 0.1 for i in range(n)],
            "x2": [float(i % 2) for i in range(n)],
        }
    )

    cc = CausalChange(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
        tau_max=2,
        changepoints=ChangepointMode.FIXED,
        fixed_changepoints=[20],
    ).fit(X)

    assert cc.result_.changepoints == [20]
    assert cc.cfg.spacetime is not None
    assert cc.cfg.spacetime.fixed_changepoints == [20]
    assert cc.graph_ is not None


def test_spacetime_detect_changepoints():
    n = 60
    X = pd.DataFrame(
        {
            "x0": [float(i) for i in range(n)],
            "x1": [float(i) + 0.1 if i < 30 else float(2 * i) for i in range(n)],
            "x2": [float(i % 2) for i in range(n)],
        }
    )

    cc = CausalChange(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
        tau_max=2,
        changepoints=ChangepointMode.DETECT,
        d_min=10,
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert isinstance(cc.result_.changepoints, list)


def test_spacetime_globe_gp():
    n = 35
    X = pd.DataFrame(
        {
            "x0": [float(i) for i in range(n)],
            "x1": [float(i) + 0.1 for i in range(n)],
            "x2": [float(i % 2) for i in range(n)],
        }
    )

    cc = CausalChange(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=GPType.EXACT,
        aggregation=ContextAggregation.SKIP,
        tau_max=2,
        score_kwargs={
            "restarts": 2,
            "refine": False,
            "seed": 42,
        },
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert cc.result_ is not None
    assert cc.result_.changepoints == []


def test_spacetime_globe_rff():
    n = 35
    X = pd.DataFrame(
        {
            "x0": [float(i) for i in range(n)],
            "x1": [float(i) + 0.1 for i in range(n)],
            "x2": [float(i % 2) for i in range(n)],
        }
    )

    cc = CausalChange(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=GPType.FOURIER,
        aggregation=ContextAggregation.SKIP,
        tau_max=2,
        score_kwargs={
            "D": 64,
            "restarts": 2,
            "refine": False,
            "seed": 42,
        },
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert cc.result_ is not None


def test_spacetime_contexts_globe_stationary():
    X = pd.DataFrame(
        {
            "context": ["a"] * 20 + ["b"] * 20,
            "x0": [float(i) for i in range(20)] * 2,
            "x1": [float(i) + 0.1 for i in range(20)] * 2,
        }
    )

    cc = CausalChange(
        data_mode=DataMode.TIME_CONTEXTS,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
        context_col="context",
        tau_max=2,
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert cc.result_ is not None
    assert cc.result_.changepoints == []

    assert cc.result_.partitions.contexts["x0"] == {"a": 0, "b": 0}
    assert cc.result_.partitions.contexts["x1"] == {"a": 0, "b": 0}


def test_spacetime_partitions_target_specific():
    n = 30
    X = pd.DataFrame(
        {
            "x0": [float(i) for i in range(n)],
            "x1": [float(i) + 0.1 for i in range(n)],
        }
    )

    cc = CausalChange(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
        tau_max=2,
        changepoints=ChangepointMode.FIXED,
        fixed_changepoints=[15],
    ).fit(X)

    partitions = cc.result_.partitions

    assert set(partitions.contexts) == {"x0", "x1"}
    assert set(partitions.regimes) == {"x0", "x1"}

    assert partitions.contexts["x0"] == {0: 0}
    assert partitions.contexts["x1"] == {0: 0}
    assert partitions.regimes["x0"] == {0: 0, 1: 1}
    assert partitions.regimes["x1"] == {0: 0, 1: 1}


def test_kci_mechanism_test_uses_mmd_without_parents():
    pytest.importorskip("hyppo")

    sample_a = pd.DataFrame(
        {
            "target": [float(i % 5) for i in range(40)],
        }
    )
    sample_b = pd.DataFrame(
        {
            "target": [float(i % 5) for i in range(40)],
        }
    )

    test = KCIMechanismEqualityTest(
        alpha=0.01,
        min_samples=5,
    )

    result = test.same_mechanism(
        sample_a=sample_a,
        sample_b=sample_b,
        target_col="target",
        parent_cols=[],
    )

    assert result.method == "mmd"
    assert 0.0 <= result.pvalue <= 1.0
    assert result.same is True


def test_spacetime_partitioning_detect_regimes_merges_identical_regimes():
    pytest.importorskip("hyppo")

    n = 40
    repeated = [float(i % 5) for i in range(n // 2)] * 2

    X = pd.DataFrame(
        {
            "x0": repeated,
            "x1": repeated,
        }
    )

    panel = TimePanel(
        datasets={0: X},
        variables=["x0", "x1"],
        context_col=None,
    )

    cfg = SpaceTimeConfig(
        tau_max=2,
        changepoints=ChangepointMode.FIXED,
        fixed_changepoints=[20],
        detect_regimes=True,
        detect_contexts=False,
        mechanism_test_alpha=0.01,
    )

    partitioning = SpaceTimePartitioning(cfg)

    partitions = partitioning.fit_predict(
        panel=panel,
        graph=None,
        changepoints=[20],
    )

    assert partitions.diagnostics["detect_regimes"] is True
    assert partitions.diagnostics["n_regimes"] == 2
    assert len(partitions.diagnostics["tests"]) == 2

    assert partitions.regimes["x0"] == {0: 0, 1: 0}
    assert partitions.regimes["x1"] == {0: 0, 1: 0}


def test_spacetime_partitioning_detect_contexts_merges_identical_contexts():
    pytest.importorskip("hyppo")

    n = 30
    X_a = pd.DataFrame(
        {
            "x0": [float(i % 5) for i in range(n)],
            "x1": [float((i + 1) % 5) for i in range(n)],
        }
    )
    X_b = X_a.copy()

    panel = TimePanel(
        datasets={"a": X_a, "b": X_b},
        variables=["x0", "x1"],
        context_col="context",
    )

    cfg = SpaceTimeConfig(
        tau_max=2,
        changepoints=ChangepointMode.NONE,
        detect_contexts=True,
        detect_regimes=False,
        mechanism_test_alpha=0.01,
    )

    partitioning = SpaceTimePartitioning(cfg)

    partitions = partitioning.fit_predict(
        panel=panel,
        graph=None,
        changepoints=[],
    )

    assert partitions.diagnostics["detect_contexts"] is True
    assert partitions.diagnostics["n_contexts"] == 2
    assert len(partitions.diagnostics["tests"]) == 2

    assert partitions.contexts["x0"]["a"] == partitions.contexts["x0"]["b"]
    assert partitions.contexts["x1"]["a"] == partitions.contexts["x1"]["b"]


def test_spacetime_contexts_detect_changepoints():
    n = 60

    x0_a = [float(i) for i in range(n)]
    x1_a = [float(i) + 0.1 if i < 30 else float(2 * i) for i in range(n)]

    x0_b = [float(i) for i in range(n)]
    x1_b = [float(i) + 0.2 if i < 30 else float(2 * i) + 0.2 for i in range(n)]

    X = pd.DataFrame(
        {
            "context": ["a"] * n + ["b"] * n,
            "x0": x0_a + x0_b,
            "x1": x1_a + x1_b,
        }
    )

    cc = CausalChange(
        data_mode=DataMode.TIME_CONTEXTS,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
        context_col="context",
        tau_max=2,
        changepoints=ChangepointMode.DETECT,
        d_min=10,
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert cc.result_ is not None
    assert isinstance(cc.result_.changepoints, list)

    assert "x0" in cc.result_.partitions.contexts
    assert "x1" in cc.result_.partitions.contexts
