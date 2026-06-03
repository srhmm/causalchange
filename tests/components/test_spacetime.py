import pandas as pd
import pytest
from discovery.base import TimeGrid
from discovery.changepoints import SpaceTimeChangepointDetection
from discovery.partitioning import SpaceTimePartitioning
from scoring import EdgeScoreTime
from scoring.mechanism_tests import KCIMechanismEqualityTest

from causalchange.causal_change import CausalChange
from causalchange.config.cc_config import (
    CausalChangeConfigTabular,
    CausalChangeConfigTime,
    ChangepointMode,
    ChangepointScope,
)
from causalchange.config.cc_types import ContextMode, DataMode, GPType, GraphSearch, ScoreType


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
        context_mode=ContextMode.SKIP,
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
        context_mode=ContextMode.SKIP,
        tau_max=2,
        changepoint_mode=ChangepointMode.FIXED,
        fixed_changepoints=[20],
    ).fit(X)

    assert cc.result_.changepoints == [20]
    assert cc.cfg is not None
    assert cc.cfg.fixed_changepoints == [20]
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
        context_mode=ContextMode.SKIP,
        tau_max=2,
        changepoint_mode=ChangepointMode.DETECT,
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
        context_mode=ContextMode.SKIP,
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
        context_mode=ContextMode.SKIP,
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
        context_mode=ContextMode.SKIP,
        context_col="context",
        tau_max=2,
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert cc.result_ is not None
    assert cc.result_.changepoints == []

    assert cc.result_.grid_clusters.contexts["x0"] == {"a": 0, "b": 0}
    assert cc.result_.grid_clusters.contexts["x1"] == {"a": 0, "b": 0}


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
        context_mode=ContextMode.SKIP,
        tau_max=2,
        changepoint_mode=ChangepointMode.FIXED,
        fixed_changepoints=[15],
    ).fit(X)

    partitions = cc.result_.grid_clusters

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

    panel = TimeGrid(
        datasets={0: X},
        variables=["x0", "x1"],
        context_col=None,
    )

    cfg = CausalChangeConfigTime(
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

    panel = TimeGrid(
        datasets={"a": X_a, "b": X_b},
        variables=["x0", "x1"],
        context_col="context",
    )

    cfg = CausalChangeConfigTime(
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
        context_mode=ContextMode.SKIP,
        context_col="context",
        tau_max=2,
        changepoint_mode=ChangepointMode.DETECT,
        d_min=10,
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert cc.result_ is not None
    assert isinstance(cc.result_.changepoints, list)

    assert "x0" in cc.result_.grid_clusters.contexts
    assert "x1" in cc.result_.grid_clusters.contexts


def test_spacetime_contexts_detect_changepoints_auto_penalty_runs():
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
        context_mode=ContextMode.SKIP,
        context_col="context",
        tau_max=2,
        changepoint_mode=ChangepointMode.DETECT,
        d_min=10,
        pelt_penalty="auto",
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert cc.result_ is not None
    assert isinstance(cc.result_.changepoints, list)

    def test_spacetime_partitioning_does_not_merge_by_bad_transitivity():
        partitioning = SpaceTimePartitioning(
            CausalChangeConfigTime(
                tau_max=1,
                changepoints=ChangepointMode.NONE,
            )
        )

        labels = partitioning._cluster_from_pairwise_tests(
            nodes=["A", "B", "C"],
            same_pairs={
                frozenset(("A", "B")),
                frozenset(("B", "C")),
            },
            different_pairs={
                frozenset(("A", "C")),
            },
            pvalues={
                frozenset(("A", "B")): 0.9,
                frozenset(("B", "C")): 0.8,
                frozenset(("A", "C")): 0.01,
            },
        )

        assert labels["A"] == labels["B"]
        assert labels["A"] != labels["C"]


def test_changepoint_detection_per_context_uses_union_grid():
    X_a = pd.DataFrame(
        {
            "x0": [0.0] * 30 + [3.0] * 50,
            "x1": [0.0] * 30 + [2.0] * 50,
        }
    )
    X_b = pd.DataFrame(
        {
            "x0": [0.0] * 50 + [3.0] * 30,
            "x1": [0.0] * 50 + [2.0] * 30,
        }
    )

    panel = TimeGrid(
        datasets={"A": X_a, "B": X_b},
        variables=["x0", "x1"],
        context_col="context",
    )

    cfg = CausalChangeConfigTime(
        tau_max=1,
        changepoints=ChangepointMode.DETECT,
        changepoint_scope=ChangepointScope.PER_CONTEXT,
        d_min=10,
        pelt_penalty=1.0,
    )

    scorer = EdgeScoreTime(
        cfg=CausalChangeConfigTabular(
            data_mode=DataMode.TIME_CONTEXTS,
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType.LIN,
            aggregation=ContextMode.SKIP,
            context_col="context",
            spacetime=cfg,
        )
    )
    scorer.fit_panel(panel)

    detector = SpaceTimeChangepointDetection(cfg)

    changepoints = detector.detect(
        time_grid=panel,
        graph=None,
        scorer=scorer,
        variables=["x0", "x1"],
    )

    assert isinstance(changepoints, list)
    assert detector.diagnostics_["scope"] == ChangepointScope.PER_CONTEXT.value
    assert "A" in detector.diagnostics_["by_context"]
    assert "B" in detector.diagnostics_["by_context"]

    # Exact PELT locations may shift by tau_max / signal construction,
    # so assert approximate recovery.
    assert any(abs(cp - 30) <= 2 for cp in detector.diagnostics_["by_context"]["A"])
    assert any(abs(cp - 50) <= 2 for cp in detector.diagnostics_["by_context"]["B"])
    assert any(abs(cp - 30) <= 2 for cp in changepoints)
    assert any(abs(cp - 50) <= 2 for cp in changepoints)
