from __future__ import annotations

import numpy as np
import pandas as pd

from causalchange.causal_change import CausalChange
from causalchange.config.cc_types import ContextAggregation, DataMode, GPType, GraphSearch, ScoreType


def test_topic_iid_smoke():
    n = 40
    X = pd.DataFrame(
        {
            "x0": [float(i) for i in range(n)],
            "x1": [float(i) + 0.1 for i in range(n)],
            "x2": [float(i % 2) for i in range(n)],
        }
    )

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert set(cc.graph_.nodes()) == {"x0", "x1", "x2"}
    assert cc.result_ is not None
    assert isinstance(cc.result_.topological_order, list)


def test_topic_result_has_edge_strengths():
    n = 40
    X = pd.DataFrame(
        {
            "x0": [float(i) for i in range(n)],
            "x1": [float(i) + 0.1 for i in range(n)],
        }
    )

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
    ).fit(X)

    assert cc.result_ is not None
    assert isinstance(cc.result_.edge_strengths, dict)
    assert set(cc.result_.edge_strengths).issubset(set(cc.graph_.edges()))


def test_topic_linc_contexts_smoke():
    n = 30
    X = pd.DataFrame(
        {
            "context": ["a"] * n + ["b"] * n,
            "x0": [float(i) for i in range(n)] * 2,
            "x1": [float(i) + 0.1 for i in range(n)] + [float(i) + 0.2 for i in range(n)],
            "x2": [float(i % 2) for i in range(n)] * 2,
        }
    )

    cc = CausalChange(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.LINC,
        context_col="context",
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert set(cc.graph_.nodes()) == {"x0", "x1", "x2"}
    assert cc.result_ is not None
    assert isinstance(cc.result_.edge_strengths, dict)


def test_topic_rff_smoke():
    rng = np.random.default_rng(42)
    n = 60

    x0 = rng.normal(size=n)
    x1 = np.tanh(x0) + rng.normal(scale=0.2, size=n)

    X = pd.DataFrame(
        {
            "x0": x0,
            "x1": x1,
        }
    )

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=GPType.FOURIER,
        aggregation=ContextAggregation.SKIP,
        score_kwargs={
            "D": 32,
            "restarts": 1,
            "refine": False,
            "seed": 42,
        },
    ).fit(X)

    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert set(cc.graph_.nodes()) == {"x0", "x1"}
    assert cc.result_ is not None