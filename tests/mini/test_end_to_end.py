from __future__ import annotations

import numpy as np
import pandas as pd

from causalchange.causal_change import CausalChange
from causalchange.config.cc_types import DataMode, GraphSearch, ScoreType, ContextAggregation


def _linear_chain_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = 2.0 * x0 + rng.normal(scale=0.1, size=n)
    x2 = 2.0 * x1 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


def test_causalchange_fit_discovers_single_context_graph_and_returns_dag_like_object():
    df = _linear_chain_df(2000, seed=1)
    est = CausalChange(data_mode=DataMode.IID, graph_search=GraphSearch.TOPIC, score_type=ScoreType.LIN)
    dag = est.fit(df)

    nodes = list(dag.nodes())
    edges = list(dag.edges())
    assert set(nodes) == {"X0", "X1", "X2"}
    assert len(edges) > 0


def test_causalchange_fit_multi_context_does_not_treat_context_col_as_variable():
    df0 = _linear_chain_df(1000, seed=2)
    df1 = _linear_chain_df(1000, seed=3)
    df0["context"] = 0
    df1["context"] = 1
    df = pd.concat([df0, df1], ignore_index=True)

    est = CausalChange(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.LINC,
        context_col="context",
    )
    dag = est.fit(df)
    assert "context" not in set(dag.nodes())
