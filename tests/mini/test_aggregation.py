from __future__ import annotations

import pandas as pd

from causalchange.config.cc_config import CausalChangeConfig
from causalchange.config.cc_types import (
    ContextAggregation,
    DataMode,
    GraphSearch,
    ScoreType,
)
from causalchange.discovery.search_multi.chain import ChainAggregator
from causalchange.discovery.search_multi.linc import LINCAggregator, LINCGroupingParams


def test_chain_aggregator_sums_context_scores_and_returns_diagnostics():
    contexts = {
        "a": pd.DataFrame({"X0": [0, 1, 2]}),
        "b": pd.DataFrame({"X0": [0, 1]}),
    }

    def score_ctx(df: pd.DataFrame) -> float:
        return float(len(df))

    cfg = CausalChangeConfig(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.CHAIN,
    )
    agg = ChainAggregator(cfg=cfg)
    res = agg.aggregate(contexts=contexts, effect="X0", parents=(), score_ctx=score_ctx)
    assert res.diagnostics["fit"] == 5.0
    assert res.total == 5.0


def test_linc_aggregator_components_does_not_merge_when_gain_not_significant():
    contexts = {
        0: pd.DataFrame({"X0": [0, 1, 2, 3]}),
        1: pd.DataFrame({"X0": [4, 5, 6, 7]}),
    }

    def score_ctx(df: pd.DataFrame) -> float:
        return float(len(df))

    grouping = LINCGroupingParams(method="components", gain_threshold=0.1)
    agg = LINCAggregator(grouping=grouping, higher_is_better=False)

    res = agg.aggregate(contexts=contexts, effect="X0", parents=(), score_ctx=score_ctx)

    # old_score = 4 + 4, pooled_score = 8 -> gain = old - new = 0
    assert res.diagnostics["edges"] == []
    assert sorted([sorted(g) for g in res.diagnostics["groups"]]) == [[0], [1]]
    assert res.total == 8.0


def test_linc_aggregator_agglomerative_does_not_merge_when_gain_not_significant():
    contexts = {
        "c0": pd.DataFrame({"X0": [0, 1, 2]}),
        "c1": pd.DataFrame({"X0": [3, 4, 5]}),
        "c2": pd.DataFrame({"X0": [6, 7, 8]}),
    }

    def score_ctx(df: pd.DataFrame) -> float:
        return float(len(df))

    grouping = LINCGroupingParams(method="agglomerative", gain_threshold=0.1)
    agg = LINCAggregator(grouping=grouping, higher_is_better=False)

    res = agg.aggregate(contexts=contexts, effect="X0", parents=(), score_ctx=score_ctx)
    assert len(res.diagnostics["groups"]) == 3
    assert res.total == 9.0
