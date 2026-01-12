from __future__ import annotations

import numpy as np
import pandas as pd

from causalchange.config.cc_types import ContextAggregation, DataMode, GraphSearch, ScoreType
from causalchange.config.cc_config import CausalChangeConfig
from causalchange.discovery.scoring.edge_score_tabular import EdgeScoreTabular
from causalchange.discovery.scoring.edge_score import EdgeScore
from causalchange.discovery.search.topic import TopicSearch


def _make_df(seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = 2.0 * x0 + 0.1 * rng.normal(size=n)
    x2 = -1.0 * x1 + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


def test_edge_score_tabular_rebinds_per_dataframe_and_avoids_cache_cross_talk():
    cfg = CausalChangeConfig(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        aggregation=ContextAggregation.SKIP,
        score_type=ScoreType.LIN,
    )
    scoring = EdgeScoreTabular(cfg=cfg)

    df1 = _make_df(0)
    df2 = _make_df(1)

    s1 = scoring.score_edge(df1, "X2", ("X1",))
    s2 = scoring.score_edge(df2, "X2", ("X1",))

    # scores should reflect different datasets (and therefore differ)
    assert s1 != s2

    # and they should match what you'd get by fitting a fresh EdgeScore per df
    e1 = EdgeScore(data_mode=DataMode.IID, score_type=ScoreType.LIN)
    e1.fit(df1.to_numpy(float))
    j = {c: i for i, c in enumerate(df1.columns)}
    direct1 = float(e1.score_edge(j=j["X2"], pa=[j["X1"]], ret_full_result=False))

    e2 = EdgeScore(data_mode=DataMode.IID, score_type=ScoreType.LIN)
    e2.fit(df2.to_numpy(float))
    j2 = {c: i for i, c in enumerate(df2.columns)}
    direct2 = float(e2.score_edge(j=j2["X2"], pa=[j2["X1"]], ret_full_result=False))

    assert s1 == direct1
    assert s2 == direct2


def test_topic_search_produces_acyclic_graph_and_history_length_matches_candidates():
    class DummyScoring:
        higher_is_better = False  # lower score is better

        def transition_gain(self, old_score: float, new_score: float) -> float:
            # improvement means decreased score
            return float(old_score - new_score)

        def score_significant(self, gain: float) -> bool:
            return gain > 0.0

        def score_is_better(self, a: float, b: float) -> bool:
            return a < b

    def allowed_edge(a, b) -> bool:
        return a != b

    def score_fun(effect, parents) -> float:
        # prefer fewer parents (simple, deterministic score)
        return float(len(parents))

    search = TopicSearch(scoring=DummyScoring())

    nodes = ["X0", "X1", "X2", "X3"]
    res = search.run(nodes=nodes, candidates=list(nodes), allowed_edge=allowed_edge, score_fun=score_fun)
    assert set(res.graph.nodes()) == set(nodes)
    assert res.graph.number_of_nodes() == 4
    assert set(res.topological_order) == set(nodes)
    assert len(res.history) == 4
