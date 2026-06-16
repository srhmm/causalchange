from __future__ import annotations

import numpy as np
import pandas as pd

from causalchange.config.causal_change_config import (
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
from causalchange.core.types import DataMode, GraphSearch, ScoreType
from causalchange.scoring.tabular import SCMScoreTabular
from causalchange.scoring.temporal import SCMScoreTemporal


def test_tabular_linear_score_improves_with_true_parent():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
    )
    scorer = SCMScoreTabular(cfg)

    x = np.linspace(-2.0, 2.0, 30)
    y = 2.0 * x + 0.01 * np.sin(np.arange(len(x)))
    X = pd.DataFrame({"x": x, "y": y})

    scorer.fit(X)

    no_parent = scorer.local_score(X, "y", ())
    with_parent = scorer.local_score(X, "y", ("x",))

    assert np.isfinite(no_parent)
    assert np.isfinite(with_parent)
    assert scorer.transition_gain(no_parent, with_parent) > 0


def test_tabular_scorer_cache_safe_for_repeated_scores():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
    )
    scorer = SCMScoreTabular(cfg)

    X = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 20),
            "y": np.linspace(0.0, 2.0, 20),
        }
    )

    scorer.fit(X)

    score1 = scorer.local_score(X, "y", ("x",))
    score2 = scorer.local_score(X, "y", ("x",))

    assert score1 == score2


def test_temporal_build_design_aligns_lagged_columns():
    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        tau_max=2,
    )
    scorer = SCMScoreTemporal(cfg=cfg)

    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    Z = scorer.build_design(X)

    assert Z.columns.tolist() == ["x_lag0", "x_lag1", "x_lag2"]
    assert Z["x_lag0"].tolist() == [3.0, 4.0]
    assert Z["x_lag1"].tolist() == [2.0, 3.0]
    assert Z["x_lag2"].tolist() == [1.0, 2.0]


def test_temporal_linear_score_improves_with_true_lagged_parent():
    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        tau_max=1,
    )
    scorer = SCMScoreTemporal(cfg=cfg)

    x = np.linspace(0.0, 4.0, 40)
    y = np.r_[0.0, 2.0 * x[:-1]]
    X = pd.DataFrame({"x": x, "y": y})

    scorer.fit(X)

    no_parent = scorer.local_score(X, ("y", 0), ())
    with_parent = scorer.local_score(X, ("y", 0), (("x", 1),))

    assert np.isfinite(no_parent)
    assert np.isfinite(with_parent)
    assert scorer.transition_gain(no_parent, with_parent) > 0
