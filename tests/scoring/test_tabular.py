from __future__ import annotations

import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigTabular
from causalchange.core.types import DataMode, GraphSearch, ScoreType
from causalchange.scoring.tabular import SCMScoreTabular


def test_tabular_scorer_basic_state_without_fitting_regression():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
    )
    scorer = SCMScoreTabular(cfg=cfg)

    X = pd.DataFrame({0: [1.0, 2.0], "y": [3.0, 4.0]})
    X_str = scorer._stringify_columns(X)

    assert X_str.columns.tolist() == ["0", "y"]
    assert scorer._df_key(X_str) == (id(X_str), ("0", "y"), 2)
    assert scorer.transition_gain(10.0, 7.0) == 3.0
