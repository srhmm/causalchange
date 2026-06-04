from __future__ import annotations

import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigTemporal
from causalchange.core.types import DataMode, GraphSearch, ScoreType
from causalchange.scoring.temporal import SCMScoreTemporal


def test_temporal_scorer_builds_lagged_design():
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
    assert len(Z) == 2
    assert Z["x_lag0"].tolist() == [3.0, 4.0]
    assert Z["x_lag2"].tolist() == [1.0, 2.0]
