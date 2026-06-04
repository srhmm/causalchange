from __future__ import annotations

import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigTemporal
from causalchange.core.types import DataMode, GraphSearch, ScoreType
from causalchange.discovery.scm_clustering import SpaceTimeClustering
from causalchange.domain.temporal import TimeGrid


def test_spacetime_clustering_initial_partitions_without_testing():
    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
    )
    clustering = SpaceTimeClustering(cfg)
    panel = TimeGrid(
        datasets={0: pd.DataFrame({"x": [1, 2, 3, 4]})},
        variables=["x"],
    )

    result = clustering.fit_predict(panel=panel, changepoints=[])

    assert result.contexts == {"x": {0: 0}}
    assert result.regimes == {"x": {0: 0}}
    assert result.diagnostics["mode"] == "initial"
