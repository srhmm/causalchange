from __future__ import annotations

import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigTemporal
from causalchange.core.results import GridCell
from causalchange.core.types import DataMode, GraphSearch, ScoreType
from causalchange.discovery.scm_clustering import TemporalSCMClustering
from causalchange.domain.temporal import TimeGrid


def test_spacetime_clustering_initial_partitions_without_testing():
    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
    )
    clustering = TemporalSCMClustering(cfg)
    panel = TimeGrid(
        datasets={0: pd.DataFrame({"x": [1, 2, 3, 4]})},
        variables=["x"],
    )

    result = clustering.fit_predict(panel=panel, changepoints=[])

    assert result.intervals_by_context == {0: [(0, 4)]}
    assert result.cell_clusters == {
        "x": {
            GridCell(dataset_id=0, interval_id=0): 0,
        }
    }
    assert result.cluster_for(target="x", dataset_id=0, interval_id=0) == 0
    assert result.diagnostics["mode"] == "skip"
