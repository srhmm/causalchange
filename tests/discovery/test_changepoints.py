from __future__ import annotations

from causalchange.config.causal_change_config import CausalChangeConfigTemporal
from causalchange.core.types import ChangepointMode, DataMode, GraphSearch, ScoreType
from causalchange.discovery.changepoints import ChangepointDetection


def test_changepoint_detection_skip_mode_returns_empty_list():
    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        changepoint_mode=ChangepointMode.SKIP,
    )
    detector = ChangepointDetection(cfg)

    assert detector.detect() == []
    assert detector.diagnostics_["mode"] == "none"
