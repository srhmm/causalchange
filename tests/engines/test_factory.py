from __future__ import annotations

from causalchange.config.causal_change_config import CausalChangeConfigTabular, CausalChangeConfigTemporal
from causalchange.core.types import DataMode, GraphSearch, ScoreType
from causalchange.engines.factory import EngineFactory
from causalchange.engines.tabular import TabularDiscoveryEngine
from causalchange.engines.temporal import TemporalDiscoveryEngine


def test_engine_factory_builds_tabular_engine():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
    )

    engine = EngineFactory.from_config(cfg)

    assert isinstance(engine, TabularDiscoveryEngine)


def test_engine_factory_builds_temporal_engine():
    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
    )

    engine = EngineFactory.from_config(cfg)

    assert isinstance(engine, TemporalDiscoveryEngine)
