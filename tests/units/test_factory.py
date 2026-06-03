from engines.engine import TemporalDiscoveryEngine
from engines.factory import EngineFactory
from engines.pipeline import TabularDiscoveryEngine

from causalchange.config.cc_config import CausalChangeConfigTabular, CausalChangeConfigTime, ChangepointMode
from causalchange.config.cc_types import ContextMode, DataMode, GraphSearch, ScoreType


def test_factory_routes_iid_to_tabular_engine():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextMode.SKIP,
    )

    engine = EngineFactory.from_config(cfg)

    assert isinstance(engine, TabularDiscoveryEngine)


def test_factory_routes_contexts_to_tabular_engine():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextMode.CHAIN,
        context_col="context",
    )

    engine = EngineFactory.from_config(cfg)

    assert isinstance(engine, TabularDiscoveryEngine)


def test_factory_routes_time_to_spacetime_engine():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        aggregation=ContextMode.SKIP,
        spacetime=CausalChangeConfigTime(
            tau_max=2,
            changepoints=ChangepointMode.NONE,
        ),
    )

    engine = EngineFactory.from_config(cfg)

    assert isinstance(engine, TemporalDiscoveryEngine)


def test_factory_routes_time_contexts_to_spacetime_engine():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TIME_CONTEXTS,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        aggregation=ContextMode.SKIP,
        context_col="context",
        spacetime=CausalChangeConfigTime(
            tau_max=2,
            changepoints=ChangepointMode.NONE,
        ),
    )

    engine = EngineFactory.from_config(cfg)

    assert isinstance(engine, TemporalDiscoveryEngine)
