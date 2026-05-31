from causalchange.config.cc_config import CausalChangeConfig, ChangepointMode, SpaceTimeConfig
from causalchange.config.cc_types import ContextAggregation, DataMode, GraphSearch, ScoreType
from causalchange.discovery.factory import PipelineFactory
from causalchange.discovery.pipeline import TabularDiscoveryEngine
from causalchange.discovery.search_time.engine import SpaceTimeEngine


def test_factory_routes_iid_to_tabular_engine():
    cfg = CausalChangeConfig(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
    )

    engine = PipelineFactory.from_config(cfg)

    assert isinstance(engine, TabularDiscoveryEngine)


def test_factory_routes_contexts_to_tabular_engine():
    cfg = CausalChangeConfig(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.CHAIN,
        context_col="context",
    )

    engine = PipelineFactory.from_config(cfg)

    assert isinstance(engine, TabularDiscoveryEngine)


def test_factory_routes_time_to_spacetime_engine():
    cfg = CausalChangeConfig(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
        spacetime=SpaceTimeConfig(
            tau_max=2,
            changepoints=ChangepointMode.NONE,
        ),
    )

    engine = PipelineFactory.from_config(cfg)

    assert isinstance(engine, SpaceTimeEngine)


def test_factory_routes_time_contexts_to_spacetime_engine():
    cfg = CausalChangeConfig(
        data_mode=DataMode.TIME_CONTEXTS,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
        context_col="context",
        spacetime=SpaceTimeConfig(
            tau_max=2,
            changepoints=ChangepointMode.NONE,
        ),
    )

    engine = PipelineFactory.from_config(cfg)

    assert isinstance(engine, SpaceTimeEngine)
