from __future__ import annotations

from causalchange.config.causal_change_config import (
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
from causalchange.core.types import (
    ChangepointMethod,
    ChangepointMode,
    ChangepointScope,
    DataMode,
    GraphSearch,
    MechanismClusteringMethod,
    MechanismClusteringScope,
    MixedSCMType,
    ScoreType,
    StatisticalTestingMethod,
    TabularContextMode,
)
from causalchange.discovery.graph_tabular import GraphSearchTabularTopological
from causalchange.discovery.graph_temporal import GraphSearchTemporalGreedy
from causalchange.engines.factory import EngineFactory
from causalchange.engines.tabular import TabularDiscoveryEngine
from causalchange.engines.temporal import TemporalDiscoveryEngine
from causalchange.scoring.regression_mixtures import SCMScoreCMM
from causalchange.scoring.tabular import SCMScoreTabular
from causalchange.scoring.temporal import SCMScoreTemporal


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


def test_factory_wires_topic_pipeline():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        context_mode=TabularContextMode.SKIP,
        mix_type=MixedSCMType.SKIP,
    )

    engine = EngineFactory.from_config(cfg)

    assert isinstance(engine, TabularDiscoveryEngine)
    assert isinstance(engine.search, GraphSearchTabularTopological)
    assert isinstance(engine.scoring, SCMScoreTabular)
    assert not isinstance(engine.scoring, SCMScoreCMM)


def test_factory_wires_cmm_pipeline_to_cmm_scorer():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        context_mode=TabularContextMode.SKIP,
        mix_type=MixedSCMType.LIN,
        score_kwargs={"k_max": 2},
    )

    engine = EngineFactory.from_config(cfg)

    assert isinstance(engine, TabularDiscoveryEngine)
    assert isinstance(engine.search, GraphSearchTabularTopological)
    assert isinstance(engine.scoring, SCMScoreCMM)


def test_factory_wires_temporal_pipeline():
    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        tau_max=1,
        changepoint_mode=ChangepointMode.SKIP,
        changepoint_scope=ChangepointScope.SKIP,
        changepoint_method=ChangepointMethod.SKIP,
        clustering_scope=MechanismClusteringScope.SKIP,
        clustering_method=MechanismClusteringMethod.SKIP,
        testing_method=StatisticalTestingMethod.SKIP,
    )

    engine = EngineFactory.from_config(cfg)

    assert isinstance(engine, TemporalDiscoveryEngine)
    assert isinstance(engine.search, GraphSearchTemporalGreedy)
    assert isinstance(engine.scoring, SCMScoreTemporal)
