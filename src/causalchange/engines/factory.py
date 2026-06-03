from __future__ import annotations

from causalchange.config.causal_change_config import CausalChangeConfigTabular
from causalchange.config.types import GraphSearch, ContextMode
from causalchange.discovery.changepoint_detection import ChangepointDetection
from causalchange.discovery.context_combination import CHAINContextCombination, LINCContextCombination, SkipCombination
from causalchange.discovery.graphsearch_tabular import GlobeSearch, TopicSearch
from causalchange.discovery.graphsearch_temporal import TemporalGlobeSearch, TemporalTopicSearch
from causalchange.discovery.scm_clustering import SpaceTimeClustering
from causalchange.domain.context import MultiContextDomain, SingleContextDomain
from causalchange.domain.tabular import TabularDomain
from causalchange.domain.temporal import TemporalDomain
from causalchange.scoring.tabular import SCMScoreTabular
from causalchange.scoring.temporal import SCMScoreTemporal
from causalchange.engines.temporal import TemporalDiscoveryEngine
from causalchange.engines.tabular import TabularDiscoveryEngine



class EngineFactory:
    """ shows the high-level control flow for causal discovery. """
    # factory = converts a config to a discovery engine
    @staticmethod
    def from_config(cfg: CausalChangeConfigTabular):
        if cfg.data_mode.is_temporal():
            return EngineFactory._make_temporal_engine(cfg)

        return EngineFactory._make_tabular_engine(cfg)

    @staticmethod
    def _make_tabular_engine(cfg: CausalChangeConfigTabular) -> TabularDiscoveryEngine:
        domain = TabularDomain()
        scoring = SCMScoreTabular(cfg=cfg)
        search = TopicSearch(scoring=scoring) if cfg.graph_search == GraphSearch.TOPIC else GlobeSearch()

        context_preprocessing = (
            MultiContextDomain(context_col=cfg.context_col) if cfg.data_mode.is_context() else SingleContextDomain()
        )
        context_combination = (
            LINCContextCombination(
                grouping=cfg.grouping,
                higher_is_better=scoring.higher_is_better,
            )
            if cfg.aggregation == ContextMode.LINC
            else (CHAINContextCombination(cfg=cfg) if cfg.aggregation == ContextMode.CHAIN else SkipCombination())
        )

        return TabularDiscoveryEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            context_preprocessing=context_preprocessing,
            scoring=scoring,
            context_combination=context_combination,
            search=search,
        )

    @staticmethod
    def _make_temporal_engine(cfg: CausalChangeConfigTabular) -> TemporalDiscoveryEngine:
        if cfg.spacetime is None:
            raise ValueError("spacetime config is required for temporal data.")

        domain = TemporalDomain(tau_max=cfg.spacetime.tau_max)
        scoring = SCMScoreTemporal(cfg=cfg)

        search = (
            TemporalTopicSearch(scoring=scoring)
            if cfg.graph_search == GraphSearch.TOPIC
            else TemporalGlobeSearch(scoring=scoring)
        )

        changepoint_detection = ChangepointDetection(cfg.spacetime)
        scm_clustering = SpaceTimeClustering(cfg.spacetime)

        return TemporalDiscoveryEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            scoring=scoring,
            search=search,
            changepoint_detection=changepoint_detection,
            scm_clustering=scm_clustering,
            cfg=cfg,
        )
