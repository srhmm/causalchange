# causalchange/discovery/factory.py

from __future__ import annotations

from causalchange.config.cc_config import CausalChangeConfig
from causalchange.config.cc_types import ContextAggregation, GraphSearch
from causalchange.discovery.domain.multi import MultiContextDomain
from causalchange.discovery.domain.single import SingleContextDomain
from causalchange.discovery.domain.tabular import TabularDomain
from causalchange.discovery.domain.time import TimeDomain
from causalchange.discovery.pipeline import TabularDiscoveryEngine
from causalchange.discovery.scoring.edge_score_tabular import EdgeScoreTabular
from causalchange.discovery.scoring.edge_score_time import EdgeScoreTime
from causalchange.discovery.search.globe import GlobeSearch
from causalchange.discovery.search.temporal_globe import TemporalGlobeSearch
from causalchange.discovery.search.temporal_topic import TemporalTopicSearch
from causalchange.discovery.search.topic import TopicSearch
from causalchange.discovery.search_multi.chain import ChainAggregator
from causalchange.discovery.search_multi.linc import LINCAggregator
from causalchange.discovery.search_multi.none import NoAggregation
from causalchange.discovery.search_time.changepoints import SpaceTimeChangepointDetection
from causalchange.discovery.search_time.engine import SpaceTimeEngine
from causalchange.discovery.search_time.partitioning import SpaceTimePartitioning


class PipelineFactory:
    @staticmethod
    def from_config(cfg: CausalChangeConfig):
        if cfg.data_mode.is_temporal():
            return PipelineFactory._make_spacetime_engine(cfg)

        return PipelineFactory._make_tabular_engine(cfg)

    @staticmethod
    def _make_tabular_engine(cfg: CausalChangeConfig) -> TabularDiscoveryEngine:
        domain = TabularDomain()
        scorer = EdgeScoreTabular(cfg=cfg)

        search = TopicSearch(scoring=scorer) if cfg.graph_search == GraphSearch.TOPIC else GlobeSearch()

        context_preproc = (
            MultiContextDomain(context_col=cfg.context_col) if cfg.data_mode.is_context() else SingleContextDomain()
        )

        aggregation = (
            LINCAggregator(
                grouping=cfg.grouping,
                higher_is_better=scorer.higher_is_better,
            )
            if cfg.aggregation == ContextAggregation.LINC
            else (ChainAggregator(cfg=cfg) if cfg.aggregation == ContextAggregation.CHAIN else NoAggregation())
        )

        return TabularDiscoveryEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            context_preproc=context_preproc,
            scoring=scorer,
            aggregation=aggregation,
            search=search,
        )

    @staticmethod
    def _make_spacetime_engine(cfg: CausalChangeConfig) -> SpaceTimeEngine:
        if cfg.spacetime is None:
            raise ValueError("spacetime config is required for temporal data.")

        domain = TimeDomain(tau_max=cfg.spacetime.tau_max)
        scorer = EdgeScoreTime(cfg=cfg)

        search = (
            TemporalTopicSearch(scoring=scorer)
            if cfg.graph_search == GraphSearch.TOPIC
            else TemporalGlobeSearch(scoring=scorer)
        )

        changepoint_detection = SpaceTimeChangepointDetection(cfg.spacetime)
        partitioning = SpaceTimePartitioning(cfg.spacetime)

        return SpaceTimeEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            scoring=scorer,
            search=search,
            changepoint_detection=changepoint_detection,
            partitioning=partitioning,
            cfg=cfg,
        )
