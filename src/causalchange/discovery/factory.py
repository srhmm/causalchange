# causalchange/discovery/factory.py

from __future__ import annotations

from causalchange.config.cc_config import CausalChangeConfig
from causalchange.config.cc_types import ContextAggregation, GraphSearch
from causalchange.discovery.domain.multi import MultiContextDomain
from causalchange.discovery.domain.single import SingleContextDomain
from causalchange.discovery.domain.tabular import TabularDomain
from causalchange.discovery.domain.temporal import TemporalDomain
from causalchange.discovery.pipeline import DiscoveryEngine
from causalchange.discovery.scoring.edge_score_tabular import EdgeScoreTabular
from causalchange.discovery.scoring.edge_score_temporal import EdgeScoreTemporal
from causalchange.discovery.search.globe import GlobeSearch
from causalchange.discovery.search.topic import TopicSearch
from causalchange.discovery.search_multi.chain import ChainAggregator
from causalchange.discovery.search_multi.linc import LINCAggregator


class PipelineFactory:
    @staticmethod
    def from_config(cfg: CausalChangeConfig) -> DiscoveryEngine:
        # decide data domain, continuous/tabular data or time series, multi-context or not
        domain = TabularDomain() if not cfg.data_mode.is_temporal() else TemporalDomain(tau_max=cfg.tau_max)
        scoring = EdgeScoreTabular(cfg=cfg) if not cfg.data_mode.is_temporal() else EdgeScoreTemporal(cfg=cfg)

        # decide search algo
        search = TopicSearch(scoring=scoring) if cfg.graph_search == GraphSearch.TOPIC else GlobeSearch()

        # stuff for multiple contexts
        context_preproc = (
            MultiContextDomain(context_col=cfg.context_col) if cfg.data_mode.is_context() else SingleContextDomain()
        )
        context_aggregation = (
            LINCAggregator(grouping=cfg.grouping, higher_is_better=scoring.higher_is_better)
            if cfg.aggregation == ContextAggregation.LINC
            else ChainAggregator(cfg=cfg)
        )

        return DiscoveryEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            context_preproc=context_preproc,
            scoring=scoring,
            aggregation=context_aggregation,
            search=search,
        )
