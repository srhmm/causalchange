# causalchange/discovery/factory.py

from __future__ import annotations

from causalchange.config.cc_config import CausalChangeConfig
from causalchange.config.cc_types import ContextAggregation, GraphSearch
from causalchange.discovery.domain.multi import MultiContextDomain
from causalchange.discovery.domain.single import SingleContextDomain
from causalchange.discovery.domain.tabular import TabularDomain
from causalchange.discovery.domain.time import TemporalDomain
from causalchange.discovery.pipeline import DiscoveryEngine
from causalchange.discovery.scoring.edge_score_tabular import EdgeScoreTabular
from causalchange.discovery.scoring.edge_score_time import EdgeScoreTemporal
from causalchange.discovery.search.globe import GlobeSearch
from causalchange.discovery.search.topic import TopicSearch
from causalchange.discovery.search_multi.chain import ChainAggregator
from causalchange.discovery.search_multi.linc import LINCAggregator
from causalchange.discovery.search_multi.none import NoAggregation
from causalchange.discovery.search_time.engine import SpaceTimeEngine


def make_domain(cfg: CausalChangeConfig):
    return TemporalDomain(tau_max=cfg.tau_max) if cfg.data_mode.is_temporal() else TabularDomain()


def make_scorer(cfg: CausalChangeConfig):
    return EdgeScoreTemporal(cfg=cfg) if cfg.data_mode.is_temporal() else EdgeScoreTabular(cfg=cfg)


def make_context_preproc(cfg: CausalChangeConfig):
    return MultiContextDomain(context_col=cfg.context_col) if cfg.data_mode.is_context() else SingleContextDomain()


def make_aggregator(cfg: CausalChangeConfig, scorer):
    if cfg.aggregation == ContextAggregation.SKIP:
        return NoAggregation()

    if cfg.aggregation == ContextAggregation.CHAIN:
        return ChainAggregator(cfg=cfg)

    if cfg.aggregation == ContextAggregation.LINC:
        return LINCAggregator(
            grouping=cfg.grouping,
            higher_is_better=scorer.higher_is_better,
        )

    raise ValueError(f"Unsupported context aggregation: {cfg.aggregation}")


def make_search(cfg: CausalChangeConfig, scorer):
    if cfg.graph_search == GraphSearch.TOPIC:
        return TopicSearch(scoring=scorer)

    if cfg.graph_search == GraphSearch.GLOBE:
        return GlobeSearch()

    raise ValueError(f"Unsupported graph search: {cfg.graph_search}")


class PipelineFactory:
    @staticmethod
    def from_config(cfg: CausalChangeConfig) -> DiscoveryEngine:
        # decide data domain, continuous/tabular data or time series, multi-context or not
        domain = make_domain(cfg)
        scorer = make_scorer(cfg)
        # stuff for multiple contexts
        context_preproc = make_context_preproc(cfg)
        aggregator = make_aggregator(cfg, scorer)
        # decide search algo
        search = make_search(cfg, scorer)

        if cfg.data_mode.is_temporal():
            return SpaceTimeEngine.from_config(cfg)
        return DiscoveryEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            context_preproc=context_preproc,
            scoring=scorer,
            aggregation=aggregator,
            search=search,
        )
