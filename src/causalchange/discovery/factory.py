# causalchange/discovery/factory.py

from __future__ import annotations

from causalchange.config._cc_types import DataMode, GraphSearch, ContextAggregation
from causalchange.config.cc_config import CausalChangeConfig

from causalchange.discovery.pipeline import DiscoveryEngine

from causalchange.discovery.domain.tabular import TabularDomain
from causalchange.discovery.domain.temporal import TemporalDomain

from causalchange.discovery.contexts.single import SingleContextProvider
from causalchange.discovery.contexts.iid import IIDContextProvider

from causalchange.discovery.scoring.edge_score_tabular import EdgeScoreTabularScorer
from causalchange.discovery.scoring.edge_score_autoreg import EdgeScoreAutoRegressiveScorer

from causalchange.discovery.aggregation.linc import LINCAggregator
from causalchange.discovery.aggregation.chain import ChainAggregator

from causalchange.discovery.search.topic import TopicSearch
from causalchange.discovery.search.globe import GlobeSearch


class PipelineFactory:
    @staticmethod
    def from_config(cfg: CausalChangeConfig) -> DiscoveryEngine:
        # --- domain ---
        if cfg.data_mode.is_temporal():
            domain = TemporalDomain(
                tau_max=cfg.tau_max,
                allow_instantaneous=cfg.allow_instantaneous,
            )
        else:
            domain = TabularDomain()

        # --- contexts ---
        if cfg.data_mode in (DataMode.CONTEXTS, DataMode.TIME_CONTEXTS):
            context_provider = IIDContextProvider(context_col=cfg.context_col)
        else:
            context_provider = SingleContextProvider()

        # --- scorer ---
        # Decide which scorer to use based on temporal-ness
        if cfg.data_mode.is_temporal():
            scorer = EdgeScoreAutoRegressiveScorer(
                data_mode=cfg.data_mode,
                score_type=cfg.score_type,
                tau_max=cfg.tau_max,
                allow_instantaneous=cfg.allow_instantaneous,
                score_params=cfg.score_kwargs,   # rename if yours is score_params
                higher_is_better=cfg.higher_is_better,
                gain_threshold=cfg.gain_threshold,  # or cfg.grouping.gain_threshold if you unify
            )
        else:
            scorer = EdgeScoreTabularScorer(
                data_mode=cfg.data_mode,
                score_type=cfg.score_type,
                score_params=cfg.score_kwargs,
                higher_is_better=cfg.higher_is_better,
                gain_threshold=cfg.gain_threshold,
            )

        # --- aggregation ---
        if cfg.aggregation == ContextAggregation.LINC:
            aggregator = LINCAggregator(
                grouping=cfg.grouping,
                higher_is_better=scorer.higher_is_better,
            )
        elif cfg.aggregation == ContextAggregation.CHAIN:
            aggregator = ChainAggregator(
                lambda_inv=cfg.lambda_inv,
                mmd_max_samples=cfg.mmd_max_samples,
                mmd_gamma=cfg.mmd_gamma,
                mmd_compare_to=cfg.mmd_compare_to,
                higher_is_better=scorer.higher_is_better,
                seed=cfg.seed,
            )
        else:
            raise ValueError(f"Unknown aggregation: {cfg.aggregation}")

        # --- search ---
        if cfg.graph_search == GraphSearch.TOPIC:
            search = TopicSearch(
                transition_gain=scorer.transition_gain,
                score_significant=scorer.score_significant,
                score_is_better=scorer.score_is_better,
            )
        elif cfg.graph_search == GraphSearch.GLOBE:
            search = GlobeSearch()
        else:
            raise ValueError(f"Unknown graph_search: {cfg.graph_search}")

        return DiscoveryEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            context_provider=context_provider,
            scorer=scorer,
            aggregator=aggregator,
            search=search,
        )
