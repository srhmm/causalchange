"""Factory for constructing discovery engines from causal change config."""

from __future__ import annotations

from causalchange.config.causal_change_config import (
    CausalChangeConfig,
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
from causalchange.core.types import ContextMode, GraphSearch
from causalchange.discovery.changepoint_detection import ChangepointDetection
from causalchange.discovery.context_combination import CHAINContextCombination, LINCContextCombination, SkipCombination
from causalchange.discovery.graphsearch_tabular import GraphSearchTabularGreedy, GraphSearchTabularTopological
from causalchange.discovery.graphsearch_temporal import GraphSearchTemporalGreedy, GraphSearchTemporalTopological
from causalchange.discovery.scm_clustering import SpaceTimeClustering
from causalchange.domain.context import MultiContextDomain, SingleContextDomain
from causalchange.domain.tabular import TabularDomain
from causalchange.domain.temporal import TemporalDomain
from causalchange.engines.tabular import TabularDiscoveryEngine
from causalchange.engines.temporal import TemporalDiscoveryEngine
from causalchange.scoring.tabular import SCMScoreTabular
from causalchange.scoring.temporal import SCMScoreTemporal


class EngineFactory:
    """shows the high-level control flow for causal discovery."""

    # factory = converts a config to a discovery engine
    @staticmethod
    def from_config(cfg: CausalChangeConfig):
        if cfg.data_mode.is_temporal():
            return EngineFactory._make_temporal_engine(cfg)

        return EngineFactory._make_tabular_engine(cfg)

    @staticmethod
    def _make_tabular_engine(cfg: CausalChangeConfigTabular) -> TabularDiscoveryEngine:
        domain = TabularDomain()
        scoring = SCMScoreTabular(cfg=cfg)
        search = (
            GraphSearchTabularTopological(scoring=scoring)
            if cfg.graph_search == GraphSearch.TOPIC
            else GraphSearchTabularGreedy()
        )

        context_preprocessing = (
            MultiContextDomain(context_col=cfg.context_col) if cfg.data_mode.is_context() else SingleContextDomain()
        )
        context_combination = (
            LINCContextCombination(
                grouping=cfg.grouping,
                higher_is_better=scoring.higher_is_better,
            )
            if cfg.context_mode == ContextMode.LINC
            else (CHAINContextCombination() if cfg.context_mode == ContextMode.CHAIN else SkipCombination())
        )

        return TabularDiscoveryEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            context_preproc=context_preprocessing,
            scoring=scoring,
            context_comb=context_combination,
            search=search,
        )

    @staticmethod
    def _make_temporal_engine(cfg: CausalChangeConfigTemporal) -> TemporalDiscoveryEngine:
        domain = TemporalDomain(tau_max=cfg.tau_max)
        scoring = SCMScoreTemporal(cfg=cfg)

        search = (
            GraphSearchTemporalTopological(scoring=scoring)
            if cfg.graph_search == GraphSearch.TOPIC
            else GraphSearchTemporalGreedy(scoring=scoring)
        )

        changepoint_detection = ChangepointDetection(cfg)
        scm_clustering = SpaceTimeClustering(cfg)

        return TemporalDiscoveryEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            scoring=scoring,
            search=search,
            changepoint_detection=changepoint_detection,
            scm_clustering=scm_clustering,
            context_col=cfg.context_col,
            tau_max=cfg.tau_max,
            changepoint_mode=cfg.changepoints,
            changepoint_scope=cfg.changepoint_scope,
            max_iter=cfg.max_iter,
            detect_contexts=cfg.detect_contexts,
            detect_regimes=cfg.detect_regimes,
            diagnostics={
                "graph_search": cfg.graph_search.value,
                "score_type": str(cfg.score_type),
            },
        )
