"""Factory for constructing discovery engines from causal change config."""

from __future__ import annotations

from causalchange.config.causal_change_config import (
    CausalChangeConfig,
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
from causalchange.core.protocols import (
    TabularScoringProtocol,
)
from causalchange.core.types import GraphSearch, MixedSCMType, TabularContextMethod, TabularContextMode
from causalchange.discovery.changepoints import ChangepointDetection
from causalchange.discovery.context_combination import CHAINContextCombination, LINCContextCombination, SkipCombination
from causalchange.discovery.graph_tabular import GraphSearchTabularGreedy, GraphSearchTabularTopological
from causalchange.discovery.graph_temporal import GraphSearchTemporalGreedy, GraphSearchTemporalTopological
from causalchange.discovery.scm_clustering import TemporalSCMClustering
from causalchange.domain.context import MultiContextDomain, SingleContextDomain
from causalchange.domain.tabular import TabularDomain
from causalchange.domain.temporal import TemporalDomain
from causalchange.engines.tabular import TabularDiscoveryEngine
from causalchange.engines.temporal import TemporalDiscoveryEngine
from causalchange.scoring.regression_mixtures import SCMScoreCMM
from causalchange.scoring.tabular import SCMScoreTabular
from causalchange.scoring.temporal import SCMScoreTemporal


class EngineFactory:
    """creates engines that control the flow of causal discovery"""

    # factory = converts a config to a discovery engine
    @staticmethod
    def from_config(cfg: CausalChangeConfig) -> TabularDiscoveryEngine | TemporalDiscoveryEngine:
        if isinstance(cfg, CausalChangeConfigTemporal):
            return EngineFactory._make_temporal_engine(cfg)

        if isinstance(cfg, CausalChangeConfigTabular):
            return EngineFactory._make_tabular_engine(cfg)

        raise TypeError(f"Unsupported config type: {type(cfg)!r}")

    @staticmethod
    def _make_tabular_engine(cfg: CausalChangeConfigTabular) -> TabularDiscoveryEngine:
        domain = TabularDomain()

        scoring: TabularScoringProtocol = (
            SCMScoreCMM(cfg=cfg)
            if cfg.mix_type != MixedSCMType.SKIP and cfg.context_mode == TabularContextMode.SKIP
            else SCMScoreTabular(cfg=cfg)
        )

        search = (
            GraphSearchTabularTopological(scoring=scoring)
            if cfg.graph_search == GraphSearch.TOPIC
            else GraphSearchTabularGreedy(scoring=scoring)
        )

        context_preprocessing = (
            MultiContextDomain(context_col=cfg.context_col) if cfg.data_mode.is_context() else SingleContextDomain()
        )
        context_combination = (
            LINCContextCombination(
                grouping=cfg.context_combination_kwargs,
                gain_threshold=cfg.context_gain_threshold,
                higher_is_better=scoring.higher_is_better,
                mechanism_clustering_method=cfg.mechanism_clustering_method,
                testing_method=cfg.testing_method,
                mechanism_test_alpha=cfg.mechanism_test_alpha,
                mechanism_clustering_n_clusters=cfg.mechanism_clustering_n_clusters,
                mechanism_clustering_distance_threshold=cfg.mechanism_clustering_distance_threshold,
                seed=cfg.seed,
            )
            if cfg.context_combination_method == TabularContextMethod.LINC
            else (
                CHAINContextCombination(
                    higher_is_better=scoring.higher_is_better,
                    seed=cfg.seed,
                )
                if cfg.context_combination_method == TabularContextMethod.CHAIN
                else SkipCombination()
            )
        )
        return TabularDiscoveryEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            context_preproc=context_preprocessing,
            scoring=scoring,
            context_comb=context_combination,
            search=search,
            postprocessing_mode=cfg.postprocessing_mode,
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
        scm_clustering = TemporalSCMClustering(cfg)

        return TemporalDiscoveryEngine(
            data_mode=cfg.data_mode,
            domain=domain,
            scoring=scoring,
            search=search,
            changepoint_detection=changepoint_detection,
            scm_clustering=scm_clustering,
            clustering_scope=cfg.clustering_scope,
            context_col=cfg.context_col,
            tau_max=cfg.tau_max,
            changepoint_mode=cfg.changepoint_mode,
            changepoint_scope=cfg.changepoint_scope,
            max_iter=cfg.max_iter,
            postprocessing_mode=cfg.postprocessing_mode,
        )
