"""Factory for constructing causal change config."""

from __future__ import annotations

from typing import Any

from causalchange.config.causal_change_config import (
    CausalChangeConfig,
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
from causalchange.core.types import (
    ChangepointMethod,
    ChangepointMode,
    ChangepointScope,
    DataMode,
    GPType,
    GraphSearch,
    MechanismClusteringMethod,
    MechanismClusteringScope,
    MixedSCMType,
    PostprocessingMode,
    ScoreType,
    StatisticalTestingMethod,
    TabularContextMethod,
    TabularContextMode,
)


class ConfigFactory:
    """creates config from input parameters"""

    @staticmethod
    def make_causal_change_config(
        *,
        cfg: CausalChangeConfig | None,
        data_mode: DataMode,
        graph_search: GraphSearch,
        score_type: ScoreType | GPType,
        mix_type: MixedSCMType,
        context_mode: TabularContextMode,
        context_combination_method: TabularContextMethod,
        context_col: str | None,
        changepoint_mode: ChangepointMode,
        changepoint_scope: ChangepointScope,
        changepoint_method: ChangepointMethod,
        clustering_scope: MechanismClusteringScope,
        clustering_method: MechanismClusteringMethod,
        testing_method: StatisticalTestingMethod,
        postprocessing_mode: PostprocessingMode,
        tau_max: int | None,
        d_min: int,
        max_iter: int = 3,
        pelt_penalty: float | str = "auto",
        mechanism_test_alpha: float = 0.05,
        fixed_changepoints: list[int] | None = None,
        score_kwargs: dict[str, Any] | None = None,
    ) -> CausalChangeConfig:
        score_kwargs = {} if score_kwargs is None else dict(score_kwargs)
        if cfg is not None:
            return cfg

        if data_mode == DataMode.SKIP:
            raise ValueError("data_mode required when cfg is not provided.")
        if graph_search == GraphSearch.SKIP:
            raise ValueError("graph_search required when cfg is not provided.")
        if score_type == ScoreType.SKIP:
            raise ValueError("score_type required when cfg is not provided.")
        if score_type == ScoreType.GP:
            raise ValueError("decide GPType.EXACT or GPType.FOURIER")

        if data_mode.is_temporal():
            if tau_max is None:
                raise ValueError("tau_max required for temporal data")

            return CausalChangeConfigTemporal(
                data_mode=data_mode,
                graph_search=graph_search,
                score_type=score_type,
                context_col="context" if context_col is None else context_col,
                tau_max=tau_max,
                changepoint_mode=changepoint_mode,
                changepoint_scope=changepoint_scope,
                changepoint_method=changepoint_method,
                fixed_changepoints=fixed_changepoints or [],
                clustering_scope=clustering_scope,
                clustering_method=clustering_method,
                testing_method=testing_method,
                d_min=d_min,
                max_iter=max_iter,
                pelt_penalty=pelt_penalty,
                mechanism_test_alpha=mechanism_test_alpha,
                postprocessing_mode=postprocessing_mode,
                score_kwargs=score_kwargs,
            )

        if data_mode.is_context() and context_col is None:
            raise ValueError("context_col required for context data")

        return CausalChangeConfigTabular(
            data_mode=data_mode,
            graph_search=graph_search,
            score_type=score_type,
            context_col="context" if context_col is None else context_col,
            context_mode=context_mode,
            context_combination_method=context_combination_method,
            mix_type=mix_type,
            postprocessing_mode=postprocessing_mode,
            score_kwargs=score_kwargs,
        )
