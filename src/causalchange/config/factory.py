"""Factory for constructing causal change config."""

from __future__ import annotations

from causalchange.config.causal_change_config import (
    CausalChangeConfig,
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
from causalchange.core.types import (
    ChangepointMode,
    ChangepointScope,
    ContextMode,
    DataMode,
    GPType,
    GraphSearch,
    ScoreType,
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
        context_mode: ContextMode,
        context_col: str | None,
        tau_max: int | None,
        changepoints: ChangepointMode,
        changepoint_scope: ChangepointScope,
        fixed_changepoints: list[int] | None,
        d_min: int,
        max_iter: int = 3,
        pelt_penalty: float | str = "auto",
        detect_contexts: bool = True,
        detect_regimes: bool = True,
        mechanism_test_alpha: float = 0.05,
    ) -> CausalChangeConfig:
        if cfg is not None:
            return cfg
        if data_mode == DataMode.SKIP:
            raise ValueError("data_mode is required when cfg is not provided.")
        if graph_search == GraphSearch.SKIP:
            raise ValueError("graph_search is required when cfg is not provided.")
        if score_type == ScoreType.SKIP:
            raise ValueError("score_type is required when cfg is not provided.")
        if score_type == ScoreType.GP:
            raise ValueError("pass GPType.EXACT or GPType.FOURIER instd of ScoreType.GP.")

        if data_mode.is_temporal():
            if tau_max is None:
                raise ValueError("tau_max is required for temporal data modes.")

            return CausalChangeConfigTemporal(
                data_mode=data_mode,
                graph_search=graph_search,
                score_type=score_type,
                tau_max=tau_max,
                changepoints=changepoints,
                changepoint_scope=changepoint_scope,
                fixed_changepoints=fixed_changepoints or [],
                d_min=d_min,
                max_iter=max_iter,
                pelt_penalty=pelt_penalty,
                detect_contexts=detect_contexts,
                detect_regimes=detect_regimes,
                mechanism_test_alpha=mechanism_test_alpha,
            )

        if data_mode.is_context() and context_col is None:
            raise ValueError("context_col is required for context data mode.")

        return CausalChangeConfigTabular(
            data_mode=data_mode,
            graph_search=graph_search,
            score_type=score_type,
            context_mode=context_mode,
            context_col=context_col or "context",
        )
