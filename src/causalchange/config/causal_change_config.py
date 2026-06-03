from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causalchange.config.types import (
    ContextMode,
    DataMode,
    GPType,
    GraphSearch,
    ScoreType, ChangepointMode, ChangepointMethod, PartitioningMethod, ChangepointScope,
)
from causalchange.discovery.context_combination import ContextCombinationParams



class CausalChangeConfigTabular(BaseModel):
    model_config = ConfigDict()  # extra="forbid")

    data_mode: DataMode
    graph_search: GraphSearch
    score_type: ScoreType | GPType
    aggregation: ContextMode

    spacetime: CausalChangeConfigTime | None = None #todo remove

    context_col: str = "context"
    grouping: ContextCombinationParams = Field(default_factory=ContextCombinationParams)

    score_kwargs: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42

    @model_validator(mode="after")
    def _validate_combo(self):
        if self.data_mode not in self.graph_search.compatible_modes():
            raise ValueError(f"{self.graph_search=} is not compatible with {self.data_mode=}.")
        if self.data_mode not in self.aggregation.compatible_modes():
            raise ValueError(f"{self.aggregation=} is not compatible with {self.data_mode=}.")
        if self.data_mode.is_context() and self.context_col is None:
            raise ValueError("context_col required for multi context data")
        if not self.data_mode.is_temporal() and self.spacetime is not None:
            raise ValueError("spacetime config  only valid for temporal data.")
        if self.data_mode.is_temporal() and self.spacetime is None:
            raise ValueError("spacetime config is required for temporal data.")
        if self.data_mode.is_temporal() and self.aggregation != ContextMode.SKIP:
            raise ValueError("aggregation must be SKIP for temporal data; SpaceTime handles contexts/regimes.")
        if self.score_type == ScoreType.GP:
            raise ValueError("score_type must be concrete. Use GPType.EXACT, GPType.FOURIER ")
        return self



class CausalChangeConfigTime(BaseModel):
    tau_max: int = 1
    changepoints: ChangepointMode = ChangepointMode.NONE
    fixed_changepoints: list[int] = Field(default_factory=list)
    d_min: int = 30
    max_iter: int = 3
    mechanism_test_alpha: float = 0.05

    detect_contexts: bool = True
    detect_regimes: bool = True

    changepoint_method: ChangepointMethod = ChangepointMethod.PELT
    changepoint_scope: ChangepointScope = ChangepointScope.GLOBAL
    partitioning_method: PartitioningMethod = PartitioningMethod.KERNEL
    pelt_penalty: float | Literal["bic", "mbic", "auto"] = "auto"

    @model_validator(mode="after")
    def _validate_spacetime(self):
        if self.tau_max <= 0:
            raise ValueError("tau_max must be positive.")
        if self.d_min <= 0:
            raise ValueError("d_min must be positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.changepoints == ChangepointMode.FIXED and not self.fixed_changepoints:
            raise ValueError("fixed_changepoints must be provided when changepoints=FIXED.")
        if not 0.0 < self.mechanism_test_alpha < 1.0:
            raise ValueError("mechanism_test_alpha must be in (0, 1).")
        if self.changepoints != ChangepointMode.FIXED and self.fixed_changepoints:
            raise ValueError("fixed_changepoints is only valid when changepoints=FIXED.")

        if isinstance(self.pelt_penalty, str):
            if self.pelt_penalty not in {"bic", "mbic", "auto"}:
                raise ValueError("pelt_penalty must be a positive number or one of " "{'bic', 'mbic', 'auto'}.")
        elif self.pelt_penalty <= 0:
            raise ValueError("pelt_penalty must be positive.")

        return self

