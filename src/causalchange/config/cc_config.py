from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causalchange.config.cc_types import (
    ContextAggregation,
    DataMode,
    GraphSearch,
    ScoreType,
)
from causalchange.discovery.search_multi.linc import LINCGroupingParams


class ChangepointMode(Enum):
    NONE = "none"
    FIXED = "fixed"
    DETECT = "detect"


class SpaceTimeConfig(BaseModel):
    tau_max: int = 1
    changepoints: ChangepointMode = ChangepointMode.NONE
    fixed_changepoints: list[int] = Field(default_factory=list)
    d_min: int = 30
    max_iter: int = 3

    detect_contexts: bool = False
    detect_regimes: bool = False

    changepoint_method: str = "pelt"
    partitioning_method: str = "kernel"

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

        if self.changepoints != ChangepointMode.FIXED and self.fixed_changepoints:
            raise ValueError("fixed_changepoints is only valid when changepoints=FIXED.")

        return self


class CausalChangeConfig(BaseModel):
    model_config = ConfigDict()  # extra="forbid")

    data_mode: DataMode
    graph_search: GraphSearch
    score_type: ScoreType
    aggregation: ContextAggregation

    spacetime: SpaceTimeConfig | None = None

    context_col: str = "context"
    grouping: LINCGroupingParams = Field(default_factory=LINCGroupingParams)

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
        if self.data_mode.is_temporal() and self.aggregation != ContextAggregation.SKIP:
            raise ValueError("aggregation must be SKIP for temporal data; SpaceTime handles contexts/regimes.")
        return self
