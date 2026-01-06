from __future__ import annotations

from typing import Optional, Any

from pydantic import BaseModel, Field, ConfigDict, model_validator


from causalchange.discovery.old._mixins import LINCGroupingParams

from causalchange.config.config_types import DataMode, GraphSearch, ScoreType, ContextAggregation


class CausalChangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_mode: DataMode
    graph_search: GraphSearch
    score_type: ScoreType
    aggregation: ContextAggregation

    seed: int = 42
    context_col: str = "context"
    tau_max: Optional[int] = None
    grouping: LINCGroupingParams = Field(default_factory=LINCGroupingParams)

    score_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_combo(self):
        if self.data_mode not in set(self.graph_search.compatible_modes()):
            raise ValueError(
                f"{self.graph_search=} is not compatible with {self.data_mode=}."
            )

        if self.data_mode in {DataMode.CONTEXTS, DataMode.TIME_CONTEXTS}:
            if not self.context_col: raise ValueError("context_col is required for context-based modes")

        if self.data_mode.is_temporal():
            if self.tau_max is None: raise ValueError("tau_max required for temporal modes")


        return self


