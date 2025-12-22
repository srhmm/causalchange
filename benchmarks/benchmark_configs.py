from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

MetricName = Literal["shd", "edge_f1", "skel_f1", "time_s"]

Nonlinearity = Literal["tanh", "sin", "relu"]
InterventionLinear = Literal["hard", "soft_weight", "shift", "noise"]
InterventionNonlinear = Literal["hard", "soft_weight", "soft_mechanism", "shift", "noise"]


class DataConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_nodes: int = Field(..., ge=1)
    edge_prob: float = Field(..., ge=0.0, le=1.0)
    seed: int = 42

    weight_scale: float = 2.0
    noise_scale: float = 0.7


class SingleLinearDataConfig(DataConfigBase):
    setting: Literal["single"] = "single"
    linearity: Literal["linear"] = "linear"

    n_samples: int = Field(..., ge=1)



class SingleNonlinearDataConfig(DataConfigBase):
    setting: Literal["single"] = "single"
    linearity: Literal["nonlinear"] = "nonlinear"

    n_samples: int = Field(..., ge=1)
    nonlinearity: Nonlinearity


class MultiLinearDataConfig(DataConfigBase):
    setting: Literal["multi"] = "multi"
    linearity: Literal["linear"] = "linear"

    context_col: str = "context"
    n_contexts: int = Field(..., ge=1)
    n_samples_per_context: int = Field(..., ge=1)
    n_intervened_per_context: int = Field(1, ge=0)

    intervention_type: InterventionLinear = "soft_weight"

    weight_scale_intervened: float = 2.0
    shift_scale: float = 2.0
    noise_scale_intervened: Optional[float] = None



class MultiNonlinearDataConfig(DataConfigBase):
    setting: Literal["multi"] = "multi"
    linearity: Literal["nonlinear"] = "nonlinear"

    context_col: str = "context"
    n_contexts: int = Field(..., ge=1)
    n_samples_per_context: int = Field(..., ge=1)
    n_intervened_per_context: int = Field(1, ge=0)

    intervention_type: InterventionNonlinear = "soft_weight"

    weight_scale_intervened: float = 2.0
    shift_scale: float = 2.0
    noise_scale_intervened: Optional[float] = None

    nonlinearity: Nonlinearity
    alt_nonlinearity: Optional[Nonlinearity] = None

    @model_validator(mode="after")
    def _alt_required_for_soft_mechanism(self):
        if self.intervention_type == "soft_mechanism" and self.alt_nonlinearity is None:
            raise ValueError("alt_nonlinearity is required when intervention_type='soft_mechanism'.")
        return self


SingleDataConfig = Annotated[
    Union[SingleLinearDataConfig, SingleNonlinearDataConfig],
    Field(discriminator="linearity"),
]

MultiDataConfig = Annotated[
    Union[MultiLinearDataConfig, MultiNonlinearDataConfig],
    Field(discriminator="linearity"),
]

DataConfig = Annotated[
    Union[SingleDataConfig, MultiDataConfig],
    Field(discriminator="setting"),
]


class LincAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["linc"] = "linc"
    context_col: str = "context"
    score_type: Literal["aic-g", "bic-g"] = "bic-g"


class TopicAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["topic"] = "topic"
    score_type: Literal["aic-g", "bic-g"] = "bic-g"

from typing import Literal, Optional, Union, Annotated
from pydantic import BaseModel, Field, ConfigDict

PCVariant = Literal["orig", "stable", "parallel"]
PCReturnType = Literal["pdag", "cpdag", "dag"]

class PcAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["pc"] = "pc"

    variant: PCVariant = "parallel"
    ci_test: Optional[Union[str, object]] = None
    return_type: PCReturnType = "dag"
    significance_level: float = 0.01
    max_cond_vars: int = 5
    n_jobs: int = -1
    show_progress: bool = False


AlgoConfig = Annotated[
    Union[LincAlgoConfig, TopicAlgoConfig, PcAlgoConfig],
    Field(discriminator="name"),
]


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: set[MetricName] = Field(default_factory=lambda: {"edge_f1", "skel_f1", "time_s"})

    @field_validator("metrics")
    @classmethod
    def _non_empty(cls, v: set[MetricName]) -> set[MetricName]:
        if not v:
            raise ValueError("metrics must not be empty.")
        return v

class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    algo: AlgoConfig
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)

    @model_validator(mode="after")
    def _couple_algo_and_data(self):
        if self.algo.name == "linc" and self.data.setting != "multi":
            raise ValueError("algo=linc is only valid with data.setting='multi'.")
        if self.algo.name == "topic" and self.data.setting != "single":
            raise ValueError("algo=topic is only valid with data.setting='single'.")
        if self.algo.name == "pc" and self.data.setting != "single":
            raise ValueError(
                "algo=pc is only valid with data.setting='single' (otherwise it will treat context as a variable).")
        return self
