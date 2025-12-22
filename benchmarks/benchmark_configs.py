from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from causalchange._cc_types import DataMode

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
    setting: Literal[DataMode.IID] = DataMode.IID
    linearity: Literal["linear"] = "linear"

    n_samples: int = Field(..., ge=1)


class SingleNonlinearDataConfig(DataConfigBase):
    setting: Literal[DataMode.IID] = DataMode.IID
    linearity: Literal["nonlinear"] = "nonlinear"

    n_samples: int = Field(..., ge=1)
    nonlinearity: Nonlinearity




class MultiLinearDataConfig(DataConfigBase):
    setting: Literal[DataMode.CONTEXTS] = DataMode.CONTEXTS
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
    setting: Literal[DataMode.CONTEXTS] = DataMode.CONTEXTS
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




class SingleTemporalLinearDataConfig(DataConfigBase):
    setting: Literal[DataMode.TIME] = DataMode.TIME
    linearity: Literal["linear"] = "linear"

    n_samples: int = Field(..., ge=1)
    tau_max: int = Field(1, ge=1)


class SingleTemporalNonlinearDataConfig(DataConfigBase):
    setting: Literal[DataMode.TIME] = DataMode.TIME
    linearity: Literal["nonlinear"] = "nonlinear"

    n_samples: int = Field(..., ge=1)
    tau_max: int = Field(1, ge=1)
    nonlinearity: Nonlinearity


class MultiTemporalLinearDataConfig(DataConfigBase):
    setting: Literal[DataMode.TIME_CONTEXTS] = DataMode.TIME_CONTEXTS
    linearity: Literal["linear"] = "linear"

    context_col: str = "context"
    n_contexts: int = Field(..., ge=1)
    n_samples_per_context: int = Field(..., ge=1)
    tau_max: int = Field(1, ge=1)


class MultiTemporalNonlinearDataConfig(DataConfigBase):
    setting: Literal[DataMode.TIME_CONTEXTS] = DataMode.TIME_CONTEXTS
    linearity: Literal["nonlinear"] = "nonlinear"

    context_col: str = "context"
    n_contexts: int = Field(..., ge=1)
    n_samples_per_context: int = Field(..., ge=1)
    tau_max: int = Field(1, ge=1)
    nonlinearity: Nonlinearity



IIDDataConfig = Annotated[
    Union[SingleLinearDataConfig, SingleNonlinearDataConfig],
    Field(discriminator="linearity"),
]

ContextsDataConfig = Annotated[
    Union[MultiLinearDataConfig, MultiNonlinearDataConfig],
    Field(discriminator="linearity"),
]

TimeDataConfig = Annotated[
    Union[SingleTemporalLinearDataConfig, SingleTemporalNonlinearDataConfig],
    Field(discriminator="linearity"),
]

TimeContextsDataConfig = Annotated[
    Union[MultiTemporalLinearDataConfig, MultiTemporalNonlinearDataConfig],
    Field(discriminator="linearity"),
]

DataConfig = Annotated[
    Union[IIDDataConfig, ContextsDataConfig, TimeDataConfig, TimeContextsDataConfig],
    Field(discriminator="setting"),
]




class LincAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["linc"] = "linc"
    context_col: str = "context"
    scoring_method: Literal["aic-g", "bic-g"] = "bic-g"


class TopicAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["topic"] = "topic"
    scoring_method: Literal["aic-g", "bic-g"] = "bic-g"


class SpaceTimeAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["spacetime"] = "spacetime"
    scoring_method: Optional[Literal["aic-g", "bic-g"]] = None


class SpaceTimeCAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["spacetime_c"] = "spacetime_c"
    context_col: str = "context"
    scoring_method: Optional[Literal["aic-g", "bic-g"]] = None


AlgoConfig = Annotated[
    Union[LincAlgoConfig, TopicAlgoConfig, SpaceTimeAlgoConfig, SpaceTimeCAlgoConfig],
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
        if self.algo.name == "topic" and self.data.setting != DataMode.IID:
            raise ValueError("algo=topic is only valid with data.setting=DataMode.IID")
        if self.algo.name == "linc" and self.data.setting != DataMode.CONTEXTS:
            raise ValueError("algo=linc is only valid with data.setting=DataMode.CONTEXTS")
        if self.algo.name == "spacetime" and self.data.setting != DataMode.TIME:
            raise ValueError("algo=spacetime is only valid with data.setting=DataMode.TIME")
        if self.algo.name == "spacetime_c" and self.data.setting != DataMode.TIME_CONTEXTS:
            raise ValueError("algo=spacetime_c is only valid with data.setting=DataMode.TIME_CONTEXTS")
        return self
