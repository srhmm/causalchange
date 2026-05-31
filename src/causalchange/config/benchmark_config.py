from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MetricName = Literal[
    "shd",
    "edge_f1",
    "skel_f1",
    "edge_precision",
    "edge_recall",
    "skel_precision",
    "skel_recall",
    "time_s",
]
Nonlinearity = Literal["lin", "tanh", "sin", "relu"]
InterventionLinear = Literal["hard", "soft_weight", "shift", "noise"]
InterventionNonlinear = Literal["hard", "soft_weight", "shift", "noise"]


class DataConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_nodes: int = Field(..., ge=1)
    edge_prob: float = Field(..., ge=0.0, le=1.0)
    seed: int = 42

    weight_scale: float = 2.0
    noise_scale: float = 0.7


class SingleDataConfig(DataConfigBase):
    setting: Literal["single"] = "single"

    n_samples: int = Field(..., ge=1)
    nonlinearity: Nonlinearity


class MultiDataConfig(DataConfigBase):
    setting: Literal["multi"] = "multi"
    nonlinearity: Nonlinearity

    context_col: str = "context"
    n_contexts: int = Field(..., ge=1)
    n_samples_per_context: int = Field(..., ge=1)
    n_intervened_per_context: int = Field(1, ge=0)

    weight_scale_intervened: float = 2.0
    shift_scale: float = 2.0
    noise_scale_intervened: float | None = None

    intervention_type: InterventionNonlinear = "soft_weight"
    alt_nonlinearity: Nonlinearity | None = None

    @model_validator(mode="after")
    def _alt_required_for_soft_mechanism(self):
        if self.intervention_type == "soft_mechanism" and self.alt_nonlinearity is None:
            raise ValueError("alt_nonlinearity is required when intervention_type='soft_mechanism'.")
        return self


class MixedDataConfig(DataConfigBase):
    setting: Literal["mixed"] = "mixed"

    context_col: str = "context"
    n_contexts: int = Field(..., ge=1)
    n_samples_per_context: int = Field(..., ge=1)
    n_intervened_per_context: int = Field(1, ge=0)

    intervention_type: InterventionNonlinear = "soft_weight"

    weight_scale_intervened: float = 2.0
    shift_scale: float = 2.0
    noise_scale_intervened: float | None = None

    nonlinearity: Nonlinearity
    alt_nonlinearity: Nonlinearity | None = None

    @model_validator(mode="after")
    def _alt_required_for_soft_mechanism(self):
        if self.intervention_type == "soft_mechanism" and self.alt_nonlinearity is None:
            raise ValueError("alt_nonlinearity is required when intervention_type='soft_mechanism'.")
        return self


class SingleTemporalDataConfig(DataConfigBase):
    setting: Literal["time"] = "time"

    n_nodes: int = Field(..., ge=1)
    edge_prob: float = Field(..., ge=0.0, le=1.0)
    seed: int = 42

    n_samples: int = Field(..., ge=1)
    tau_max: int = Field(1, ge=1)
    weight_scale: float = 2.0
    noise_scale: float = 0.7

    nonlinearity: Nonlinearity = "tanh"


class MultiTemporalDataConfig(DataConfigBase):
    setting: Literal["time-contexts"] = "time-contexts"

    n_nodes: int = Field(..., ge=1)
    edge_prob: float = Field(..., ge=0.0, le=1.0)
    seed: int = 42

    context_col: str = "context"
    n_contexts: int = Field(..., ge=1)
    n_samples_per_context: int = Field(..., ge=1)
    tau_max: int = Field(1, ge=1)

    n_intervened_per_context: int = Field(1, ge=0)
    intervention_type: Literal["hard", "soft_weight", "shift", "noise"] = "hard"  # , "soft_mechanism"] = "hard"

    weight_scale: float = 2.0
    noise_scale: float = 0.7
    nonlinearity: Nonlinearity = "tanh"


DataConfig = Annotated[
    SingleDataConfig | MultiDataConfig | SingleTemporalDataConfig | MultiTemporalDataConfig | MixedDataConfig,
    Field(discriminator="setting"),
]


class LincAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["linc"] = "linc"
    context_col: str = "context"
    score_type: Literal["lin", "gam", "spline", "krr", "gp", "ff"] = "gam"


class ChainAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["chain"] = "chain"
    context_col: str = "context"
    score_type: Literal["lin", "gam", "spline", "krr", "gp", "ff"] = "gam"


class TopicAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["topic"] = "topic"
    score_type: Literal["lin", "gam", "spline", "krr", "gp", "ff"] = "gam"


class SpaceTimeCAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["spacetime-c"] = "spacetime-c"
    context_col: str = "context"
    score_type: Literal["lin", "gam", "spline", "krr", "gp", "ff"] = "gam"
    tau_max: int | None = Field(default=None, ge=1)


class SpaceTimeAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["spacetime"] = "spacetime"
    score_type: Literal["lin", "gam", "spline", "krr", "gp", "ff"] = "gam"
    tau_max: int | None = Field(default=None, ge=1)


AlgoConfig = Annotated[
    LincAlgoConfig | ChainAlgoConfig | TopicAlgoConfig | SpaceTimeAlgoConfig | SpaceTimeCAlgoConfig,
    Field(discriminator="name"),
]


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: set[MetricName] = Field(default_factory=lambda: {"edge_f1", "skel_f1"})

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
        if self.algo.name == "spacetime" and self.data.setting != "time":
            raise ValueError("algo=time is only valid with data.setting='time'.")
        if self.algo.name == "spacetime-c" and self.data.setting != "time-contexts":
            raise ValueError("algo=spacetime-c is only valid with data.setting='time-contexts'.")
        return self
