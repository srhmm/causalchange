from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    n_changepoints: int = Field(2, ge=0)
    n_regimes: int = Field(2, ge=1)
    min_segment_length: int = Field(30, ge=1)
    mechanism_change_fraction: float = Field(0.5, ge=0.0, le=1.0)
    mechanism_shift_scale: float = Field(0.75, ge=0.0)
    burnin: int | None = Field(default=None, ge=0)
    allow_self_lag: bool = True

    @model_validator(mode="after")
    def _validate_spacetime_temporal_config(self):
        n_intervals = self.n_changepoints + 1

        if self.n_regimes > n_intervals:
            raise ValueError("n_regimes must be <= n_changepoints + 1.")

        if self.n_samples < n_intervals * self.min_segment_length:
            raise ValueError("n_samples is too small for n_changepoints and min_segment_length.")

        return self


class MultiTemporalDataConfig(DataConfigBase):
    setting: Literal["time-contexts"] = "time-contexts"

    n_nodes: int = Field(..., ge=1)
    edge_prob: float = Field(..., ge=0.0, le=1.0)
    seed: int = 42

    context_col: str = "context"
    n_contexts: int = Field(..., ge=1)
    n_samples_per_context: int = Field(..., ge=1)
    tau_max: int = Field(1, ge=1)

    weight_scale: float = 2.0
    noise_scale: float = 0.7
    nonlinearity: Nonlinearity = "tanh"

    n_datasets: int | None = None

    n_changepoints: int = Field(2, ge=0)
    n_regimes: int = Field(2, ge=1)
    n_context_clusters: int = Field(2, ge=1)
    min_segment_length: int = Field(30, ge=1)
    mechanism_change_fraction: float = Field(0.5, ge=0.0, le=1.0)
    mechanism_shift_scale: float = Field(0.75, ge=0.0)
    burnin: int | None = Field(default=None, ge=0)
    allow_self_lag: bool = True

    @model_validator(mode="after")
    def _validate_spacetime_temporal_context_config(self):
        n_intervals = self.n_changepoints + 1
        n_datasets = self.n_datasets or self.n_contexts

        if self.n_regimes > n_intervals:
            raise ValueError("n_regimes must be <= n_changepoints + 1.")

        if self.n_samples_per_context < n_intervals * self.min_segment_length:
            raise ValueError("n_samples_per_context is too small for n_changepoints and min_segment_length.")

        if self.n_context_clusters > n_datasets:
            raise ValueError("n_context_clusters must be <= number of datasets.")

        return self


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


class SpaceTimeAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["spacetime"] = "spacetime"
    score_type: Literal["lin", "gam", "spline", "krr", "gp", "ff"] = "gam"
    tau_max: int | None = Field(default=None, ge=1)

    changepoint_mode: Literal["none", "detect", "oracle"] = "detect"
    detect_contexts: bool = True
    detect_regimes: bool = True


AlgoConfig = Annotated[
    LincAlgoConfig | ChainAlgoConfig | TopicAlgoConfig | SpaceTimeAlgoConfig,
    Field(discriminator="name"),
]


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    algo: AlgoConfig

    @model_validator(mode="after")
    def _couple_algo_and_data(self):
        allowed_settings_by_algo = {
            "topic": {"single"},
            "linc": {"multi"},
            "chain": {"multi"},
            "spacetime": {"time", "time-contexts"},
        }

        allowed = allowed_settings_by_algo.get(self.algo.name)
        if allowed is None:
            raise ValueError(f"Unknown benchmark algo: {self.algo.name!r}")

        if self.data.setting not in allowed:
            raise ValueError(f"algo={self.algo.name!r} is only valid with " f"data.setting in {sorted(allowed)}.")

        return self
