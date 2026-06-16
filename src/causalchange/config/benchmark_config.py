from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Nonlinearity = Literal["lin", "tanh", "sin", "relu"]
InterventionLinear = Literal["hard", "soft-weight", "shift", "noise"]
InterventionNonlinear = Literal["hard", "soft-weight", "soft-mechanism", "shift", "noise"]
ChangepointMode = Literal["skip", "detect", "fixed"]
ChangepointScope = Literal["skip", "global", "per-context"]
ChangepointMethod = Literal["skip", "pelt"]

ClusteringScope = Literal["skip", "contexts", "regimes", "regimes-contexts"]
ClusteringMethod = Literal["skip", "mechanism-clustering", "testing"]
TestingMethod = Literal["skip", "kernel"]
MixType = Literal["lin", "quadratic", "cubic", "nspline", "bspline"]


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

    intervention_type: InterventionNonlinear = "soft-weight"
    alt_nonlinearity: Nonlinearity | None = None

    @model_validator(mode="after")
    def _alt_required_for_soft_mechanism(self):
        if self.intervention_type == "soft-mechanism" and self.alt_nonlinearity is None:
            raise ValueError("alt_nonlinearity is required when intervention_type='soft-mechanism'.")
        return self


ClusterMode = Literal["global", "local"]
MixedMechanismChange = Literal["soft-weight", "soft-mechanism", "shift", "noise"]


class MixedDataConfig(DataConfigBase):
    setting: Literal["mixed"] = "mixed"

    n_mechanisms: int = Field(..., ge=1)
    n_samples_per_mechanism: int = Field(..., ge=1)
    n_mixed_variables: int = Field(1, ge=0)
    cluster_mode: ClusterMode = "local"

    mechanism_change: MixedMechanismChange = "soft-weight"

    weight_scale_intervened: float = 2.0
    shift_scale: float = 2.0
    noise_scale_intervened: float | None = None

    nonlinearity: Nonlinearity
    alt_nonlinearity: Nonlinearity | None = None

    @model_validator(mode="after")
    def _validate_mixed_data_config(self):
        if self.n_mixed_variables > self.n_nodes:
            raise ValueError("n_mixed_variables cannot exceed n_nodes.")

        if self.mechanism_change == "soft-mechanism" and self.alt_nonlinearity is None:
            raise ValueError("alt_nonlinearity is required when mechanism_change='soft-mechanism'.")

        return self


class SingleTemporalDataConfig(DataConfigBase):
    setting: Literal["time"] = "time"

    n_nodes: int = Field(..., ge=1)
    edge_prob: float = Field(..., ge=0.25, le=1.0)
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


class TopicAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["topic"] = "topic"
    score_type: Literal["lin", "gam", "spline", "krr", "gp", "ff"] = "gam"


class CmmAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["cmm"] = "cmm"
    score_type: Literal["lin", "gam", "spline", "krr", "gp", "ff"] = "lin"
    mix_type: MixType = "lin"
    k_max: int = Field(5, ge=1)
    lambda_mix: float = Field(0.0, ge=0.0)
    hybrid_mixing: bool = False
    max_em_iter: int = Field(100, ge=1)
    n_init: int = Field(3, ge=1)
    tol: float = Field(1e-5, gt=0.0)
    ridge: float = Field(1e-8, ge=0.0)


class SpaceTimeAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["spacetime"] = "spacetime"
    score_type: Literal["lin", "gam", "spline", "krr", "gp", "ff"] = "lin"

    tau_max: int | None = Field(default=None, ge=1)
    d_min: int | None = Field(default=None, ge=1)

    changepoint_mode: ChangepointMode = "skip"
    changepoint_scope: ChangepointScope = "skip"
    changepoint_method: ChangepointMethod = "skip"
    fixed_changepoints: list[int] | None = None

    clustering_scope: ClusteringScope = "skip"
    clustering_method: ClusteringMethod = "skip"
    testing_method: TestingMethod = "skip"

    @model_validator(mode="after")
    def _validate_spacetime_algo_config(self):
        if self.changepoint_mode == "skip":
            if self.changepoint_scope != "skip" or self.changepoint_method != "skip":
                raise ValueError(
                    "changepoint_scope and changepoint_method must be 'skip' " "when changepoint_mode='skip'."
                )

        if self.changepoint_mode == "detect":
            if self.changepoint_scope == "skip":
                raise ValueError("changepoint_scope must not be 'skip' when changepoint_mode='detect'.")
            if self.changepoint_method == "skip":
                raise ValueError("changepoint_method must not be 'skip' when changepoint_mode='detect'.")

        if self.changepoint_mode == "fixed":
            if self.changepoint_scope == "skip":
                raise ValueError("changepoint_scope must not be 'skip' when changepoint_mode='fixed'.")
            if self.changepoint_method != "skip":
                raise ValueError("changepoint_method should be 'skip' when changepoint_mode='fixed'.")

        if self.clustering_scope == "skip":
            if self.clustering_method != "skip" or self.testing_method != "skip":
                raise ValueError("clustering_method and testing_method must be 'skip' " "when clustering_scope='skip'.")

        if self.clustering_method == "mechanism-clustering":
            if self.clustering_scope == "skip":
                raise ValueError("clustering_scope must not be 'skip' for mechanism clustering.")
            if self.testing_method != "skip":
                raise ValueError("testing_method should be 'skip' for mechanism clustering.")

        if self.clustering_method == "testing":
            if self.clustering_scope == "skip":
                raise ValueError("clustering_scope must not be 'skip' for statistical testing.")
            if self.testing_method == "skip":
                raise ValueError("testing_method must not be 'skip' when clustering_method='testing'.")

        return self


AlgoConfig = Annotated[
    LincAlgoConfig | CmmAlgoConfig | TopicAlgoConfig | SpaceTimeAlgoConfig,
    Field(discriminator="name"),
]


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    algo: AlgoConfig

    @model_validator(mode="after")
    def _couple_algo_and_data(self):
        allowed_settings_by_algo = {
            "topic": {"single", "mixed"},
            "cmm": {"single", "mixed"},
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


TemporalDataConfig = SingleTemporalDataConfig | MultiTemporalDataConfig
ContextDataConfig = MultiDataConfig | MultiTemporalDataConfig | MixedDataConfig
