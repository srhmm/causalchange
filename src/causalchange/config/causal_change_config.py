from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causalchange.core.types import (
    ChangepointMethod,
    ChangepointMode,
    ChangepointScope,
    ContextCombinationKwargs,
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


class CausalChangeConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_mode: DataMode
    graph_search: GraphSearch
    score_type: ScoreType | GPType
    postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP
    score_kwargs: dict[str, Any] = Field(default_factory=dict)  # typify ScoreKwargs
    seed: int = 42


class CausalChangeConfigTabular(CausalChangeConfigBase):
    mix_type: MixedSCMType = MixedSCMType.SKIP
    # mix_kwargs: dict[str, Any] = Field(default_factory=dict) #typify MixKwargs

    context_mode: TabularContextMode = TabularContextMode.SKIP
    context_combination_method: TabularContextMethod = TabularContextMethod.SKIP
    context_combination_kwargs: ContextCombinationKwargs = Field(default_factory=ContextCombinationKwargs)

    context_col: str = "context"

    @model_validator(mode="after")
    def _validate_combo(self):
        if self.data_mode.is_temporal():
            raise ValueError("Use CausalChangeConfigTemporal for temporal data.")
        if self.data_mode not in self.graph_search.compatible_data_modes():
            raise ValueError(f"{self.graph_search=} not compatible with {self.data_mode=}.")
        if self.data_mode not in self.context_mode.compatible_data_modes():
            raise ValueError(f"{self.context_mode=} not compatible with {self.data_mode=}.")
        if self.data_mode not in self.context_combination_method.compatible_data_modes():
            raise ValueError(f"{self.context_combination_method=} not compatible with {self.data_mode=}.")
        if self.data_mode == DataMode.TAB_CONTEXTS and self.context_col is None or not self.context_col:
            raise ValueError("non-empty context_col required for TAB_CONTEXTS.")
        if self.context_mode == TabularContextMode.DETECT and self.mix_type == MixedSCMType.SKIP:
            raise ValueError("MixingMethod required for mixed tabular data")
        if self.context_mode != TabularContextMode.DETECT and self.mix_type != MixedSCMType.SKIP:
            raise ValueError("mix_type is only valid when context_mode=DETECT")
        if self.score_type == ScoreType.GP:
            raise ValueError("decide GPType.EXACT or GPType.FOURIER")
        if (
            self.context_combination_method != TabularContextMethod.SKIP
            and self.context_mode != TabularContextMode.ORACLE
        ):
            raise ValueError("TabularContextMethods need observed tabular contexts")
        return self


class CausalChangeConfigTemporal(CausalChangeConfigBase):
    changepoint_mode: ChangepointMode = ChangepointMode.SKIP
    changepoint_scope: ChangepointScope = ChangepointScope.SKIP
    changepoint_method: ChangepointMethod = ChangepointMethod.SKIP

    clustering_scope: MechanismClusteringScope = MechanismClusteringScope.SKIP
    clustering_method: MechanismClusteringMethod = MechanismClusteringMethod.SKIP
    testing_method: StatisticalTestingMethod = StatisticalTestingMethod.SKIP

    tau_max: int = 1
    d_min: int = 30
    max_iter: int = 3
    mechanism_test_alpha: float = 0.05
    pelt_penalty: float | Literal["bic", "mbic", "auto"] = "auto"
    context_col: str = "context"

    fixed_changepoints: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_temporal(self):
        if not self.data_mode.is_temporal():
            raise ValueError("Use CausalChangeConfigTabular for non-temporal data.")

        if self.score_type == ScoreType.GP:
            raise ValueError("decide GPType.EXACT or GPType.FOURIER")

        if self.data_mode not in self.graph_search.compatible_data_modes():
            raise ValueError(f"{self.graph_search=} not compatible with {self.data_mode=}.")

        if self.data_mode == DataMode.TIME_CONTEXTS and self.context_col is None or not self.context_col:
            raise ValueError("non-empty context_col required for TIME_CONTEXTS.")

        if self.tau_max <= 0:
            raise ValueError("tau_max must be positive.")

        if self.d_min <= 0:
            raise ValueError("d_min must be positive.")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if not 0.0 < self.mechanism_test_alpha < 1.0:
            raise ValueError("mechanism_test_alpha must be in (0, 1).")

        if isinstance(self.pelt_penalty, str):
            if self.pelt_penalty not in {"bic", "mbic", "auto"}:
                raise ValueError("pelt_penalty must be a positive number or one of {'bic', 'mbic', 'auto'}.")
        elif self.pelt_penalty <= 0:
            raise ValueError("pelt_penalty must be positive.")

        if self.changepoint_mode == ChangepointMode.SKIP:
            if self.fixed_changepoints:
                raise ValueError("fixed_changepoints requires changepoint_mode=ORACLE.")
            if self.changepoint_scope != ChangepointScope.SKIP:
                raise ValueError("changepoint_scope is only valid when changepoint_mode is not SKIP.")
            if self.changepoint_method != ChangepointMethod.SKIP:
                raise ValueError("changepoint_method is only valid when changepoint_mode=DETECT.")

        if self.changepoint_mode == ChangepointMode.ORACLE:
            if not self.fixed_changepoints:
                raise ValueError("fixed_changepoints must be provided when changepoint_mode=ORACLE.")
            if self.changepoint_method != ChangepointMethod.SKIP:
                raise ValueError("changepoint_method is only valid when changepoint_mode=DETECT.")

        if self.changepoint_mode == ChangepointMode.DETECT:
            if self.changepoint_scope == ChangepointScope.SKIP:
                raise ValueError("changepoint_scope is required when changepoint_mode=DETECT.")
            if self.changepoint_method == ChangepointMethod.SKIP:
                raise ValueError("changepoint_method is required when changepoint_mode=DETECT.")
            if self.fixed_changepoints:
                raise ValueError("fixed_changepoints is only valid when changepoint_mode=ORACLE.")

        if self.changepoint_scope == ChangepointScope.PER_CONTEXT and self.data_mode != DataMode.TIME_CONTEXTS:
            raise ValueError("PER_CONTEXT changepoints require data_mode=TIME_CONTEXTS.")

        uses_regimes = self.clustering_scope in {
            MechanismClusteringScope.REGIMES,
            MechanismClusteringScope.REGIMES_CONTEXTS,
        }
        uses_contexts = self.clustering_scope in {
            MechanismClusteringScope.CONTEXTS,
            MechanismClusteringScope.REGIMES_CONTEXTS,
        }

        if self.clustering_scope == MechanismClusteringScope.SKIP:
            if self.clustering_method != MechanismClusteringMethod.SKIP:
                raise ValueError("clustering_method is only valid when clustering_scope is not SKIP.")
            if self.testing_method != StatisticalTestingMethod.SKIP:
                raise ValueError("testing_method is only valid when clustering_method=TESTING.")

        else:
            if self.clustering_method == MechanismClusteringMethod.SKIP:
                raise ValueError("clustering_method is required when clustering_scope is not SKIP.")

            if uses_regimes and self.changepoint_mode == ChangepointMode.SKIP:
                raise ValueError("Regime clustering requires changepoints. Use changepoint_mode=ORACLE or DETECT.")

            if uses_contexts and not self.data_mode.is_context():
                raise ValueError("Context clustering requires data_mode=TIME_CONTEXTS.")

            if self.clustering_method == MechanismClusteringMethod.TESTING:
                if self.testing_method == StatisticalTestingMethod.SKIP:
                    raise ValueError("testing_method is required when clustering_method=TESTING.")
            elif self.testing_method != StatisticalTestingMethod.SKIP:
                raise ValueError("testing_method is only valid when clustering_method=TESTING.")

        return self


CausalChangeConfig = CausalChangeConfigTabular | CausalChangeConfigTemporal
