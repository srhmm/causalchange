from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from causalchange.core.types import (
    ChangepointMethod,
    ChangepointMode,
    ChangepointScope,
    ClusteringMethod,
    DataMode,
    GraphSearch,
    MechanismClusteringMethod,
    MechanismClusteringScope,
    MixedSCMType,
    PostprocessingMode,
    ScoreType,
    StatisticalTestingMethod,
    TabularContextMethod,
    TabularContextMode,
    TabularMechanismClusteringMethod,
)

Alpha = Annotated[float, Field(gt=0.0, lt=1.0)]
ContextColumn = Annotated[str, Field(min_length=1)]
PeltPenalty = PositiveFloat | Literal["bic", "mbic", "auto"]


class CausalChangeConfigBase(BaseModel):
    """validated configuration for discovery"""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    data_mode: DataMode
    graph_search: GraphSearch
    score_type: ScoreType
    postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP
    score_kwargs: dict[str, Any] = Field(default_factory=dict)
    seed: NonNegativeInt = 42

    @model_validator(mode="after")
    def _validate_base_options(self):
        if self.data_mode == DataMode.SKIP:
            raise ValueError("data_mode is required.")

        if self.graph_search == GraphSearch.SKIP:
            raise ValueError("graph_search is required.")

        if self.score_type == ScoreType.SKIP:
            raise ValueError("score_type is required.")

        if self.data_mode not in self.graph_search.compatible_data_modes():
            raise ValueError(f"{self.graph_search=} is not compatible with {self.data_mode=}.")

        return self


class CausalChangeConfigTabular(CausalChangeConfigBase):
    """configuration for tabular and multi-context tabular discovery"""

    mix_type: MixedSCMType = MixedSCMType.SKIP
    context_mode: TabularContextMode = TabularContextMode.SKIP
    context_combination_method: TabularContextMethod = TabularContextMethod.SKIP
    context_combination_kwargs: ClusteringMethod = Field(default_factory=ClusteringMethod)
    context_col: ContextColumn = "context"
    context_gain_threshold: float = Field(default=0.0)

    mechanism_clustering_method: TabularMechanismClusteringMethod = TabularMechanismClusteringMethod.SCORE_MERGE
    testing_method: StatisticalTestingMethod = StatisticalTestingMethod.SKIP
    mechanism_test_alpha: Alpha = 0.05
    mechanism_clustering_n_clusters: PositiveInt | None = None
    mechanism_clustering_distance_threshold: PositiveFloat | None = None

    @model_validator(mode="after")
    def _validate_tabular_compatibility(self):
        if self.data_mode.is_temporal():
            raise ValueError("Use CausalChangeConfigTemporal for temporal data.")

        if self.data_mode not in self.context_mode.compatible_data_modes():
            raise ValueError(f"{self.context_mode=} is not compatible with {self.data_mode=}.")

        if self.data_mode not in self.context_combination_method.compatible_data_modes():
            raise ValueError(f"{self.context_combination_method=} is not compatible with {self.data_mode=}.")

        uses_hidden_context_detection = self.context_mode == TabularContextMode.DETECT

        uses_cmm_scoring = (
            self.data_mode == DataMode.TABULAR
            and self.graph_search == GraphSearch.TOPIC
            and self.context_mode == TabularContextMode.SKIP
            and self.mix_type != MixedSCMType.SKIP
        )

        if uses_hidden_context_detection and self.mix_type == MixedSCMType.SKIP:
            raise ValueError("mix_type is required when context_mode=DETECT.")

        if self.mix_type != MixedSCMType.SKIP and not uses_hidden_context_detection and not uses_cmm_scoring:
            raise ValueError(
                "mix_type is only valid for context_mode=DETECT or for CMM-style "
                "TOPIC scoring on single tabular data."
            )

        if (
            self.context_combination_method != TabularContextMethod.SKIP
            and self.context_mode != TabularContextMode.ORACLE
        ):
            raise ValueError("context_combination_method requires observed contexts: context_mode=ORACLE.")
        if self.context_combination_method != TabularContextMethod.LINC:
            if self.mechanism_clustering_method != TabularMechanismClusteringMethod.SCORE_MERGE:
                raise ValueError("mechanism_clustering_method other than SCORE_MERGE is only valid for LINC.")
            if self.testing_method != StatisticalTestingMethod.SKIP:
                raise ValueError("testing_method is only valid for LINC statistical-testing.")
            return self

        if self.mechanism_clustering_method == TabularMechanismClusteringMethod.SCORE_MERGE:
            if self.testing_method != StatisticalTestingMethod.SKIP:
                raise ValueError(
                    "testing_method is only valid when " "mechanism_clustering_method='statistical-testing'."
                )

        elif self.mechanism_clustering_method == TabularMechanismClusteringMethod.TESTING:
            if self.testing_method == StatisticalTestingMethod.SKIP:
                raise ValueError(
                    "testing_method is required when " "mechanism_clustering_method='statistical-testing'."
                )

        elif self.mechanism_clustering_method == TabularMechanismClusteringMethod.CLUSTERING:
            if self.testing_method != StatisticalTestingMethod.SKIP:
                raise ValueError(
                    "testing_method is only valid when " "mechanism_clustering_method='statistical-testing'."
                )

        return self
        return self


class CausalChangeConfigTemporal(CausalChangeConfigBase):
    """configuration for single-context and multi-context temporal discovery"""

    changepoint_mode: ChangepointMode = ChangepointMode.SKIP
    changepoint_scope: ChangepointScope = ChangepointScope.SKIP
    changepoint_method: ChangepointMethod = ChangepointMethod.SKIP

    clustering_scope: MechanismClusteringScope = MechanismClusteringScope.SKIP
    clustering_method: MechanismClusteringMethod = MechanismClusteringMethod.SKIP
    testing_method: StatisticalTestingMethod = StatisticalTestingMethod.SKIP

    # missing_mode: MissingMode = MissingMode.OBSERVED

    tau_max: PositiveInt = 1
    d_min: PositiveInt = 30
    max_iter: PositiveInt = 3
    mechanism_test_alpha: Alpha = 0.05
    pelt_penalty: PeltPenalty = "auto"
    context_col: ContextColumn = "context"

    fixed_changepoints: list[NonNegativeInt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_temporal_compatibility(self):
        if not self.data_mode.is_temporal():
            raise ValueError("Use CausalChangeConfigTabular for non-temporal data.")

        self._validate_changepoint_options()
        self._validate_clustering_options()
        return self

    def _validate_changepoint_options(self) -> None:
        if self.changepoint_mode == ChangepointMode.SKIP:
            if self.fixed_changepoints:
                raise ValueError("fixed_changepoints requires changepoint_mode=ORACLE.")
            if self.changepoint_scope != ChangepointScope.SKIP:
                raise ValueError("changepoint_scope is only valid when changepoint_mode is not SKIP.")
            if self.changepoint_method != ChangepointMethod.SKIP:
                raise ValueError("changepoint_method is only valid when changepoint_mode=DETECT.")

        elif self.changepoint_mode == ChangepointMode.ORACLE:
            if not self.fixed_changepoints:
                raise ValueError("fixed_changepoints must be provided when changepoint_mode=ORACLE.")
            if self.changepoint_method != ChangepointMethod.SKIP:
                raise ValueError("changepoint_method is only valid when changepoint_mode=DETECT.")

        elif self.changepoint_mode == ChangepointMode.DETECT:
            if self.changepoint_scope == ChangepointScope.SKIP:
                raise ValueError("changepoint_scope is required when changepoint_mode=DETECT.")
            if self.changepoint_method == ChangepointMethod.SKIP:
                raise ValueError("changepoint_method is required when changepoint_mode=DETECT.")
            if self.fixed_changepoints:
                raise ValueError("fixed_changepoints is only valid when changepoint_mode=ORACLE.")

        if self.changepoint_scope == ChangepointScope.PER_CONTEXT and self.data_mode != DataMode.TIME_CONTEXTS:
            raise ValueError("PER_CONTEXT changepoints require data_mode=TIME_CONTEXTS.")

    def _validate_clustering_options(self) -> None:
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
            return

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


CausalChangeConfig = CausalChangeConfigTabular | CausalChangeConfigTemporal
