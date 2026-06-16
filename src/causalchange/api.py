from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from causalchange.causal_change import CausalChange
from causalchange.config.causal_change_config import (
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
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

PeltPenaltyName = Literal["auto", "bic", "mbic"]

E = TypeVar("E", bound=Enum)


def _as_enum(enum_cls: type[E], value: E | str) -> E:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


class _AlgorithmConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    score_type: ScoreType | GPType = ScoreType.GAM
    postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP
    score_kwargs: dict[str, Any] = Field(default_factory=dict)
    seed: NonNegativeInt = 42

    @field_validator("score_type")
    @classmethod
    def _validate_concrete_score_type(cls, value: ScoreType | GPType) -> ScoreType | GPType:
        if value in {ScoreType.SKIP, ScoreType.GP, ScoreType.MIX}:
            raise ValueError("score_type must be a concrete score: " "'lin', 'gam', 'spline', 'krr', 'gp', or 'ff'.")
        return value


class TopicConfig(_AlgorithmConfig):
    """Public config for TOPIC on one tabular dataset."""

    def to_causal_change_config(self) -> CausalChangeConfigTabular:
        return CausalChangeConfigTabular(
            data_mode=DataMode.TABULAR,
            graph_search=GraphSearch.TOPIC,
            score_type=self.score_type,
            postprocessing_mode=self.postprocessing_mode,
            score_kwargs=dict(self.score_kwargs),
            seed=self.seed,
        )


class LincConfig(_AlgorithmConfig):
    """Public config for LINC on observed multi-context tabular data."""

    context_col: str = Field("context", min_length=1)
    context_combination_method: TabularContextMethod = TabularContextMethod.LINC
    context_combination_kwargs: ContextCombinationKwargs = Field(default_factory=ContextCombinationKwargs)

    @model_validator(mode="after")
    def _validate_linc_name(self):
        if self.context_combination_method != TabularContextMethod.LINC:
            raise ValueError("Linc uses context_combination_method='linc'. Use CausalChange directly for CHAIN.")
        return self

    def to_causal_change_config(self) -> CausalChangeConfigTabular:
        return CausalChangeConfigTabular(
            data_mode=DataMode.TAB_CONTEXTS,
            graph_search=GraphSearch.TOPIC,
            score_type=self.score_type,
            context_mode=TabularContextMode.ORACLE,
            context_combination_method=self.context_combination_method,
            context_combination_kwargs=self.context_combination_kwargs,
            context_col=self.context_col,
            postprocessing_mode=self.postprocessing_mode,
            score_kwargs=dict(self.score_kwargs),
            seed=self.seed,
        )


class CMMConfig(_AlgorithmConfig):
    """Public config for CMM: TOPIC search with mixture-regression local scoring."""

    score_type: ScoreType | GPType = ScoreType.LIN
    mix_type: MixedSCMType = MixedSCMType.LIN
    k_max: PositiveInt = 5
    lambda_mix: float = Field(1.0, ge=0.0)
    hybrid_mixing: bool = True

    @field_validator("mix_type")
    @classmethod
    def _validate_mix_type(cls, value: MixedSCMType) -> MixedSCMType:
        if value == MixedSCMType.SKIP:
            raise ValueError("CMM requires mix_type to be one of 'lin', 'quadratic', 'cubic', 'nspline', 'bspline'.")
        return value

    def to_causal_change_config(self) -> CausalChangeConfigTabular:
        score_kwargs = dict(self.score_kwargs)
        score_kwargs.update(
            {
                "k_max": int(self.k_max),
                "lambda_mix": float(self.lambda_mix),
                "hybrid_mixing": bool(self.hybrid_mixing),
            }
        )

        return CausalChangeConfigTabular(
            data_mode=DataMode.TABULAR,
            graph_search=GraphSearch.TOPIC,
            score_type=self.score_type,
            context_mode=TabularContextMode.SKIP,
            mix_type=self.mix_type,
            postprocessing_mode=self.postprocessing_mode,
            score_kwargs=score_kwargs,
            seed=self.seed,
        )


class SpaceTimeConfig(_AlgorithmConfig):
    """Public config for SpaceTime on temporal or multi-context temporal data."""

    data_mode: DataMode = DataMode.TIME_CONTEXTS
    tau_max: PositiveInt = 2
    d_min: PositiveInt = 30
    max_iter: PositiveInt = 3
    mechanism_test_alpha: float = Field(0.05, gt=0.0, lt=1.0)
    pelt_penalty: PositiveFloat | PeltPenaltyName = "auto"
    context_col: str = Field("context", min_length=1)

    changepoint_mode: ChangepointMode = ChangepointMode.DETECT
    changepoint_scope: ChangepointScope = ChangepointScope.GLOBAL
    changepoint_method: ChangepointMethod = ChangepointMethod.PELT

    clustering_scope: MechanismClusteringScope = MechanismClusteringScope.REGIMES_CONTEXTS
    clustering_method: MechanismClusteringMethod = MechanismClusteringMethod.TESTING
    testing_method: StatisticalTestingMethod = StatisticalTestingMethod.KERNEL

    fixed_changepoints: list[NonNegativeInt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_public_spacetime_options(self):
        if not self.data_mode.is_temporal():
            raise ValueError("SpaceTime data_mode must be DataMode.TIME or DataMode.TIME_CONTEXTS.")

        if self.data_mode == DataMode.TIME and self.clustering_scope in {
            MechanismClusteringScope.CONTEXTS,
            MechanismClusteringScope.REGIMES_CONTEXTS,
        }:
            raise ValueError(
                "data_mode=DataMode.TIME cannot use context clustering. "
                "Use clustering_scope='skip' or 'regimes', or set data_mode='time-contexts'."
            )

        if self.data_mode == DataMode.TIME and self.changepoint_scope == ChangepointScope.PER_CONTEXT:
            raise ValueError("changepoint_scope='per-context' requires data_mode='time-contexts'.")

        return self

    def to_causal_change_config(self) -> CausalChangeConfigTemporal:
        return CausalChangeConfigTemporal(
            data_mode=self.data_mode,
            graph_search=GraphSearch.GLOBE,
            score_type=self.score_type,
            context_col=self.context_col,
            tau_max=self.tau_max,
            d_min=self.d_min,
            max_iter=self.max_iter,
            mechanism_test_alpha=self.mechanism_test_alpha,
            pelt_penalty=self.pelt_penalty,
            fixed_changepoints=list(self.fixed_changepoints),
            changepoint_mode=self.changepoint_mode,
            changepoint_scope=self.changepoint_scope,
            changepoint_method=self.changepoint_method,
            clustering_scope=self.clustering_scope,
            clustering_method=self.clustering_method,
            testing_method=self.testing_method,
            postprocessing_mode=self.postprocessing_mode,
            score_kwargs=dict(self.score_kwargs),
            seed=self.seed,
        )


class Topic(CausalChange):
    """TOPIC causal discovery for one tabular dataset."""

    public_config_: TopicConfig

    def __init__(
        self,
        *,
        score_type: ScoreType | GPType = ScoreType.GAM,
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        public_cfg = TopicConfig(
            score_type=score_type,
            postprocessing_mode=postprocessing_mode,
            score_kwargs={} if score_kwargs is None else dict(score_kwargs),
            seed=seed,
        )
        self.public_config_ = public_cfg
        super().__init__(public_cfg.to_causal_change_config(), var_nms=var_nms, lg=lg, vb=vb)


class Linc(CausalChange):
    """LINC causal discovery for observed multi-context tabular data."""

    public_config_: LincConfig

    def __init__(
        self,
        *,
        score_type: ScoreType | GPType = ScoreType.GAM,
        context_col: str = "context",
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
        context_combination_kwargs: ContextCombinationKwargs | None = None,
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        public_cfg = LincConfig(
            score_type=score_type,
            context_col=context_col,
            context_combination_method=TabularContextMethod.LINC,
            context_combination_kwargs=(
                ContextCombinationKwargs() if context_combination_kwargs is None else context_combination_kwargs
            ),
            postprocessing_mode=postprocessing_mode,
            score_kwargs={} if score_kwargs is None else dict(score_kwargs),
            seed=seed,
        )
        self.public_config_ = public_cfg
        super().__init__(public_cfg.to_causal_change_config(), var_nms=var_nms, lg=lg, vb=vb)


class CMM(CausalChange):
    """CMM causal discovery: TOPIC search with mixture-regression local score."""

    public_config_: CMMConfig

    def __init__(
        self,
        *,
        mix_type: MixedSCMType = MixedSCMType.LIN,
        k_max: int = 5,
        lambda_mix: float = 1.0,
        hybrid_mixing: bool = True,
        score_type: ScoreType | GPType = ScoreType.LIN,
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        public_cfg = CMMConfig(
            score_type=score_type,
            mix_type=mix_type,
            k_max=k_max,
            lambda_mix=lambda_mix,
            hybrid_mixing=hybrid_mixing,
            postprocessing_mode=postprocessing_mode,
            score_kwargs={} if score_kwargs is None else dict(score_kwargs),
            seed=seed,
        )
        self.public_config_ = public_cfg
        super().__init__(public_cfg.to_causal_change_config(), var_nms=var_nms, lg=lg, vb=vb)


class SpaceTime(CausalChange):
    """SpaceTime causal discovery for temporal or multi-context temporal data."""

    public_config_: SpaceTimeConfig

    def __init__(
        self,
        *,
        score_type: ScoreType | GPType = ScoreType.GAM,
        data_mode: DataMode = DataMode.TIME_CONTEXTS,
        tau_max: int = 2,
        context_col: str = "context",
        changepoint_mode: ChangepointMode = ChangepointMode.DETECT,
        changepoint_scope: ChangepointScope | None = None,
        changepoint_method: ChangepointMethod | None = None,
        clustering_scope: MechanismClusteringScope | None = None,
        clustering_method: MechanismClusteringMethod | None = None,
        testing_method: StatisticalTestingMethod | None = None,
        d_min: int = 30,
        max_iter: int = 3,
        pelt_penalty: PositiveFloat | PeltPenaltyName = "auto",
        mechanism_test_alpha: float = 0.05,
        fixed_changepoints: list[int] | None = None,
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        dm = _as_enum(DataMode, data_mode)
        cp_mode = _as_enum(ChangepointMode, changepoint_mode)

        if not dm.is_temporal():
            raise ValueError("SpaceTime data_mode must be DataMode.TIME or DataMode.TIME_CONTEXTS.")

        resolved_changepoint_scope = (
            ChangepointScope.SKIP
            if cp_mode == ChangepointMode.SKIP
            else ChangepointScope.GLOBAL
            if changepoint_scope is None
            else changepoint_scope
        )

        resolved_changepoint_method = (
            ChangepointMethod.PELT
            if cp_mode == ChangepointMode.DETECT
            else ChangepointMethod.SKIP
            if changepoint_method is None
            else changepoint_method
        )

        if clustering_scope is None:
            if cp_mode == ChangepointMode.SKIP:
                resolved_clustering_scope = MechanismClusteringScope.SKIP
            elif dm == DataMode.TIME_CONTEXTS:
                resolved_clustering_scope = MechanismClusteringScope.REGIMES_CONTEXTS
            else:
                resolved_clustering_scope = MechanismClusteringScope.REGIMES
        else:
            resolved_clustering_scope = clustering_scope

        cl_scope = _as_enum(MechanismClusteringScope, resolved_clustering_scope)

        resolved_clustering_method = (
            MechanismClusteringMethod.SKIP
            if cl_scope == MechanismClusteringScope.SKIP
            else MechanismClusteringMethod.TESTING
            if clustering_method is None
            else clustering_method
        )

        cl_method = _as_enum(MechanismClusteringMethod, resolved_clustering_method)

        resolved_testing_method = (
            StatisticalTestingMethod.KERNEL
            if cl_method == MechanismClusteringMethod.TESTING
            else StatisticalTestingMethod.SKIP
            if testing_method is None
            else testing_method
        )

        public_cfg = SpaceTimeConfig(
            score_type=score_type,
            data_mode=dm,
            tau_max=tau_max,
            context_col=context_col,
            changepoint_mode=cp_mode,
            changepoint_scope=resolved_changepoint_scope,
            changepoint_method=resolved_changepoint_method,
            clustering_scope=resolved_clustering_scope,
            clustering_method=resolved_clustering_method,
            testing_method=resolved_testing_method,
            d_min=d_min,
            max_iter=max_iter,
            pelt_penalty=pelt_penalty,
            mechanism_test_alpha=mechanism_test_alpha,
            fixed_changepoints=[] if fixed_changepoints is None else list(fixed_changepoints),
            postprocessing_mode=postprocessing_mode,
            score_kwargs={} if score_kwargs is None else dict(score_kwargs),
            seed=seed,
        )
        self.public_config_ = public_cfg
        super().__init__(public_cfg.to_causal_change_config(), var_nms=var_nms, lg=lg, vb=vb)


__all__ = [
    "Topic",
    "Linc",
    "CMM",
    "SpaceTime",
    "TopicConfig",
    "LincConfig",
    "CMMConfig",
    "SpaceTimeConfig",
]
