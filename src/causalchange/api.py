from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from causalchange.causal_change import CausalChange
from causalchange.config.causal_change_config import CausalChangeConfigTabular, CausalChangeConfigTemporal
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
    PostprocessingMode,
    ScoreType,
    StatisticalTestingMethod,
    TabularContextMethod,
    TabularContextMode,
)

ScoreName = Literal["lin", "gam", "spline", "krr", "gp", "ff"]
PostprocessingName = Literal["skip", "edge-strengths"]
ContextCombinationName = Literal["linc", "chain"]
TemporalDataModeName = Literal["time", "time-contexts"]
ChangepointModeName = Literal["skip", "fixed", "detect"]
ChangepointScopeName = Literal["skip", "global", "per-context"]
ChangepointMethodName = Literal["skip", "pelt"]
ClusteringScopeName = Literal["skip", "regimes", "contexts", "regimes-contexts"]
ClusteringMethodName = Literal["skip", "statistical-testing", "mechanism-clustering"]
TestingMethodName = Literal["skip", "kernel", "none"]
PeltPenaltyName = Literal["auto", "bic", "mbic"]

PublicScore = ScoreName | ScoreType | GPType
PublicPostprocessing = PostprocessingName | PostprocessingMode


_SCORE_BY_NAME: dict[str, ScoreType | GPType] = {
    "lin": ScoreType.LIN,
    "gam": ScoreType.GAM,
    "spline": ScoreType.SPLINE,
    "krr": ScoreType.KRR,
    "gp": GPType.EXACT,
    "ff": GPType.FOURIER,
}
_POSTPROCESSING_BY_NAME: dict[str, PostprocessingMode] = {
    "skip": PostprocessingMode.SKIP,
    "edge-strengths": PostprocessingMode.EDGE_STRENGTHS,
}
_CONTEXT_METHOD_BY_NAME: dict[str, TabularContextMethod] = {
    "linc": TabularContextMethod.LINC,
    "chain": TabularContextMethod.CHAIN,
}
_TEMPORAL_DATA_MODE_BY_NAME: dict[str, DataMode] = {
    "time": DataMode.TIME,
    "time-contexts": DataMode.TIME_CONTEXTS,
}
_CHANGEPOINT_MODE_BY_NAME: dict[str, ChangepointMode] = {
    "skip": ChangepointMode.SKIP,
    "fixed": ChangepointMode.ORACLE,
    "detect": ChangepointMode.DETECT,
}
_CHANGEPOINT_SCOPE_BY_NAME: dict[str, ChangepointScope] = {
    "skip": ChangepointScope.SKIP,
    "global": ChangepointScope.GLOBAL,
    "per-context": ChangepointScope.PER_CONTEXT,
}
_CHANGEPOINT_METHOD_BY_NAME: dict[str, ChangepointMethod] = {
    "skip": ChangepointMethod.SKIP,
    "pelt": ChangepointMethod.PELT,
}
_CLUSTERING_SCOPE_BY_NAME: dict[str, MechanismClusteringScope] = {
    "skip": MechanismClusteringScope.SKIP,
    "regimes": MechanismClusteringScope.REGIMES,
    "contexts": MechanismClusteringScope.CONTEXTS,
    "regimes-contexts": MechanismClusteringScope.REGIMES_CONTEXTS,
}
_CLUSTERING_METHOD_BY_NAME: dict[str, MechanismClusteringMethod] = {
    "skip": MechanismClusteringMethod.SKIP,
    "statistical-testing": MechanismClusteringMethod.TESTING,
    "mechanism-clustering": MechanismClusteringMethod.CLUSTERING,
}
_TESTING_METHOD_BY_NAME: dict[str, StatisticalTestingMethod] = {
    "skip": StatisticalTestingMethod.SKIP,
    "kernel": StatisticalTestingMethod.KERNEL,
    "none": StatisticalTestingMethod.NONE,
}


def _score_type(value: PublicScore) -> ScoreType | GPType:
    if isinstance(value, GPType):
        return value
    if isinstance(value, ScoreType):
        if value in {ScoreType.SKIP, ScoreType.GP, ScoreType.MIX}:
            raise ValueError("score_type must be a concrete score: 'lin', 'gam', 'spline', 'krr', 'gp', or 'ff'.")
        return value
    return _SCORE_BY_NAME[value]


def _postprocessing_mode(value: PublicPostprocessing) -> PostprocessingMode:
    return value if isinstance(value, PostprocessingMode) else _POSTPROCESSING_BY_NAME[value]


def _context_method(value: ContextCombinationName | TabularContextMethod) -> TabularContextMethod:
    return value if isinstance(value, TabularContextMethod) else _CONTEXT_METHOD_BY_NAME[value]


def _temporal_data_mode(value: TemporalDataModeName | DataMode) -> DataMode:
    if isinstance(value, DataMode):
        if not value.is_temporal():
            raise ValueError("SpaceTime data_mode must be 'time' or 'time-contexts'.")
        return value
    return _TEMPORAL_DATA_MODE_BY_NAME[value]


def _changepoint_mode(value: ChangepointModeName | ChangepointMode) -> ChangepointMode:
    return value if isinstance(value, ChangepointMode) else _CHANGEPOINT_MODE_BY_NAME[value]


def _changepoint_scope(value: ChangepointScopeName | ChangepointScope) -> ChangepointScope:
    return value if isinstance(value, ChangepointScope) else _CHANGEPOINT_SCOPE_BY_NAME[value]


def _changepoint_method(value: ChangepointMethodName | ChangepointMethod) -> ChangepointMethod:
    return value if isinstance(value, ChangepointMethod) else _CHANGEPOINT_METHOD_BY_NAME[value]


def _clustering_scope(value: ClusteringScopeName | MechanismClusteringScope) -> MechanismClusteringScope:
    return value if isinstance(value, MechanismClusteringScope) else _CLUSTERING_SCOPE_BY_NAME[value]


def _clustering_method(value: ClusteringMethodName | MechanismClusteringMethod) -> MechanismClusteringMethod:
    return value if isinstance(value, MechanismClusteringMethod) else _CLUSTERING_METHOD_BY_NAME[value]


def _testing_method(value: TestingMethodName | StatisticalTestingMethod) -> StatisticalTestingMethod:
    return value if isinstance(value, StatisticalTestingMethod) else _TESTING_METHOD_BY_NAME[value]


class _AlgorithmConfig(BaseModel):
    """Shared public options for named algorithm wrappers.

    These public models intentionally accept short strings. They convert to the internal
    enum-based CausalChangeConfig objects before CausalChange sees them.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    score_type: PublicScore = "gam"
    postprocessing_mode: PublicPostprocessing = "skip"
    score_kwargs: dict[str, Any] = Field(default_factory=dict)
    seed: NonNegativeInt = 42


class TopicConfig(_AlgorithmConfig):
    """Public config for TOPIC on one tabular dataset."""

    def to_causal_change_config(self) -> CausalChangeConfigTabular:
        return CausalChangeConfigTabular(
            data_mode=DataMode.TABULAR,
            graph_search=GraphSearch.TOPIC,
            score_type=_score_type(self.score_type),
            postprocessing_mode=_postprocessing_mode(self.postprocessing_mode),
            score_kwargs=dict(self.score_kwargs),
            seed=self.seed,
        )


class LincConfig(_AlgorithmConfig):
    """Public config for LINC on observed multi-context tabular data."""

    context_col: str = Field("context", min_length=1)
    context_combination_method: ContextCombinationName | TabularContextMethod = "linc"
    context_combination_kwargs: ContextCombinationKwargs = Field(default_factory=ContextCombinationKwargs)

    @model_validator(mode="after")
    def _validate_linc_name(self):
        if _context_method(self.context_combination_method) != TabularContextMethod.LINC:
            raise ValueError("Linc uses context_combination_method='linc'. Use CausalChange.tabular for CHAIN.")
        return self

    def to_causal_change_config(self) -> CausalChangeConfigTabular:
        return CausalChangeConfigTabular(
            data_mode=DataMode.TAB_CONTEXTS,
            graph_search=GraphSearch.TOPIC,
            score_type=_score_type(self.score_type),
            context_mode=TabularContextMode.ORACLE,
            context_combination_method=_context_method(self.context_combination_method),
            context_combination_kwargs=self.context_combination_kwargs,
            context_col=self.context_col,
            postprocessing_mode=_postprocessing_mode(self.postprocessing_mode),
            score_kwargs=dict(self.score_kwargs),
            seed=self.seed,
        )


class SpaceTimeConfig(_AlgorithmConfig):
    """Public config for SpaceTime on temporal or multi-context temporal data."""

    data_mode: TemporalDataModeName | DataMode = "time-contexts"
    tau_max: PositiveInt = 2
    d_min: PositiveInt = 30
    max_iter: PositiveInt = 3
    mechanism_test_alpha: float = Field(0.05, gt=0.0, lt=1.0)
    pelt_penalty: PositiveFloat | PeltPenaltyName = "auto"
    context_col: str = Field("context", min_length=1)

    changepoint_mode: ChangepointModeName | ChangepointMode = "detect"
    changepoint_scope: ChangepointScopeName | ChangepointScope = "global"
    changepoint_method: ChangepointMethodName | ChangepointMethod = "pelt"

    clustering_scope: ClusteringScopeName | MechanismClusteringScope = "regimes-contexts"
    clustering_method: ClusteringMethodName | MechanismClusteringMethod = "statistical-testing"
    testing_method: TestingMethodName | StatisticalTestingMethod = "kernel"

    fixed_changepoints: list[NonNegativeInt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_public_spacetime_options(self):
        data_mode = _temporal_data_mode(self.data_mode)
        clustering_scope = _clustering_scope(self.clustering_scope)
        changepoint_scope = _changepoint_scope(self.changepoint_scope)

        if data_mode == DataMode.TIME and clustering_scope in {
            MechanismClusteringScope.CONTEXTS,
            MechanismClusteringScope.REGIMES_CONTEXTS,
        }:
            raise ValueError(
                "data_mode='time' cannot use context clustering. Use clustering_scope='skip' or 'regimes', "
                "or set data_mode='time-contexts'."
            )

        if data_mode == DataMode.TIME and changepoint_scope == ChangepointScope.PER_CONTEXT:
            raise ValueError("changepoint_scope='per-context' requires data_mode='time-contexts'.")

        return self

    def to_causal_change_config(self) -> CausalChangeConfigTemporal:
        return CausalChangeConfigTemporal(
            data_mode=_temporal_data_mode(self.data_mode),
            graph_search=GraphSearch.GLOBE,
            score_type=_score_type(self.score_type),
            context_col=self.context_col,
            tau_max=self.tau_max,
            d_min=self.d_min,
            max_iter=self.max_iter,
            mechanism_test_alpha=self.mechanism_test_alpha,
            pelt_penalty=self.pelt_penalty,
            fixed_changepoints=list(self.fixed_changepoints),
            changepoint_mode=_changepoint_mode(self.changepoint_mode),
            changepoint_scope=_changepoint_scope(self.changepoint_scope),
            changepoint_method=_changepoint_method(self.changepoint_method),
            clustering_scope=_clustering_scope(self.clustering_scope),
            clustering_method=_clustering_method(self.clustering_method),
            testing_method=_testing_method(self.testing_method),
            postprocessing_mode=_postprocessing_mode(self.postprocessing_mode),
            score_kwargs=dict(self.score_kwargs),
            seed=self.seed,
        )


class Topic(CausalChange):
    """TOPIC causal discovery for one tabular dataset.

    Public options use short strings, e.g. `score_type='gam'` instead of `ScoreType.GAM`.
    """

    public_config_: TopicConfig

    def __init__(
        self,
        *,
        score_type: PublicScore = "gam",
        postprocessing_mode: PublicPostprocessing = "skip",
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
        score_type: PublicScore = "gam",
        context_col: str = "context",
        postprocessing_mode: PublicPostprocessing = "skip",
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
            context_combination_method="linc",
            context_combination_kwargs=(
                ContextCombinationKwargs() if context_combination_kwargs is None else context_combination_kwargs
            ),
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
        score_type: PublicScore = "gam",
        data_mode: TemporalDataModeName | DataMode = "time-contexts",
        tau_max: int = 2,
        context_col: str = "context",
        changepoint_mode: ChangepointModeName | ChangepointMode = "detect",
        changepoint_scope: ChangepointScopeName | ChangepointScope | None = None,
        changepoint_method: ChangepointMethodName | ChangepointMethod | None = None,
        clustering_scope: ClusteringScopeName | MechanismClusteringScope | None = None,
        clustering_method: ClusteringMethodName | MechanismClusteringMethod | None = None,
        testing_method: TestingMethodName | StatisticalTestingMethod | None = None,
        d_min: int = 30,
        max_iter: int = 3,
        pelt_penalty: PositiveFloat | PeltPenaltyName = "auto",
        mechanism_test_alpha: float = 0.05,
        fixed_changepoints: list[int] | None = None,
        postprocessing_mode: PublicPostprocessing = "skip",
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        dm = _temporal_data_mode(data_mode)
        cp_mode = _changepoint_mode(changepoint_mode)

        resolved_changepoint_scope: ChangepointScopeName | ChangepointScope
        if changepoint_scope is None:
            resolved_changepoint_scope = "skip" if cp_mode == ChangepointMode.SKIP else "global"
        else:
            resolved_changepoint_scope = changepoint_scope

        resolved_changepoint_method: ChangepointMethodName | ChangepointMethod
        if changepoint_method is None:
            resolved_changepoint_method = "pelt" if cp_mode == ChangepointMode.DETECT else "skip"
        else:
            resolved_changepoint_method = changepoint_method

        resolved_clustering_scope: ClusteringScopeName | MechanismClusteringScope
        if clustering_scope is None:
            if cp_mode == ChangepointMode.SKIP:
                resolved_clustering_scope = "skip"
            elif dm == DataMode.TIME_CONTEXTS:
                resolved_clustering_scope = "regimes-contexts"
            else:
                resolved_clustering_scope = "regimes"
        else:
            resolved_clustering_scope = clustering_scope

        cl_scope = _clustering_scope(resolved_clustering_scope)

        resolved_clustering_method: ClusteringMethodName | MechanismClusteringMethod
        if clustering_method is None:
            resolved_clustering_method = "skip" if cl_scope == MechanismClusteringScope.SKIP else "statistical-testing"
        else:
            resolved_clustering_method = clustering_method

        cl_method = _clustering_method(resolved_clustering_method)

        resolved_testing_method: TestingMethodName | StatisticalTestingMethod
        if testing_method is None:
            resolved_testing_method = "kernel" if cl_method == MechanismClusteringMethod.TESTING else "skip"
        else:
            resolved_testing_method = testing_method

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
    "SpaceTime",
    "TopicConfig",
    "LincConfig",
    "SpaceTimeConfig",
    "ScoreName",
    "TemporalDataModeName",
]
