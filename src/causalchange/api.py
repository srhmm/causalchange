from __future__ import annotations

import logging
from typing import Any, Literal

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

ScoreName = Literal["lin", "gam", "spline", "krr", "gp", "ff"]
PostprocessingName = Literal["skip", "edge-strengths"]
MixTypeName = Literal["lin", "quadratic", "cubic", "nspline", "bspline"]

DataName = Literal["time", "time-contexts"]
ChangepointModeName = Literal["skip", "fixed", "detect"]
ChangepointScopeName = Literal["skip", "global", "per-context"]
ChangepointMethodName = Literal["skip", "pelt"]
ClusteringScopeName = Literal["skip", "regimes", "contexts", "regimes-contexts"]
ClusteringMethodName = Literal["skip", "statistical-testing", "mechanism-clustering"]
TestingMethodName = Literal["skip", "kernel", "none"]
PeltPenaltyName = Literal["auto", "bic", "mbic"]


class Topic(CausalChange):
    """TOPIC causal discovery for one tabular dataset."""

    public_config_: CausalChangeConfigTabular

    def __init__(
        self,
        *,
        score_type: ScoreName = "gam",
        postprocessing_mode: PostprocessingName = "skip",
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        public_cfg = CausalChangeConfigTabular(
            data_mode=DataMode.TABULAR,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType(score_type),
            postprocessing_mode=PostprocessingMode(postprocessing_mode),
            score_kwargs={} if score_kwargs is None else dict(score_kwargs),
            seed=seed,
        )
        self.public_config_ = public_cfg
        super().__init__(public_cfg, var_nms=var_nms, lg=lg, vb=vb)


class Linc(CausalChange):
    """LINC causal discovery for observed multi-context tabular data."""

    public_config_: CausalChangeConfigTabular

    def __init__(
        self,
        *,
        score_type: ScoreName = "gam",
        context_col: str = "context",
        postprocessing_mode: PostprocessingName = "skip",
        context_combination_kwargs: ContextCombinationKwargs | None = None,
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        public_cfg = CausalChangeConfigTabular(
            data_mode=DataMode.TAB_CONTEXTS,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType(score_type),
            context_mode=TabularContextMode.ORACLE,
            context_combination_method=TabularContextMethod.LINC,
            context_combination_kwargs=(
                ContextCombinationKwargs() if context_combination_kwargs is None else context_combination_kwargs
            ),
            context_col=context_col,
            postprocessing_mode=PostprocessingMode(postprocessing_mode),
            score_kwargs={} if score_kwargs is None else dict(score_kwargs),
            seed=seed,
        )
        self.public_config_ = public_cfg
        super().__init__(public_cfg, var_nms=var_nms, lg=lg, vb=vb)


class CMM(CausalChange):
    """CMM causal discovery: TOPIC search with mixture-regression local scoring."""

    public_config_: CausalChangeConfigTabular

    def __init__(
        self,
        *,
        mix_type: MixTypeName = "lin",
        k_max: int = 5,
        lambda_mix: float = 1.0,
        hybrid_mixing: bool = True,
        score_type: ScoreName = "lin",
        postprocessing_mode: PostprocessingName = "skip",
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        resolved_score_kwargs = {} if score_kwargs is None else dict(score_kwargs)
        resolved_score_kwargs.update(
            {
                "k_max": int(k_max),
                "lambda_mix": float(lambda_mix),
                "hybrid_mixing": bool(hybrid_mixing),
            }
        )

        public_cfg = CausalChangeConfigTabular(
            data_mode=DataMode.TABULAR,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType(score_type),
            context_mode=TabularContextMode.SKIP,
            mix_type=MixedSCMType(mix_type),
            postprocessing_mode=PostprocessingMode(postprocessing_mode),
            score_kwargs=resolved_score_kwargs,
            seed=seed,
        )
        self.public_config_ = public_cfg
        super().__init__(public_cfg, var_nms=var_nms, lg=lg, vb=vb)


class SpaceTime(CausalChange):
    """SpaceTime causal discovery for temporal or multi-context temporal data."""

    public_config_: CausalChangeConfigTemporal

    def __init__(
        self,
        *,
        score_type: ScoreName = "gam",
        data_mode: DataName = "time-contexts",
        tau_max: int = 2,
        context_col: str = "context",
        changepoint_mode: ChangepointModeName = "detect",
        changepoint_scope: ChangepointScopeName = "global",
        changepoint_method: ChangepointMethodName = "pelt",
        clustering_scope: ClusteringScopeName = "regimes-contexts",
        clustering_method: ClusteringMethodName = "statistical-testing",
        testing_method: TestingMethodName = "kernel",
        d_min: int = 30,
        max_iter: int = 3,
        pelt_penalty: float | PeltPenaltyName = "auto",
        mechanism_test_alpha: float = 0.05,
        fixed_changepoints: list[int] | None = None,
        postprocessing_mode: PostprocessingName = "skip",
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        public_cfg = CausalChangeConfigTemporal(
            data_mode=DataMode(data_mode),
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType(score_type),
            context_col=context_col,
            tau_max=tau_max,
            d_min=d_min,
            max_iter=max_iter,
            mechanism_test_alpha=mechanism_test_alpha,
            pelt_penalty=pelt_penalty,
            fixed_changepoints=[] if fixed_changepoints is None else list(fixed_changepoints),
            changepoint_mode=ChangepointMode(changepoint_mode),
            changepoint_scope=ChangepointScope(changepoint_scope),
            changepoint_method=ChangepointMethod(changepoint_method),
            clustering_scope=MechanismClusteringScope(clustering_scope),
            clustering_method=MechanismClusteringMethod(clustering_method),
            testing_method=StatisticalTestingMethod(testing_method),
            postprocessing_mode=PostprocessingMode(postprocessing_mode),
            score_kwargs={} if score_kwargs is None else dict(score_kwargs),
            seed=seed,
        )
        self.public_config_ = public_cfg
        super().__init__(public_cfg, var_nms=var_nms, lg=lg, vb=vb)


__all__ = [
    "Topic",
    "Linc",
    "CMM",
    "SpaceTime",
    "ScoreName",
    "PostprocessingName",
    "MixTypeName",
    "DataName",
    "ChangepointModeName",
    "ChangepointScopeName",
    "ChangepointMethodName",
    "ClusteringScopeName",
    "ClusteringMethodName",
    "TestingMethodName",
    "PeltPenaltyName",
]
