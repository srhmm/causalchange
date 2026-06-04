from __future__ import annotations

from typing import Any

from causalchange.causal_change import CausalChange
from causalchange.core.types import (
    ChangepointMethod,
    ChangepointMode,
    ChangepointScope,
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


class Topic(CausalChange):
    """TOPIC causal discovery for one tabular dataset."""

    def __init__(
        self,
        *,
        score_type: ScoreType | GPType = ScoreType.GAM,
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
        score_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            data_mode=DataMode.TABULAR,
            graph_search=GraphSearch.TOPIC,
            score_type=score_type,
            postprocessing_mode=postprocessing_mode,
            score_kwargs=score_kwargs,
            **kwargs,
        )


class Linc(CausalChange):
    """LINC causal discovery for multi-context tabular data."""

    def __init__(
        self,
        *,
        score_type: ScoreType | GPType = ScoreType.GAM,
        context_col: str = "context",
        context_method: TabularContextMethod = TabularContextMethod.LINC,
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
        score_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            data_mode=DataMode.TAB_CONTEXTS,
            graph_search=GraphSearch.TOPIC,
            score_type=score_type,
            context_mode=TabularContextMode.ORACLE,
            context_method=context_method,
            context_col=context_col,
            postprocessing_mode=postprocessing_mode,
            score_kwargs=score_kwargs,
            **kwargs,
        )


class SpaceTime(CausalChange):
    """SpaceTime causal discovery for temporal or multi-context temporal data."""

    def __init__(
        self,
        *,
        score_type: ScoreType | GPType = ScoreType.GAM,
        data_mode: DataMode = DataMode.TIME_CONTEXTS,
        tau_max: int = 2,
        context_col: str = "context",
        changepoint_mode: ChangepointMode = ChangepointMode.DETECT,
        changepoint_scope: ChangepointScope = ChangepointScope.GLOBAL,
        changepoint_method: ChangepointMethod = ChangepointMethod.PELT,
        clustering_scope: MechanismClusteringScope = MechanismClusteringScope.REGIMES_CONTEXTS,
        clustering_method: MechanismClusteringMethod = MechanismClusteringMethod.TESTING,
        testing_method: StatisticalTestingMethod = StatisticalTestingMethod.KERNEL,
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
        score_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if not data_mode.is_temporal():
            raise ValueError("SpaceTime requires DataMode.TIME or DataMode.TIME_CONTEXTS.")

        super().__init__(
            data_mode=data_mode,
            graph_search=GraphSearch.GLOBE,
            score_type=score_type,
            tau_max=tau_max,
            context_col=context_col,
            changepoint_mode=changepoint_mode,
            changepoint_scope=changepoint_scope,
            changepoint_method=changepoint_method,
            clustering_scope=clustering_scope,
            clustering_method=clustering_method,
            testing_method=testing_method,
            postprocessing_mode=postprocessing_mode,
            score_kwargs=score_kwargs,
            **kwargs,
        )
