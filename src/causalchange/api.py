from __future__ import annotations

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
TabularMechanismClusteringName = Literal[
    "score-merge",
    "statistical-testing",
    "mechanism-clustering",
]


class Topic(CausalChange):
    r"""TOPIC causal discovery for one tabular dataset.
        :param optargs: optional arguments

    :Arguments:
    * *score_type* (``str``) -- local regression/scoring model
    * *postprocessing_mode* (``str``) -- optional postprocessing
    * *score_kwargs* (``dict``) -- additional keyword arguments passed to the local scoring model
    * *seed* (``int``) -- random seed used by stochastic scoring components
    * *var_nms* (``list[str]``) -- optional variable names for display/debugging/plotting

    Attributes after fit

    * *graph_* -- discovered causal graph as ``networkx.DiGraph``
    * *topological_order_* -- discovered topological order if produced by the graph search
    * *edge_strengths_* -- optional edge-strength postprocessing result when
      ``postprocessing_mode="edge-strengths"``
    * *history_* -- graph-search history
    """

    public_config_: CausalChangeConfigTabular

    def __init__(
        self,
        *,
        score_type: ScoreName = "gam",
        postprocessing_mode: PostprocessingName = "skip",
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
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
        super().__init__(public_cfg, var_nms=var_nms)


class Linc(CausalChange):
    r"""LINC causal discovery for observed multi-context tabular data.
        :param optargs: optional arguments

    * *score_type* (``str``) -- local regression/scoring model
    * *context_col* (``str``) -- name of the observed context indicator column
    * *postprocessing_mode* (``str``) -- optional postprocessing
    * *context_combination_kwargs* (``ContextCombinationKwargs | None``) --
      optional parameters controlling how context-specific scores are combined
    * *score_kwargs* (``dict``) -- additional keyword arguments passed to the local scoring model
    * *seed* (``int``) -- random seed used by stochastic scoring components
    * *var_nms* (``list[str]``) -- optional variable names for display/debugging/plotting

    Attributes after fit

    * *graph_* --  discovered causal graph as ``networkx.DiGraph``
    * *topological_order_* -- discovered topological order if produced by the graph search
    * *edge_strengths_* -- optional edge-strength postprocessing result when
      ``postprocessing_mode="edge-strengths"``
    * *linc_components_* -- final-graph context partitions per target
    * *linc_labels_* -- hard context-cluster labels per target
    * *linc_groups_* -- context groups per target
    * *last_context_combo_* -- debug-only most recent local context-combination result
    * *history_* -- graph-search history
    * *result_* -- full result object
    """

    public_config_: CausalChangeConfigTabular

    def __init__(
        self,
        *,
        score_type: ScoreName = "ff",
        context_col: str = "context",
        postprocessing_mode: PostprocessingName = "skip",
        mechanism_clustering_method: TabularMechanismClusteringName = "score-merge",
        context_combination_kwargs: ClusteringMethod | None = None,
        testing_method: TestingMethodName | None = None,
        mechanism_test_alpha: float = 0.05,
        mechanism_clustering_n_clusters: int | None = None,
        mechanism_clustering_distance_threshold: float | None = None,
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
    ):
        resolved_method = TabularMechanismClusteringMethod(mechanism_clustering_method)

        if testing_method is None:
            resolved_testing_method = (
                StatisticalTestingMethod.KERNEL
                if resolved_method == TabularMechanismClusteringMethod.TESTING
                else StatisticalTestingMethod.SKIP
            )
        else:
            resolved_testing_method = StatisticalTestingMethod(testing_method)

        public_cfg = CausalChangeConfigTabular(
            data_mode=DataMode.TAB_CONTEXTS,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType(score_type),
            context_mode=TabularContextMode.ORACLE,
            context_combination_method=TabularContextMethod.LINC,
            context_combination_kwargs=(
                context_combination_kwargs if context_combination_kwargs is not None else ClusteringMethod.AGGLOMERATIVE
            ),
            context_col=context_col,
            mechanism_clustering_method=resolved_method,
            testing_method=resolved_testing_method,
            mechanism_test_alpha=mechanism_test_alpha,
            mechanism_clustering_n_clusters=mechanism_clustering_n_clusters,
            mechanism_clustering_distance_threshold=mechanism_clustering_distance_threshold,
            postprocessing_mode=PostprocessingMode(postprocessing_mode),
            score_kwargs={} if score_kwargs is None else dict(score_kwargs),
            seed=seed,
        )

        self.public_config_ = public_cfg
        super().__init__(public_cfg, var_nms=var_nms)


class CMM(CausalChange):
    r"""CMM causal discovery for mixed-population tabular data.
        :param optargs: optional arguments

    :Arguments:
    * *mix_type* (``str``) -- mixture-regression family
    * *k_max* (``int``) -- maximum number of mixture components considered and compared using model selection
    * *postprocessing_mode* (``str``) -- optional postprocessing
    * *score_kwargs* (``dict``) -- additional CMM scoring arguments, for example
      ``degree`` for spline mixture terms.
    * *seed* (``int``) -- random seed used by stochastic scoring components
    * *var_nms* (``list[str]``) -- optional variable names for display/debugging

    Attributes after fit

    * *graph_* --  discovered causal graph as ``networkx.DiGraph``
    * *topological_order_* -- discovered topological order if produced by the graph search
    * *edge_strengths_* -- optional edge-strength postprocessing result when
      ``postprocessing_mode="edge-strengths"``
    * *cmm_components_* -- final-graph mixture assignments and responsibilities per target
    * *cmm_labels_* -- hard component labels per target
    * *history_* -- graph-search history
    * *result_* -- full result object
    """

    public_config_: CausalChangeConfigTabular

    def __init__(
        self,
        *,
        mix_type: MixTypeName = "lin",
        k_max: int = 5,
        postprocessing_mode: PostprocessingName = "skip",
        score_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        var_nms: list[str] | None = None,
    ):
        resolved_score_kwargs = {} if score_kwargs is None else dict(score_kwargs)
        resolved_score_kwargs.update({"k_max": int(k_max)})

        score_type: ScoreName = "lin"  # bic like scores here
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
        super().__init__(public_cfg, var_nms=var_nms)


class SpaceTime(CausalChange):
    r"""SpaceTime causal discovery for temporal or multi-context temporal data.
        :param optargs: optional arguments

    :Arguments:
    * *score_type* (``str``) -- local scoring model
    * *data_mode* (``str``) --  input data, ``"time"`` for one time series
      series and ``"time-contexts"`` for multiple time series with a context column
    * *context_col* (``str``) -- name of the context indicator column when
      ``data_mode="time-contexts"``
    * *tau_max* (``int``) -- maximum time lag considered for lagged causal parents
    * *changepoint_mode* (``str``) -- changepoint detection on or off
    * *d_min* (``int``) -- minimum segment/window length used for changepoint detection
    * *changepoint_scope* (``str``) -- changepoint detection when ``"time-contexts"``, either over all contexts jointly
      (``"global"``) or per context separately (``"per-context"``)
    * *changepoint_method* (``str``) -- changepoint detection method
    * *clustering_scope* (``str``) -- mechanism clustering scope (over regimes, contexts, or both)
    * *clustering_method* (``str``) -- mechanism clustering method, either statistical testing as
      (``statistical-testing``, slow) or clustering heuristic (``mechanism-clustering``)
    * *testing_method* (``str``) -- mechanism testing method if ``clustering_method==statistical-testing``
    * *max_iter* (``int``) -- maximum number of SpaceTime search iterations
    * *pelt_penalty* (``float | str``) -- PELT changepoint penalty if pelt used, either a float
      or one of ``"auto"``, ``"bic"``, or ``"mbic"``.
    * *mechanism_test_alpha* (``float``) -- significance level for mechanism equality tests
    * *fixed_changepoints* (``list[int] | None``) -- fixed changepoints used when
      ``changepoint_mode="fixed"``
    * *postprocessing_mode* (``str``) -- optional postprocessing
    * *score_kwargs* (``dict``) -- additional keyword arguments passed to the local scoring model
    * *seed* (``int``) -- random seed used by stochastic scoring components
    * *var_nms* (``list[str]``) -- optional variable names for display/debugging

    Attributes after fit

    * *graph_* -- discovered temporal summary graph as ``networkx.DiGraph``
    * *edge_strengths_* -- optional edge-strength postprocessing result when
      ``postprocessing_mode="edge-strengths"``
    * *changepoints_* -- detected or fixed changepoints
    * *changepoints_by_context_* -- per-context changepoints when
      ``changepoint_scope==per-context``
    * *partitions_* -- mechanism partitioning/clustering result
    * *changepoint_diagnostics_* -- changepoint detection diagnostics
    * *history_* -- temporal graph-search history
    * *result_* -- full temporal result object
    """

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
        super().__init__(public_cfg, var_nms=var_nms)


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
