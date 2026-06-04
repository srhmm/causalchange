from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfig
from causalchange.config.factory import ConfigFactory
from causalchange.core.results import (
    CausalChangeResult,
    ContextCombinationResult,
    TemporalResult,
)
from causalchange.core.types import (
    ChangepointMethod,
    ChangepointMode,
    ChangepointScope,
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
from causalchange.engines.factory import EngineFactory
from causalchange.engines.tabular import TabularDiscoveryEngine
from causalchange.engines.temporal import TemporalDiscoveryEngine


class CausalChange:
    def __init__(
        self,
        cfg: CausalChangeConfig | None = None,
        *,
        data_mode: DataMode = DataMode.SKIP,
        graph_search: GraphSearch = GraphSearch.SKIP,
        score_type: ScoreType | GPType = ScoreType.SKIP,
        mix_type: MixedSCMType = MixedSCMType.SKIP,
        context_mode: TabularContextMode = TabularContextMode.SKIP,
        context_method: TabularContextMethod = TabularContextMethod.SKIP,
        changepoint_mode: ChangepointMode = ChangepointMode.SKIP,
        changepoint_scope: ChangepointScope = ChangepointScope.SKIP,
        changepoint_method: ChangepointMethod = ChangepointMethod.SKIP,
        clustering_scope: MechanismClusteringScope = MechanismClusteringScope.SKIP,
        clustering_method: MechanismClusteringMethod = MechanismClusteringMethod.SKIP,
        testing_method: StatisticalTestingMethod = StatisticalTestingMethod.SKIP,
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
        var_nms: list[str] | None = None,
        context_col: str | None = None,
        tau_max: int | None = None,
        d_min: int = 30,
        max_iter: int = 3,
        pelt_penalty: float | str = "auto",
        mechanism_test_alpha: float = 0.05,
        fixed_changepoints: list[int] | None = None,
        lg: logging = None,
        vb: int = 0,
        score_kwargs: dict[str, Any] | None = None,
    ):
        r"""CausalChange: Causal Discovery Algorithms under Distribution Change (continuous data, multi-context
        continuous data, continuous-valued time series, or mixtures of causal mechanisms).
        :param optargs: optional arguments

        :Arguments:
        * *cfg* (``CausalChangeConfig``) -- config with all parameters; or pass them manually below
        * *data_mode* (``DataMode``) -- input data type, one tabular dataset (``TABULAR``), tabular data from
          multiple contexts (``TAB_CONTEXTS``); one time series (``TIME``)
          or time series from multiple contexts (``TIME_CONTEXTS``).
        * *graph_search* (``GraphSearch``) -- search algorithm for DAGs / temporal graphs
        * *score_type* (``ScoreType | GPType``) -- regressor and corresponding scoring criterion,
          e.g.,``ScoreType.LIN`` or ``GPType.FOURIER``
        * *mix_type* (``MixedSCMType``) -- regressor and scoring for mixtures of SCMs, e.g.,``MixedSCMType.LIN``
        * *context_mode* (``TabularContextMode``) -- no contexts, hidden or observed
        * *context_method* (``TabularContextMethod``) -- algo to combine contexts
        * *changepoint_mode* (``ChangepointMode``) -- for time series, algorithm to detect causal changepoints
        * *changepoint_scope* (``ChangepointMode``) -- for time series from multiple contexts, whether to
          detect changepoints globally or per context
        * *changepoint_method* (``ChangepointMethod``) -- for time series, changepoint detection algo
        * *clustering_scope* (``MechanismClusteringScope``) -- for time series, clustering of similar causal mechanisms
          over regimes/contexts/both/neither
        * *clustering_method* (``MechanismClusteringMethod``) -- for time series, clustering algo
        * *testing_method* (``StatisticalTestingMethod``) -- if needed, statistical testing method
        * *postprocessing_mode* (``PostprocessingMode``) -- if needed, postprocessing, such as computing strengths
          of each pair-wise edge (X1, X2) relative to the discovered causal graph; different from edge score of
          of each set-wise causal relationship (XPa={X1,..Xn), Xtgt) in the causal graph
        * *var_nms* (``list[str]``) -- optional column names for display/debug/plotting
        * *context_col* (``str``) -- for multi-context data, the column name of an
          indicator column for the contexts
        * *tau_max* (``int``) -- for time series, maximum time lag to consider
        * *d_min* (``int``) -- for time series, minimum time window length to consider
        * *max_iter* (``int``) -- for time series, maximum number of interleaved iterations
        * *pelt_penalty* (``int``) -- for time series, sensitivity threshold for changepoint detection in PELT,
          a float number or one of {"auto", "mbic", "bic"}.
        * *mechanism_test_alpha* (``int``) -- if testing causal mechanisms for equality,
          significance threshold for testing
        * *fixed_changepoints* (``int``) -- for time series, optional known changepoints
          (used when ``changepoints==ChangepointMode.ORACLE``)
        * *lg* (``logging``) -- logger if verbosity>0
        * *vb* (``int``) -- verbosity level
        * *score_kwargs* (``dict``) -- any arguments needed for scoring functions
        """

        self.lg = lg
        self.vb = int(vb)
        self.node_nms = var_nms

        self.cfg = ConfigFactory.make_causal_change_config(
            cfg=cfg,
            data_mode=data_mode,
            graph_search=graph_search,
            score_type=score_type,
            mix_type=mix_type,
            context_mode=context_mode,
            context_combination_method=context_method,
            context_col=context_col,
            changepoint_mode=changepoint_mode,
            changepoint_scope=changepoint_scope,
            changepoint_method=changepoint_method,
            clustering_scope=clustering_scope,
            clustering_method=clustering_method,
            testing_method=testing_method,
            postprocessing_mode=postprocessing_mode,
            tau_max=tau_max,
            d_min=d_min,
            max_iter=max_iter,
            pelt_penalty=pelt_penalty,
            mechanism_test_alpha=mechanism_test_alpha,
            fixed_changepoints=fixed_changepoints,
            score_kwargs=score_kwargs,
        )

        self.X_: pd.DataFrame | None = None
        self.engine_: TabularDiscoveryEngine | TemporalDiscoveryEngine | None = None
        self.result_: CausalChangeResult | None = None

        self.N: int | None = None
        self.D: int | None = None
        self.feature_cols_: list[str] | None = None
        self.fitted_graph = False

        self.result_ = None

    def fit(self, X: pd.DataFrame) -> CausalChange:
        X_checked = self.check_input(X)

        self.engine_ = EngineFactory.from_config(self.cfg)
        self.engine_.fit(X_checked)

        self.result_ = self.engine_.discover()

        self.fitted_graph = True
        return self

    def score(self, effect, parents=()) -> float:
        self._require_fitted()
        assert self.engine_ is not None
        return float(self.engine_.local_score(effect, parents))

    def get_result(self) -> CausalChangeResult:
        self._require_fitted()
        return self.result_

    def check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """some data checks"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.empty:
            raise ValueError("X must contain at least one row and one column.")

        if self.cfg.data_mode.is_context():
            assert self.cfg.context_col in X.columns
            assert not X[self.cfg.context_col].isna().any()

        feature_cols = (
            [co for co in X.columns if co != self.cfg.context_col]
            if self.cfg.data_mode.is_context()
            else (list(X.columns))
        )
        if not feature_cols:
            raise ValueError("No feature columns found after excluding context_col.")

        if self.node_nms is not None and len(self.node_nms) != len(feature_cols):
            raise ValueError(f"node_nms has length {len(self.node_nms)}, " f"but X has {len(feature_cols)} columns.")

        self.N = int(X.shape[0])
        self.D = int(len(feature_cols))
        self.feature_cols_ = list(feature_cols)
        if self.node_nms is None:
            self.node_nms = [str(c) for c in feature_cols]

        self.X_ = X
        return X

    @property
    def graph_(self):
        self._require_fitted()
        return self.result_.graph

    @property
    def edge_strengths_(self):
        self._require_fitted()
        return self.result_.edge_strengths

    @property
    def topological_order_(self) -> list | None:
        self._require_fitted()
        return self.result_.topological_order

    @property
    def history_(self) -> list[dict]:
        self._require_fitted()
        return self.result_.history

    @property
    def changepoints_(self) -> list[int] | None:
        self._require_fitted()

        if not self.cfg.data_mode.is_temporal():
            return None

        result = cast(TemporalResult, self.result_)
        return result.changepoints

    @property
    def changepoints_by_context_(self) -> dict | None:
        self._require_fitted()

        if not self.cfg.data_mode.is_temporal():
            return None

        result = cast(TemporalResult, self.result_)
        return result.changepoints_by_context

    @property
    def partitions_(self):
        self._require_fitted()

        if not self.cfg.data_mode.is_temporal():
            return None

        result = cast(TemporalResult, self.result_)
        return result.grid_clusters

    """   should be in PostProcessingResults
    def get_spacetime_mechanism_scores(
        self,
        *,
        graph=None,
        scope: str = "global",
        changepoints: list[int] | None = None,
    ) -> pd.DataFrame:
        self._require_fitted()

        if not isinstance(self.engine_, TemporalDiscoveryEngine):
            raise RuntimeError("spacetime_mechanism_scores() is only available for SpaceTime models.")

        return self.engine_.mechanism_scores(
            graph=graph,
            scope=scope,
            changepoints=changepoints,
        )

     def get_spacetime_edge_contributions(
        self,
        *,
        graph=None,
        scope: str = "global",
        changepoints: list[int] | None = None,
    ) -> pd.DataFrame:
        self._require_fitted()

        if not isinstance(self.engine_, TemporalDiscoveryEngine):
            raise RuntimeError("spacetime_edge_contributions() is only available for SpaceTime models.")

        return self.engine_.edge_contributions(
            graph=graph,
            scope=scope,
            changepoints=changepoints,
        )"""

    @property
    def last_context_combo_(self) -> ContextCombinationResult | None:
        return None if self.engine_ is None else getattr(self.engine_, "last_context_combo_", None)

    def _require_fitted(self) -> None:
        if self.engine_ is None or self.result_ is None:
            raise RuntimeError("Call fit(X) before accessing fitted results.")
