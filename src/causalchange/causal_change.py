from __future__ import annotations

import networkx as nx
import pandas as pd

from causalchange.config.cc_config import CausalChangeConfig, ChangepointMode, ChangepointScope, SpaceTimeConfig
from causalchange.config.cc_types import (
    ContextAggregation,
    DataMode,
    GPType,
    GraphSearch,
    ScoreType,
)
from causalchange.discovery.factory import PipelineFactory
from causalchange.discovery.pipeline import AggregationResult, TabularDiscoveryEngine
from causalchange.discovery.search.topic import DAGSearchResult
from causalchange.discovery.search_time.base import SpaceTimeResult
from causalchange.discovery.search_time.engine import SpaceTimeEngine


class CausalChange:
    def __init__(
        self,
        cfg: CausalChangeConfig | None = None,
        *,
        data_mode: DataMode = DataMode.SKIP,
        graph_search: GraphSearch = GraphSearch.SKIP,
        score_type: ScoreType | GPType = ScoreType.SKIP,
        aggregation: ContextAggregation = ContextAggregation.SKIP,
        changepoints: ChangepointMode = ChangepointMode.NONE,
        changepoint_scope: ChangepointScope = ChangepointScope.GLOBAL,
        node_nms: list[str] | None = None,
        context_col: str | None = None,
        tau_max: int | None = None,
        fixed_changepoints: list[int] | None = None,
        d_min: int = 30,
        max_iter: int = 3,
        pelt_penalty: float | str = "auto",
        detect_contexts: bool = False,
        detect_regimes: bool = False,
        mechanism_test_alpha: float = 0.05,
        lg=None,
        vb: int = 0,
        **kwargs,
    ):
        r"""CausalChange: Causal Discovery Algorithms under Distribution Change (continuous data, multi-context
        continuous data, continuous-valued time series, or mixtures of causal mechanisms).
        :param optargs: optional arguments

        :Arguments:
        * *cfg* (``CausalChangeConfig``) -- config
        * *data_mode* (``DataMode``) -- input data type, one tabular dataset (``IID``), tabular data from
        multiple contexts (``CONTEXTS``), one time series (``TIME``)
        or time series from multiple contexts (``TIME_CONTEXTS``).

        * *graph_search* (``GraphSearch``) -- search algorithm for DAGs
        * *score_type* (``ScoreType``) -- regression and scoring
        * *aggregation* (``ContextAggregation``) -- for multi-context data, algorithm to combine contexts
        * *changepoints* (``ChangepointMode``) -- for time series, algorithm to detect causal changepoints
        * *context_col* (``str``) -- for multi-context data, the column name of an
        indicator column for the contexts
        * *tau_max* (``int``) -- for time series, maximum time lag to consider
        * *fixed_changepoints* (``int``) -- for time series, optional known changepoints
        (used when ``changepoints==ChangepointMode.FIXED``)
        * *truths* (``nx.DiGraph``) -- for mixed data, oracle versions, w entries 't_A', 't_Z', 't_n_Z'
        * *lg* (``logging``) -- logger if verbosity>0
        * *vb* (``int``) -- verbosity level
        """

        self.lg = lg
        self.vb = int(vb)
        self.node_nms = node_nms

        self.cfg = self._make_config(
            cfg=cfg,
            data_mode=data_mode,
            graph_search=graph_search,
            score_type=score_type,
            aggregation=aggregation,
            context_col=context_col,
            tau_max=tau_max,
            changepoints=changepoints,
            changepoint_scope=changepoint_scope,
            fixed_changepoints=fixed_changepoints,
            d_min=d_min,
            kwargs=kwargs,
            max_iter=max_iter,
            pelt_penalty=pelt_penalty,
            detect_contexts=detect_contexts,
            detect_regimes=detect_regimes,
            mechanism_test_alpha=mechanism_test_alpha,
        )

        self.data_mode = self.cfg.data_mode
        self.graph_search = self.cfg.graph_search
        self.score_type = self.cfg.score_type
        self.aggregation = self.cfg.aggregation
        self.context_col = self.cfg.context_col

        self.X_: pd.DataFrame | None = None
        self.engine_: TabularDiscoveryEngine | SpaceTimeEngine | None = None
        self.result_: DAGSearchResult | SpaceTimeResult | None = None
        self.graph_: nx.DiGraph | None = None

        self.N: int | None = None
        self.D: int | None = None
        self.feature_cols_: list[str] | None = None
        self.fitted_graph = False

        self.result_ = None
        self.graph_ = None
        self.edge_strengths_: dict = {}
        self.order_: list | None = None

        self.changepoints_: list[int] | None = None
        self.changepoints_by_context_: dict | None = None
        self.partitions_ = None

    def fit(self, X: pd.DataFrame) -> CausalChange:
        X_checked = self._check_X(X)

        self.engine_ = PipelineFactory.from_config(self.cfg)
        self.engine_.fit(X_checked)

        self.result_ = self.engine_.discover()
        self.graph_ = self.result_.graph
        self.fitted_graph = True

        self.edge_strengths_ = self.result_.edge_strengths
        self.order_ = self.result_.topological_order

        if self.cfg.data_mode.is_temporal():
            self.changepoints_ = self.result_.changepoints
            self.changepoints_by_context_ = self.result_.changepoints_by_context
            self.partitions_ = self.result_.partitions
        else:
            self.changepoints_ = None
            self.changepoints_by_context_ = None
            self.partitions_ = None

        return self

    def score(self, effect, parents=()) -> float:
        self._require_fitted()
        assert self.engine_ is not None
        return float(self.engine_.score_edge(effect, parents))

    def get_result(self) -> DAGSearchResult | SpaceTimeResult:
        return self.result

    @property
    def graph(self) -> nx.DiGraph:
        self._require_fitted()
        assert self.graph_ is not None
        return self.graph_

    @property
    def result(self) -> DAGSearchResult | SpaceTimeResult:
        self._require_fitted()
        assert self.result_ is not None
        return self.result_

    @property
    def topological_order_(self) -> list:
        self._require_fitted()
        assert self.result_ is not None
        return self.result_.topological_order

    @property
    def history_(self) -> list[dict]:
        self._require_fitted()
        assert self.result_ is not None
        return self.result_.history

    @property
    def last_aggregation_(self) -> AggregationResult | None:
        return None if self.engine_ is None else getattr(self.engine_, "last_aggregation_", None)

    def _require_fitted(self) -> None:
        if self.engine_ is None or self.result_ is None or self.graph_ is None:
            raise RuntimeError("Call fit(X) before accessing fitted results.")

    def _check_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input data and record basic fitted metadata."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if X.empty:
            raise ValueError("X must contain at least one row and one column.")

        if self.data_mode.is_context():
            if self.context_col not in X.columns:
                raise ValueError(
                    f"data_mode={self.data_mode.value} requires context column "
                    f"{self.context_col!r}, but it was not found in X.columns."
                )
            if X[self.context_col].isna().any():
                raise ValueError(f"context_col {self.context_col!r} contains NaNs.")

            feature_cols = [c for c in X.columns if c != self.context_col]
        else:
            feature_cols = list(X.columns)

        if not feature_cols:
            raise ValueError("No feature columns found after excluding context_col.")

        if self.node_nms is not None and len(self.node_nms) != len(feature_cols):
            raise ValueError(
                f"node_nms has length {len(self.node_nms)}, " f"but X has {len(feature_cols)} feature columns."
            )

        self.N = int(X.shape[0])
        self.D = int(len(feature_cols))
        self.feature_cols_ = list(feature_cols)

        if self.node_nms is None:
            self.node_nms = [str(c) for c in feature_cols]

        self.X_ = X
        return X

    def _make_config(
        self,
        *,
        cfg: CausalChangeConfig | None,
        data_mode: DataMode,
        graph_search: GraphSearch,
        score_type: ScoreType,
        aggregation: ContextAggregation,
        context_col: str | None,
        tau_max: int | None,
        changepoints: ChangepointMode,
        changepoint_scope: ChangepointScope,
        fixed_changepoints: list[int] | None,
        d_min: int,
        max_iter: int = 3,
        pelt_penalty: float | str = "auto",
        detect_contexts: bool = True,
        detect_regimes: bool = True,
        mechanism_test_alpha: float = 0.5,
        kwargs: dict,
    ) -> CausalChangeConfig:
        if cfg is not None:
            if (
                data_mode != DataMode.SKIP
                or graph_search != GraphSearch.SKIP
                or score_type != ScoreType.SKIP
                or aggregation != ContextAggregation.SKIP
                or context_col is not None
                or tau_max is not None
                or changepoints != ChangepointMode.NONE
                or fixed_changepoints is not None
                # or d_min != 30
                # or max_iter != 3
                # or pelt_penalty != 3.0
                # or detect_contexts is not False
                # or detect_regimes is not False
                # or mechanism_test_alpha != 0.5
                or kwargs
            ):
                raise ValueError("Pass either cfg=... or individual constructor options, not both.")
            return cfg
        if data_mode == DataMode.SKIP:
            raise ValueError("data_mode is required when cfg is not provided.")
        if graph_search == GraphSearch.SKIP:
            raise ValueError("graph_search is required when cfg is not provided.")
        if score_type == ScoreType.SKIP:
            raise ValueError("score_type is required when cfg is not provided.")
        if score_type == ScoreType.GP:
            raise ValueError("score_type must be concrete. Use GPType.EXACT or GPType.FOURIER.")

        if data_mode.is_temporal():
            if tau_max is None:
                raise ValueError("tau_max is required for temporal data modes.")

            if "spacetime" in kwargs:
                raise ValueError(
                    "CausalChange constructs SpaceTimeConfig internally; pass tau_max/changepoints instead."
                )

            kwargs["spacetime"] = SpaceTimeConfig(
                tau_max=tau_max,
                changepoints=changepoints,
                changepoint_scope=changepoint_scope,
                fixed_changepoints=fixed_changepoints or [],
                d_min=d_min,
                max_iter=max_iter,
                pelt_penalty=pelt_penalty,
                detect_contexts=detect_contexts,
                detect_regimes=detect_regimes,
                mechanism_test_alpha=mechanism_test_alpha,
            )

        if data_mode.is_context() and context_col is None:
            raise ValueError("context_col is required for context data modes.")

        return CausalChangeConfig(
            data_mode=data_mode,
            graph_search=graph_search,
            score_type=score_type,
            aggregation=aggregation,
            context_col=context_col or "context",
            **kwargs,
        )
