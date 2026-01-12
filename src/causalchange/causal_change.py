from __future__ import annotations

from typing import Optional, Any, Callable

import networkx as nx
import warnings

import pandas as pd

from causalchange.config.cc_config import CausalChangeConfig
from causalchange.discovery.factory import PipelineFactory
from causalchange.discovery.pipeline import DiscoveryEngine, AggregationResult
from causalchange.discovery.scoring.edge_score import EdgeScore
from causalchange.config._cc_types import ScoreType, GPType, DataMode, GraphSearch, MixingType, ContextAggregation
from causalchange.discovery.search.topic import DAGSearchResult


class CausalChange:
    X: pd.DataFrame
    D: int
    N: int
    data_mode: DataMode
    graph_search: GraphSearch
    score_type: ScoreType | GPType | MixingType
    aggregation: ContextAggregation
    tau_max: int
    score_params: dict[str, Any]
    context_col: str

    # debug info
    lg: Optional[Any]
    vb: int
    truths: dict[str, Any]
    true_graph: nx.DiGraph | None
    node_nms: list[str] | None
    is_true_edge: Callable[[int], Callable[[int], str]]

    # state
    result_: DAGSearchResult
    graph_: nx.DiGraph
    edges_state: EdgeScore
    # flags
    fitted_graph: bool

    def __init__(
            self,
            cfg: CausalChangeConfig | None = None, *,
            data_mode: DataMode = DataMode.SKIP,
            graph_search: GraphSearch = GraphSearch.SKIP,
            score_type: ScoreType = ScoreType.SKIP,
            aggregation: ContextAggregation = ContextAggregation.LINC,
            truths: dict[str, Any] | None = None,
            node_nms: list[str] | None = None,
            context_col: str = "context",
            lg = None,
            vb = 0,
            **kwargs):
        r""" CausalChange: Causal Discovery Algorithms under Distribution Change (continuous data, multi-context continuous data, multi-context data with latent confounding, continuous-valued time series, or mixtures of causal mechanisms).
        :param optargs: optional arguments

        :Arguments:
        * *cfg* (``CausalChangeConfig``) -- config
        * *data_mode* (``DataMode``) -- input data type, one iid dataset, multi-context data, mixed data, or TS data
        * *graph_search* (``GraphSearch``) -- search algo for DAGs
        * *score_type* (``MixingType``) -- regressor
        * *mixing_type* (``MixingType``) -- for mixed data, type of mixture model inference (EM algo), ow skip
        * *context_col* (``str``) -- for multi-context data, the column name of an indicator column for the contexts
        * *truths* (``nx.DiGraph``) -- for mixed data, oracle versions, w entries 't_A', 't_Z', 't_n_Z'
        * *lg* (``logging``) -- logger if verbosity>0
        * *vb* (``int``) -- verbosity level
        """
        self.aggregation = aggregation
        self.node_nms = node_nms # Don't like that this is none
        self.truths = truths if truths is not None else {}
        self.lg = lg
        self.vb = vb
        if cfg is not None:
            if any([ty.value != 'skip' for ty in [data_mode, graph_search, score_type]]): # or kwargs:
                raise ValueError(
                    "Pass either cfg=... OR (data_mode, graph_search, score_type), not both.")
        else:
            if any([ty.value == 'skip' for ty in [data_mode, graph_search, score_type]]):
                raise ValueError("When cfg is None you must pass data_mode, graph_search, score_type")
            cfg = CausalChangeConfig(
                data_mode=data_mode,
                graph_search=graph_search,
                score_type=score_type,
                aggregation=self.aggregation,
                context_col=context_col,
                **kwargs,
            )

        self.cfg = cfg
        self.data_mode = cfg.data_mode
        self.graph_search = cfg.graph_search
        self.score_type = cfg.score_type
        self.aggregation = cfg.aggregation
        self.context_col = cfg.context_col
        self.tau_max = cfg.tau_max

        assert self.graph_search.is_compatible_with(self.data_mode), (
            f"Graph search {self.graph_search} is not compatible with data_mode {self.data_mode}"
        )

        def _info(st, strength=0):
            (self.lg.info(st) if self.lg is not None else print(st)) if self.vb + strength > 0 else None
        self._info = _info
        self.is_true_edge = (lambda i: lambda j: "") if 'true_g' not in self.truths else \
            (lambda node: lambda other: 'causal' if self.truths['true_g'].has_edge(node, other) else (
                'rev' if self.truths['true_g'].has_edge(other, node) else 'spurious'))
        self.graph_state_ = nx.DiGraph()
        self.fitted_graph = False
        self.search_history: list[dict] = []
        self.engine: Optional[DiscoveryEngine] = None


    def _check_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Check input data is compat with DataMode
       :param X: ``pd.DataFrame``: input data
        """
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)

        if self.data_mode in (DataMode.CONTEXTS, DataMode.TIME_CONTEXTS):
            if self.context_col not in X.columns:
                raise ValueError(
                    f"data_mode={self.data_mode.value} requires a context column "
                    f"'{self.context_col}', but it was not found in X.columns."
                )
            if X[self.context_col].isna().any():
                raise ValueError(f"context_col '{self.context_col}' contains NaNs.")

            feature_cols = [c for c in X.columns if c != self.context_col]
        else:
            feature_cols = list(X.columns)

        if len(feature_cols) == 0:
            raise ValueError("No feature columns found (after excluding context_col).")

        self.N = int(X.shape[0])
        self.D = int(len(feature_cols))

        if self.N <= 0 or self.D <= 0:
            raise ValueError(f"Invalid data shape after checks: N={self.N}, D={self.D}")

        if self.N < self.D:
            warnings.warn("n_samples < n_nodes", RuntimeWarning)

        if self.node_nms is not None:
            if len(self.node_nms) != self.D:
                raise ValueError("wrong number of node names")
        else:
            self.node_nms = [str(c) for c in feature_cols]

        self.X = X
        return X

    #%% Graph search
    def fit(self, X: pd.DataFrame) -> nx.DiGraph:
        """ Discover a causal DAG
       :param X: ``pd.DataFrame``: input data
       :return: ``nx.DiGraph``: causal DAG over nodes in X
        """
        X = self._check_X(X)
        self.engine = PipelineFactory.from_config(self.cfg)
        self.engine.fit(X)
        self.result_ : DAGSearchResult = self.engine.discover()
        self.graph_ = self.result_.graph

        self.fitted_graph = True
        return self.graph_


    @property
    def last_aggregation_(self) -> AggregationResult | None:
        if self.engine is None:
            return None
        return self.engine.last_aggregation_

    def score(self, effect, parents) -> float:
        if self.engine is None:
            raise RuntimeError("Call fit(X) before score().")
        return float(self.engine.score_edge(effect, parents))