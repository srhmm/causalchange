from __future__ import annotations

from typing import Optional, Any, Callable

import networkx as nx
import warnings

import pandas as pd

from causalchange.discovery._estimators import TOPIC, LINC, SpaceTime, SpaceTime_C, LINC_GLOBE, GLOBE, SpaceTime_GLOBE, \
    SpaceTime_GLOBE_C, CHAIN
from causalchange.scoring.edge_score import EdgeScore
from causalchange._cc_types import ScoreType, GPType, DataMode, GraphSearch, MixingType


class CausalChange:
    X: pd.DataFrame
    D: int
    N: int
    data_mode: DataMode
    graph_search: GraphSearch
    score_type: ScoreType | GPType
    mixing_type: MixingType
    tau_max: int
    score_params: dict[str, Any]

    # debug info
    lg: Optional[Any]
    vb: int
    truths: dict[str, Any]
    true_graph: nx.DiGraph | None
    true_top_order: list[int] | None
    node_nms: list[str] | None
    is_true_edge: Callable[[int], Callable[[int], str]]

    # state
    #-graph state
    graph_state: nx.DiGraph
    edges_state: EdgeScore
    _estimator:  TOPIC | GLOBE | None
    # flags
    fitted_graph: bool
    context_col: str

    def __init__(self, **kwargs):
        r""" CausalChange: Causal Discovery Algorithms under Distribution Change (continuous data, multi-context continuous data, multi-context data with latent confounding, continuous-valued time series, or mixtures of causal mechanisms).
        :param optargs: optional arguments

        :Keyword Arguments:
        * *data_mode* (``DataMode``) -- input data type, one iid dataset, multi-context data, mixed data, or TS data
        * *graph_search* (``GraphSearch``) -- search algo for DAGs
        * *score_type* (``MixingType``) -- regressor
        * *mixing_type* (``MixingType``) -- for mixed data, type of mixture model inference (EM algo), ow skip
        * *context_col* (``str``) -- for multi-context data, the column name of an indicator column for the contexts
        * *truths* (``nx.DiGraph``) -- for mixed data, oracle versions, w entries 't_A', 't_Z', 't_n_Z'
        * *lg* (``logging``) -- logger if verbosity>0
        * *vb* (``int``) -- verbosity level
        """
        self.defaultargs = {
            "data_mode": DataMode.IID,
            "graph_search": GraphSearch.TOPIC,
            "score_type": ScoreType.GAM,
            "mixing_type": MixingType.SKIP,
            "context_col": "context",
            "score_params": dict(),
            "tau_max": 2,
            "truths": dict(),
            "lg": None,
            "node_nms": None,
            "vb": 0,
        }
        self.__dict__.update((k, v) for k, v in self.defaultargs.items() if k not in kwargs.keys())
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.defaultargs.keys())

        assert self.mixing_type != MixingType.SKIP if self.data_mode == DataMode.MIXED else self.mixing_type == MixingType.SKIP, "provide MixingType as input arg"
        assert self.graph_search.is_compatible_with(self.data_mode), (
            f"Graph search {self.graph_search} is not compatible with data_mode {self.data_mode}"
        )

        def _info(st, strength=0):
            (self.lg.info(st) if self.lg is not None else print(st)) if self.vb + strength > 0 else None
        self._info = _info
        self.is_true_edge = (lambda i: lambda j: "") if 'true_g' not in self.truths else \
            (lambda node: lambda other: 'causal' if self.truths['true_g'].has_edge(node, other) else (
                'rev' if self.truths['true_g'].has_edge(other, node) else 'spurious'))
        self.true_top_order = [] if ('true_order' not in self.truths  and 'true_g' not in self.truths) else list(self.truths['true_order']) if  'true_order'  in self.truths else list(
            nx.topological_sort(self.truths['true_g']))

        self.graph_state = nx.DiGraph()
        self.topological_order = []
        self.fitted_graph = False
        self.search_history: list[dict] = []

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

        estimator_args = dict(
            data_mode=self.data_mode,
            score_type=self.score_type,
            mixing_type=self.mixing_type,
            score_params=self.score_params,
            vb=self.vb,
            lg=self.lg,
        )
        if self.data_mode in [DataMode.CONTEXTS, DataMode.TIME_CONTEXTS]:
            estimator_args['context_col'] = self.context_col
        elif self.data_mode in [DataMode.TIME, DataMode.TIME_CONTEXTS]:
            estimator_args['tau_max'] = self.tau_max

        if self.data_mode == DataMode.IID:
            estimator = TOPIC(**estimator_args) if self.graph_search == GraphSearch.TOPIC else GLOBE(**estimator_args) \
                if self.graph_search == GraphSearch.GLOBE else None

        elif self.data_mode == DataMode.CONTEXTS:
            estimator = LINC(**estimator_args) if self.graph_search == GraphSearch.TOPIC \
                else LINC_GLOBE(**estimator_args) if self.graph_search == GraphSearch.GLOBE \
                else CHAIN(**estimator_args) if self.graph_search == GraphSearch.CHAIN else None
        elif self.data_mode == DataMode.TIME:
            estimator = SpaceTime(**estimator_args) if self.graph_search == GraphSearch.TOPIC \
                else SpaceTime_GLOBE(**estimator_args) if self.graph_search == GraphSearch.GLOBE else None

        elif self.data_mode == DataMode.TIME_CONTEXTS:
            estimator = SpaceTime_C(**estimator_args) if self.graph_search == GraphSearch.TOPIC\
                else SpaceTime_GLOBE_C(**estimator_args) if self.graph_search == GraphSearch.GLOBE else None

        elif self.data_mode == DataMode.CONFOUNDED:
            raise NotImplementedError

        elif self.data_mode == DataMode.MIXED:
            raise NotImplementedError

        else: raise ValueError(self.data_mode)
        if estimator is None: raise ValueError(self.graph_search)

        self.graph_state = estimator.fit(X)
        self._estimator = estimator
        self.fitted_graph = True
        return self.graph_state