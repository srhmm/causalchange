from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import pandas as pd

from causalchange.config.causal_change_config import (
    CausalChangeConfig,
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
from causalchange.core.results import (
    CausalChangeResult,
    ContextCombinationResult,
    TemporalResult,
)
from causalchange.core.types import DataMode
from causalchange.engines.factory import EngineFactory
from causalchange.engines.tabular import TabularDiscoveryEngine
from causalchange.engines.temporal import TemporalDiscoveryEngine


class CausalChange:
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

    def __init__(
        self,
        cfg: CausalChangeConfig,
        *,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        if not isinstance(cfg, (CausalChangeConfigTabular | CausalChangeConfigTemporal)):
            raise TypeError(
                "cfg must be a CausalChangeConfigTabular or CausalChangeConfigTemporal instance. "
                "Use CausalChange.tabular(...), CausalChange.temporal(...), or pass a validated config."
            )

        self.cfg = cfg
        self.lg = lg
        self.vb = int(vb)
        self.node_nms = var_nms

        self.X_: pd.DataFrame | None = None
        self.engine_: TabularDiscoveryEngine | TemporalDiscoveryEngine | None = None
        self.result_: CausalChangeResult | None = None

        self.N: int | None = None
        self.D: int | None = None
        self.feature_cols_: list[str] | None = None
        self.fitted_graph = False

    @classmethod
    def tabular(
        cls,
        *,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
        **config_kwargs: Any,
    ) -> CausalChange:
        """Build a tabular estimator from CausalChangeConfigTabular fields."""
        cfg = CausalChangeConfigTabular.model_validate(config_kwargs)
        return cls(cfg, var_nms=var_nms, lg=lg, vb=vb)

    @classmethod
    def temporal(
        cls,
        *,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
        **config_kwargs: Any,
    ) -> CausalChange:
        """Build a temporal estimator from CausalChangeConfigTemporal fields."""
        cfg = CausalChangeConfigTemporal.model_validate(config_kwargs)
        return cls(cfg, var_nms=var_nms, lg=lg, vb=vb)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ) -> CausalChange:
        """Build an estimator from a dict-like object using data_mode to choose the config class."""
        if "data_mode" not in data:
            raise ValueError("data_mode is required to choose tabular vs temporal config.")

        data_mode = data["data_mode"]
        mode = data_mode if isinstance(data_mode, DataMode) else DataMode(data_mode)
        cfg_cls = CausalChangeConfigTemporal if mode.is_temporal() else CausalChangeConfigTabular
        cfg = cfg_cls.model_validate(dict(data))
        return cls(cfg, var_nms=var_nms, lg=lg, vb=vb)

    def fit(self, X: pd.DataFrame) -> CausalChange:
        X_checked = self.check_input(X)

        self.engine_ = EngineFactory.from_config(self.cfg)
        self.engine_.fit(X_checked)

        self.result_ = self.engine_.discover()
        self.fitted_graph = True
        return self

    def score(self, effect: Any, parents: tuple[Any, ...] = ()) -> float:
        self._require_fitted()
        if self.engine_ is None:
            raise RuntimeError("Call fit(X) before scoring.")
        return float(self.engine_.local_score(effect, parents))

    def get_result(self) -> CausalChangeResult:
        self._require_fitted()
        if self.result_ is None:
            raise RuntimeError("Call fit(X) before accessing fitted results.")
        return self.result_

    def check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize runtime data passed to fit()."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if X.empty:
            raise ValueError("X must contain at least one row and one column.")

        if self.cfg.data_mode.is_context():
            context_col = self.cfg.context_col
            if context_col not in X.columns:
                raise ValueError(f"context_col={context_col!r} was not found in X.")
            if X[context_col].isna().any():
                raise ValueError(f"context_col={context_col!r} contains missing values.")

        feature_cols = (
            [column for column in X.columns if column != self.cfg.context_col]
            if self.cfg.data_mode.is_context()
            else list(X.columns)
        )

        if not feature_cols:
            raise ValueError("No feature columns found after excluding context_col.")

        if self.node_nms is not None and len(self.node_nms) != len(feature_cols):
            raise ValueError(
                f"node_nms has length {len(self.node_nms)}, but X has {len(feature_cols)} feature columns."
            )

        self.N = int(X.shape[0])
        self.D = int(len(feature_cols))
        self.feature_cols_ = list(feature_cols)
        if self.node_nms is None:
            self.node_nms = [str(column) for column in feature_cols]

        self.X_ = X
        return X

    @property
    def graph_(self):
        return self.get_result().graph

    @property
    def edge_strengths_(self):
        return self.get_result().edge_strengths

    @property
    def topological_order_(self) -> list[Any] | None:
        return self.get_result().topological_order

    @property
    def history_(self) -> list[dict[str, Any]]:
        return self.get_result().history

    @property
    def changepoints_(self) -> list[int] | None:
        result = self.get_result()
        if not self.cfg.data_mode.is_temporal():
            return None
        return cast(TemporalResult, result).changepoints

    @property
    def changepoints_by_context_(self) -> dict[Any, list[int]] | None:
        result = self.get_result()
        if not self.cfg.data_mode.is_temporal():
            return None
        return cast(TemporalResult, result).changepoints_by_context

    @property
    def partitions_(self):
        result = self.get_result()
        if not self.cfg.data_mode.is_temporal():
            return None
        return cast(TemporalResult, result).grid_clusters

    @property
    def last_context_combo_(self) -> ContextCombinationResult | None:
        return None if self.engine_ is None else getattr(self.engine_, "last_context_combo_", None)

    def _require_fitted(self) -> None:
        if self.engine_ is None or self.result_ is None:
            raise RuntimeError("Call fit(X) before accessing fitted results.")
