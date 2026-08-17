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
    CMMMixtureResult,
    ContextCombinationResult,
    LincMixtureResult,
    TabularResult,
    TemporalResult,
)
from causalchange.core.types import DataMode
from causalchange.engines.factory import EngineFactory
from causalchange.engines.tabular import TabularDiscoveryEngine
from causalchange.engines.temporal import TemporalDiscoveryEngine


class CausalChange:
    r"""CausalChange: Causal Discovery Algorithms under Distribution Change (continuous data, multi-context
    continuous data, continuous-valued time series, or mixtures of causal mechanisms).
    """

    def __init__(
        self,
        cfg: CausalChangeConfig,
        *,
        var_nms: list[str] | None = None,
        lg: logging.Logger | None = None,
        vb: int = 0,
    ):
        assert isinstance(cfg, (CausalChangeConfigTabular | CausalChangeConfigTemporal))

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
        """CausalChangeConfigTabular to tabular estimator"""
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
        """CausalChangeConfigTemporal to temporal estimator"""
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
        """estimator from a dict-like object using data_mode to choose the config class."""
        if "data_mode" not in data:
            raise ValueError("data_mode is required to choose tabular vs temporal config.")

        data_mode = data["data_mode"]
        mode = data_mode if isinstance(data_mode, DataMode) else DataMode(data_mode)
        cfg_cls = CausalChangeConfigTemporal if mode.is_temporal() else CausalChangeConfigTabular
        cfg = cfg_cls.model_validate(dict(data))
        return cls(cfg, var_nms=var_nms, lg=lg, vb=vb)

    def fit(self, X: pd.DataFrame) -> CausalChange:
        """main fit function"""
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
    def spacetime_components_(self):
        result = self.get_result()
        if not self.cfg.data_mode.is_temporal():
            return None
        return cast(TemporalResult, result).grid_clusters

    @property
    def cmm_components_(self) -> CMMMixtureResult | None:
        result = self.get_result()

        if self.cfg.data_mode.is_temporal():
            return None

        return cast(TabularResult, result).mixture_components

    @property
    def cmm_labels_(self) -> dict[Any, list[int]] | None:
        components = self.cmm_components_

        if components is None:
            return None

        return {target: target_result.labels for target, target_result in components.target_components.items()}

    @property
    def linc_components_(self) -> LincMixtureResult | None:
        result = self.get_result()

        if self.cfg.data_mode.is_temporal():
            return None

        return cast(TabularResult, result).linc_components

    @property
    def linc_labels_(self) -> dict[Any, dict[Any, int]] | None:
        components = self.linc_components_

        if components is None:
            return None

        return {
            target: target_result.labels_by_context for target, target_result in components.target_components.items()
        }

    @property
    def linc_groups_(self) -> dict[Any, list[frozenset[Any]]] | None:
        components = self.linc_components_

        if components is None:
            return None

        return {target: target_result.groups for target, target_result in components.target_components.items()}

    @property
    def last_context_combo_(self) -> ContextCombinationResult | None:
        self._require_fitted()
        return None if self.engine_ is None else getattr(self.engine_, "last_context_combo_", None)

    def _require_fitted(self) -> None:
        if self.engine_ is None or self.result_ is None:
            raise RuntimeError("Call fit(X) before accessing fitted results.")
