from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigTemporal
from causalchange.core.results import SpaceTimeGridClusters
from causalchange.core.types import DataMode, GPType
from causalchange.domain.temporal import TemporalNode, TimeGrid, util_changepoints_to_intervals
from causalchange.scoring.regression import (
    fit_score_functional_model,
    fit_score_gp,
    fit_score_rff,
)
from causalchange.scoring.tabular import SCMScoreTabular


@dataclass(frozen=True)
class TimeLocalScoreResult:
    score: float
    model: Any
    residuals: np.ndarray | None
    y_true: np.ndarray | None
    design: pd.DataFrame


class SCMScoreTemporal:
    """
    Temporal scoring via a lagged design matrix.

    Nodes are represented as (variable, lag), where lag=0 is the current
    time point and lag>0 is a past value.

    For non-GP score types, this delegates to SCMScoreTabular on the lagged
    design matrix. For ScoreType.GP, it dispatches each temporal local
    mechanism directly to fit_score_gp.
    """

    def __init__(self, *, cfg: CausalChangeConfigTemporal):
        if cfg.data_mode not in (DataMode.TIME, DataMode.TIME_CONTEXTS):
            raise ValueError(f"EdgeScoreTime expects temporal data, got {cfg.data_mode=}")

        if cfg.tau_max <= 0:
            raise ValueError("SpaceTimeConfig.tau_max must be positive.")

        self.data_mode = cfg.data_mode
        self.score_type = cfg.score_type
        self.score_kwargs = dict(cfg.score_kwargs or {})
        self.tau_max = int(cfg.tau_max)

        self._time_windows: list[tuple[int, int]] | None = None
        self._global_n_samples: int | None = None

        # Used for fast/debug non-GP scores.
        self._tab = SCMScoreTabular(cfg)

        self._node_to_col: dict[TemporalNode, str] = {}
        self._Z: pd.DataFrame | None = None
        self._bound_key: tuple[int, tuple[str, ...], tuple[int, int]] | None = None

    @property
    def higher_is_better(self) -> bool:
        # Current score convention: all scores are description lengths/losses.
        # Lower score is better; compression gain is old_score - new_score.
        return False

    def _is_gp_score(self) -> bool:
        return self.score_type in (GPType.EXACT, GPType.FOURIER)

    def _ar_col(self, node: TemporalNode) -> str:
        variable, lag = node
        return f"{variable}_lag{lag}"

    def _df_key(self, X: pd.DataFrame) -> tuple[int, tuple[str, ...], tuple[int, int]]:
        return (id(X), tuple(map(str, X.columns)), tuple(X.shape))

    def fit(self, X: pd.DataFrame, *, bound_key=None) -> None:
        X0 = X.copy()
        X0.columns = [str(c) for c in X0.columns]

        Z = self.build_design(X0)

        self._node_to_col = {(v, lag): self._ar_col((v, lag)) for v in X0.columns for lag in range(0, self.tau_max + 1)}
        self._Z = Z
        self._bound_key = bound_key if bound_key is not None else self._df_key(X)
        self._global_n_samples = int(Z.shape[0])

        if not self._is_gp_score():
            self._tab.fit(Z)

    def _ensure_bound(self, X: pd.DataFrame) -> None:
        key = self._df_key(X)
        if self._Z is None or not self._node_to_col or self._bound_key != key:
            self.fit(X, bound_key=key)

    def build_design(self, X: pd.DataFrame) -> pd.DataFrame:
        tau = self.tau_max
        cols: dict[str, pd.Series] = {}

        for variable in X.columns:
            variable = str(variable)
            for lag in range(0, tau + 1):
                cols[self._ar_col((variable, lag))] = X[variable].shift(lag)

        Z = pd.DataFrame(cols)
        Z = Z.iloc[tau:].copy()
        Z.reset_index(drop=True, inplace=True)
        return Z

    def set_time_windows(
        self,
        *,
        n_raw_samples: int,
        changepoints: list[int],
    ) -> None:
        """
        Set scoring windows in original time coordinates.

        The lagged design matrix drops the first tau_max rows, so an original
        interval [a, b) maps to design rows [max(a, tau_max)-tau_max, b-tau_max).
        """

        raw_windows = util_changepoints_to_intervals(n_raw_samples, changepoints)

        design_windows: list[tuple[int, int]] = []
        for start, stop in raw_windows:
            design_start = max(start, self.tau_max) - self.tau_max
            design_stop = stop - self.tau_max

            if design_stop > design_start:
                design_windows.append((design_start, design_stop))

        if not design_windows:
            raise ValueError(
                "No valid scoring windows remain after applying tau_max. "
                f"n_raw_samples={n_raw_samples}, tau_max={self.tau_max}, changepoints={changepoints}"
            )

        self._time_windows = design_windows

    def score_edge(
        self,
        X: pd.DataFrame,
        effect: TemporalNode,
        parents: Sequence[TemporalNode],
        *,
        ret_full_result: bool = False,
        ret_residuals: bool = False,
    ) -> float | TimeLocalScoreResult:
        self._ensure_bound(X)
        assert self._Z is not None

        if self._time_windows is None:
            return self._score_edge_on_design(
                Z=self._Z,
                effect=effect,
                parents=parents,
                ret_full_result=ret_full_result,
                ret_residuals=ret_residuals,
            )

        if ret_full_result:
            # Graph search only needs scalar scores over windows.
            # For changepoints, residual_signal() temporarily disables windows.
            raise NotImplementedError("ret_full_result=True with multiple time windows is not implemented yet.")

        total = 0.0
        for start, stop in self._time_windows:
            Z_window = self._Z.iloc[start:stop].copy()
            total += float(
                self._score_edge_on_design(
                    Z=Z_window,
                    effect=effect,
                    parents=parents,
                    ret_full_result=False,
                    ret_residuals=False,
                )
            )

        return float(total)

    def _score_edge_on_design(
        self,
        *,
        Z: pd.DataFrame,
        effect: TemporalNode,
        parents: Sequence[TemporalNode],
        ret_full_result: bool,
        ret_residuals: bool,
    ) -> float | TimeLocalScoreResult:
        eff_col = self._node_to_col[effect]
        parent_cols = [self._node_to_col[p] for p in parents]

        if self._is_gp_score():
            return self._score_gp_on_design(
                Z=Z,
                eff_col=eff_col,
                parent_cols=parent_cols,
                ret_full_result=ret_full_result,
                ret_residuals=ret_residuals,
            )

        return self._score_tabular_on_design(
            Z=Z,
            eff_col=eff_col,
            parent_cols=parent_cols,
            ret_full_result=ret_full_result,
            ret_residuals=ret_residuals,
        )

    def _score_tabular_on_design(
        self,
        *,
        Z: pd.DataFrame,
        eff_col: str,
        parent_cols: list[str],
        ret_full_result: bool,
        ret_residuals: bool,
    ) -> float | TimeLocalScoreResult:
        if ret_full_result:
            self._tab._ensure_bound(Z)
            assert self._tab._edges is not None

            j = self._tab._col_index[eff_col]
            pa = [self._tab._col_index[p] for p in parent_cols]

            score, res = self._tab._edges.score_edge(
                j=j,
                pa=pa,
                ret_full_result=True,
                ret_residuals=ret_residuals,
            )

            residuals = res.get("residuals") if ret_residuals else None
            y_true = Z[eff_col].to_numpy(dtype=float) if ret_residuals else None

            return TimeLocalScoreResult(
                score=float(score),
                model=res.get("model"),
                residuals=residuals,
                y_true=y_true,
                design=Z[[eff_col] + parent_cols].copy(),
            )

        return float(self._tab.score_edge(Z, eff_col, parent_cols))

    def _score_gp_on_design(
        self,
        *,
        Z: pd.DataFrame,
        eff_col: str,
        parent_cols: list[str],
        ret_full_result: bool,
        ret_residuals: bool,
    ) -> float | TimeLocalScoreResult:
        cols = [eff_col] + parent_cols
        data = Z[cols].to_numpy(dtype=float)

        pa = tuple(range(1, len(cols)))

        score_fun = fit_score_rff if self.score_type == GPType.FOURIER else fit_score_gp

        score, res = fit_score_functional_model(
            data,
            pa=pa,
            target=0,
            score_fun=score_fun,
            ret_residuals=ret_residuals,
            **self.score_kwargs,
        )

        if not ret_full_result:
            return float(score)

        return TimeLocalScoreResult(
            score=float(score),
            model=res.get("model"),
            residuals=res.get("residuals") if ret_residuals else None,
            y_true=Z[eff_col].to_numpy(dtype=float) if ret_residuals else None,
            design=Z[cols].copy(),
        )

    def residual(
        self,
        X: pd.DataFrame,
        *,
        graph,
        variables: list[str],
    ) -> np.ndarray:
        """
        Build a multivariate residual/error signal from the current graph.

        Shape:
            (n_design_rows, n_variables)

        For GP/RFF scores, use pointwise predictive error bits.
        For other MDL scores, use squared standardized residuals.
        """
        self._ensure_bound(X)
        assert self._Z is not None

        previous_windows = self._time_windows
        self._time_windows = None

        try:
            columns: list[np.ndarray] = []

            for variable in variables:
                effect = (str(variable), 0)

                if graph is None or effect not in graph:
                    parents: tuple[TemporalNode, ...] = tuple()
                else:
                    parents = tuple(graph.predecessors(effect))

                result = self.score_edge(
                    X,
                    effect,
                    parents,
                    ret_full_result=True,
                    ret_residuals=True,
                )

                if isinstance(result, float):
                    raise RuntimeError("Expected TimeLocalScoreResult, got float.")

                if self._is_gp_score():
                    if not isinstance(result.model, dict):
                        raise RuntimeError("GP/RFF scorer must return a dict model.")

                    if "pointwise_error_bits" not in result.model:
                        raise RuntimeError(
                            "GP/RFF scorer must return model['pointwise_error_bits'] " "for changepoint detection."
                        )

                    errors = np.asarray(
                        result.model["pointwise_error_bits"],
                        dtype=float,
                    ).reshape(-1)

                else:
                    residuals = (
                        np.zeros(len(self._Z), dtype=float)
                        if result.residuals is None
                        else np.asarray(result.residuals, dtype=float).reshape(-1)
                    )

                    residuals = np.nan_to_num(
                        residuals,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                    std = float(np.std(residuals))
                    if std <= 1e-12:
                        standardized = residuals - float(np.mean(residuals))
                    else:
                        standardized = (residuals - float(np.mean(residuals))) / std

                    errors = standardized**2

                errors = np.nan_to_num(
                    errors,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                if len(errors) != len(self._Z):
                    min_len = min(len(errors), len(self._Z))
                    errors = errors[:min_len]

                columns.append(errors)

            if not columns:
                return np.zeros((len(self._Z), 1), dtype=float)

            min_len = min(len(col) for col in columns)
            return np.column_stack([col[:min_len] for col in columns])

        finally:
            self._time_windows = previous_windows

    def residual_time_grid(
        self,
        time_grid: TimeGrid,
        *,
        graph,
        variables: list[str],
    ) -> np.ndarray:
        """
        Build a global residual/error matrix over all contexts.

        Shape:
            (n_design_rows, n_contexts * n_variables)

        Assumes aligned contexts of equal length for now.
        """
        lengths = {dataset_id: len(X) for dataset_id, X in time_grid.datasets.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) != 1:
            raise NotImplementedError(
                "Global changepoint detection currently requires all time-series "
                f"contexts to have the same length, got {lengths}."
            )

        previous_global_n = self._global_n_samples

        try:
            signals = []

            for dataset_id in time_grid.dataset_ids:
                X_context = time_grid.datasets[dataset_id]
                signal = self.residual(
                    X_context,
                    graph=graph,
                    variables=variables,
                )
                signals.append(signal)

            if not signals:
                return np.empty((0, 0), dtype=float)

            min_len = min(signal.shape[0] for signal in signals)
            signals = [signal[:min_len, :] for signal in signals]

            return np.hstack(signals)

        finally:
            self._global_n_samples = previous_global_n

    def fit_panel(self, panel: TimeGrid) -> None:
        first = panel.first_dataset()
        self.fit(first)

        self._global_n_samples = int(
            sum(max(0, len(X_context) - self.tau_max) for X_context in panel.datasets.values())
        )

    def score_edge_panel(
        self,
        *,
        panel: TimeGrid,
        effect: TemporalNode,
        parents: Sequence[TemporalNode],
        partitions: SpaceTimeGridClusters,
    ) -> float:
        """
        Score one temporal local mechanism over context/regime partitions.

        For the target variable, pool samples that share the same context-cluster
        and regime-cluster, fit one local model per pooled group, and sum scores.
        """
        if effect[1] != 0:
            raise ValueError(f"Temporal effects must have lag 0, got {effect}.")

        if not self._node_to_col:
            self.fit_panel(panel)

        target = str(effect[0])
        parents = tuple(parents)

        intervals = self._intervals_from_partitions(panel, partitions)

        context_partition = partitions.contexts.get(
            target,
            {dataset_id: 0 for dataset_id in panel.dataset_ids},
        )

        regime_partition = partitions.regimes.get(
            target,
            {regime_id: regime_id for regime_id in range(len(intervals))},
        )

        context_clusters = sorted(set(context_partition.values()))
        regime_clusters = sorted(set(regime_partition.values()))

        total = 0.0
        any_group = False

        for context_cluster in context_clusters:
            dataset_ids = [
                dataset_id for dataset_id in panel.dataset_ids if context_partition.get(dataset_id) == context_cluster
            ]

            for regime_cluster in regime_clusters:
                group_samples = []

                for dataset_id in dataset_ids:
                    X_context = panel.datasets[dataset_id]

                    for regime_id, interval in enumerate(intervals):
                        if regime_partition.get(regime_id) != regime_cluster:
                            continue

                        sample = self._panel_interval_design(
                            X=X_context,
                            effect=effect,
                            parents=parents,
                            interval=interval,
                        )

                        if not sample.empty:
                            group_samples.append(sample)

                if not group_samples:
                    continue

                Z_group = pd.concat(group_samples, axis=0, ignore_index=True)

                total += float(
                    self._score_edge_on_design(
                        Z=Z_group,
                        effect=effect,
                        parents=parents,
                        ret_full_result=False,
                        ret_residuals=False,
                    )
                )

                any_group = True

        if not any_group:
            return float("inf")

        return float(total)

    def _intervals_from_partitions(
        self,
        panel: TimeGrid,
        partitions: SpaceTimeGridClusters,
    ) -> list[tuple[int, int]]:
        intervals_raw = partitions.diagnostics.get("intervals")

        if intervals_raw:
            return [(int(start), int(stop)) for start, stop in intervals_raw]

        n_samples = len(panel.first_dataset())
        return [(0, n_samples)]

    def _panel_interval_design(
        self,
        *,
        X: pd.DataFrame,
        effect: TemporalNode,
        parents: Sequence[TemporalNode],
        interval: tuple[int, int],
    ) -> pd.DataFrame:
        target = str(effect[0])

        eff_col = self._node_to_col[effect]
        parent_cols = [self._node_to_col[parent] for parent in parents]

        start, stop = interval
        max_parent_lag = max([lag for _, lag in parents], default=0)

        first_t = max(start, self.tau_max, max_parent_lag)

        rows = []

        for t in range(first_t, stop):
            row: dict[str, float] = {}

            row[eff_col] = float(X[target].iloc[t])

            for parent, parent_col in zip(parents, parent_cols, strict=True):
                parent_var, lag = parent
                row[parent_col] = float(X[str(parent_var)].iloc[t - int(lag)])

            rows.append(row)

        return pd.DataFrame(rows, columns=[eff_col, *parent_cols])

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return float(old_score - new_score)

    def score_is_better(self, a: float, b: float) -> bool:
        return bool(a > b)

    def score_significant(self, gain: float) -> bool:
        if self._global_n_samples is None:
            raise RuntimeError("Call fit(X) before score_significant().")

        threshold = float(self.score_type.gain_threshold(self._global_n_samples))
        return bool(gain > threshold)
