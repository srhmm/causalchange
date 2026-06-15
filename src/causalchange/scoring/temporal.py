from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigTemporal
from causalchange.core.results import GridCell, SCMClusteringResult
from causalchange.core.types import DataMode
from causalchange.domain.temporal import (
    TemporalNode,
    TimeGrid,
    util_changepoints_to_intervals,
)
from causalchange.scoring.base import BaseLocalScorer
from causalchange.scoring.tabular import SCMScoreTabular


@dataclass(frozen=True)
class TimeLocalScoreResult:
    """detailed result for one temporal local mechanism score."""

    score: float
    model: Any
    residuals: np.ndarray | None
    y_true: np.ndarray | None
    design: pd.DataFrame


class SCMScoreTemporal(BaseLocalScorer):
    """temporal scoring over nodes ``(variable, lag)``, where ``lag=0`` is the
    current time point and ``lag>0`` is a past value

    flow: build a design matrix then delegate scoring to ``SCMScoreTabular``.
    """

    def __init__(self, *, cfg: CausalChangeConfigTemporal):
        super().__init__(cfg)

        if cfg.data_mode not in (DataMode.TIME, DataMode.TIME_CONTEXTS):
            raise ValueError(f"SCMScoreTemporal expects temporal data, got {cfg.data_mode=}.")

        if cfg.tau_max <= 0:
            raise ValueError("tau_max must be positive.")

        self.tau_max = int(cfg.tau_max)

        self._time_windows: list[tuple[int, int]] | None = None

        self._tab = SCMScoreTabular(cfg)

        self._node_to_col: dict[TemporalNode, str] = {}
        self._Z: pd.DataFrame | None = None
        self._bound_key: tuple[int, tuple[str, ...], tuple[int, int]] | None = None

    def fit(self, X: pd.DataFrame, *, bound_key=None) -> None:
        X0 = self._stringify_columns(X)

        Z = self.build_design(X0)

        self._node_to_col = {
            (variable, lag): self._ar_col((variable, lag)) for variable in X0.columns for lag in range(self.tau_max + 1)
        }
        self._Z = Z
        self._bound_key = bound_key if bound_key is not None else self._df_key(X)
        self._set_global_n_samples(Z.shape[0])

        self._tab.fit(Z)

    def fit_panel(self, panel: TimeGrid) -> None:
        first = panel.first_dataset()
        self.fit(first)

        self._set_global_n_samples(sum(max(0, len(X_context) - self.tau_max) for X_context in panel.datasets.values()))

    def build_design(self, X: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}

        for variable in X.columns:
            variable = str(variable)

            for lag in range(self.tau_max + 1):
                cols[self._ar_col((variable, lag))] = X[variable].shift(lag)

        Z = pd.DataFrame(cols)
        Z = Z.iloc[self.tau_max :].copy()
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

        The lagged design matrix drops the first ``tau_max`` rows, so an
        original interval ``[a, b)`` maps to design rows
        ``[max(a, tau_max) - tau_max, b - tau_max)``.
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

    def clear_time_windows(self) -> None:
        self._time_windows = None

    def local_score(
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

        parents = tuple(parents)

        if self._time_windows is None:
            return self._local_score_design(
                Z=self._Z,
                effect=effect,
                parents=parents,
                ret_full_result=ret_full_result,
                ret_residuals=ret_residuals,
            )

        if ret_full_result:
            raise NotImplementedError("ret_full_result=True with multiple time windows is not implemented yet.")

        total = 0.0

        for start, stop in self._time_windows:
            Z_window = self._Z.iloc[start:stop].copy()

            total += float(
                self._local_score_design(
                    Z=Z_window,
                    effect=effect,
                    parents=parents,
                    ret_full_result=False,
                    ret_residuals=False,
                )
            )

        return float(total)

    def local_score_grid(
        self,
        *,
        panel: TimeGrid,
        effect: TemporalNode,
        parents: Sequence[TemporalNode],
        partitions: SCMClusteringResult,
    ) -> float:
        if effect[1] != 0:
            raise ValueError(f"Temporal effects must have lag 0, got {effect}.")

        if not self._node_to_col:
            self.fit_panel(panel)

        target = str(effect[0])
        parents = tuple(parents)

        target_clusters = partitions.cell_clusters.get(target)

        if not target_clusters:
            return float(self.local_score(panel.first_dataset(), effect, parents))

        cells_by_cluster: dict[int, list[GridCell]] = {}

        for cell, cluster_id in target_clusters.items():
            cells_by_cluster.setdefault(int(cluster_id), []).append(cell)

        total = 0.0
        any_group = False

        for cells in cells_by_cluster.values():
            samples = []

            for cell in cells:
                X_context = panel.datasets[cell.dataset_id]
                interval = partitions.intervals_by_context[cell.dataset_id][cell.interval_id]

                sample = self._panel_interval_design(
                    X=X_context,
                    effect=effect,
                    parents=parents,
                    interval=interval,
                )

                if not sample.empty:
                    samples.append(sample)

            if not samples:
                continue

            Z_group = pd.concat(samples, axis=0, ignore_index=True)

            total += float(
                self._local_score_design(
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
            ``(n_design_rows, n_variables)``

        The residual signal is based on squared standardized residuals from
        the local mechanism fits.
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

                result = self.local_score(
                    X,
                    effect,
                    parents,
                    ret_full_result=True,
                    ret_residuals=True,
                )

                if isinstance(result, float):
                    raise RuntimeError("Expected TimeLocalScoreResult, got float.")

                errors = self._residual_errors(result)

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
            ``(n_design_rows, n_contexts * n_variables)``

        Currently assumes aligned contexts of equal length.
        """

        lengths = {dataset_id: len(X) for dataset_id, X in time_grid.datasets.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) != 1:
            raise NotImplementedError(
                "Global changepoint detection currently requires all time-series contexts "
                f"to have the same length, got {lengths}."
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

    def _local_score_design(
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

        if ret_full_result:
            self._tab._ensure_bound(Z)
            assert self._tab._edges is not None

            j = self._tab._col_index[eff_col]
            pa = [self._tab._col_index[p] for p in parent_cols]

            score, res = self._tab._edges.local_score(
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

        return float(self._tab.local_score(Z, eff_col, parent_cols))

    def _residual_errors(self, result: TimeLocalScoreResult) -> np.ndarray:
        residuals = (
            np.zeros(len(result.design), dtype=float)
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

        return np.nan_to_num(
            errors,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _intervals_from_partitions(
        self,
        panel: TimeGrid,
        partitions: SCMClusteringResult,
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

    def _ar_col(self, node: TemporalNode) -> str:
        variable, lag = node
        return f"{variable}_lag{lag}"

    def _df_key(self, X: pd.DataFrame) -> tuple[int, tuple[str, ...], tuple[int, int]]:
        return (id(X), tuple(map(str, X.columns)), tuple(X.shape))

    def _ensure_bound(self, X: pd.DataFrame) -> None:
        key = self._df_key(X)

        if self._Z is None or not self._node_to_col or self._bound_key != key:
            self.fit(X, bound_key=key)
