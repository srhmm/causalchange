from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigTemporal
from causalchange.core.require import _require_rpt
from causalchange.core.types import (
    ChangepointMethod,
    ChangepointMode,
    ChangepointScope,
)
from causalchange.domain.temporal import TimeGrid


class ChangepointDetection:
    def __init__(self, cfg: CausalChangeConfigTemporal):
        self.cfg = cfg
        self.changepoints_by_context_: dict[Any, list[int]] | None = None
        self.diagnostics_: dict[str, Any] = {}

    def detect(
        self,
        X: pd.DataFrame | None = None,
        *,
        time_grid: TimeGrid | None = None,
        graph: nx.DiGraph | None = None,
        scorer=None,
        variables: list[str] | None = None,
    ) -> list[int]:
        self.changepoints_by_context_ = None
        self.diagnostics_ = {}

        if self.cfg.changepoints == ChangepointMode.NONE:
            self.diagnostics_ = {
                "mode": "none",
                "scope": self.cfg.changepoint_scope.value,
                "changepoints": [],
            }
            return []

        if self.cfg.changepoints == ChangepointMode.FIXED:
            changepoints = list(self.cfg.fixed_changepoints)
            self.diagnostics_ = {
                "mode": "fixed",
                "scope": self.cfg.changepoint_scope.value,
                "changepoints": changepoints,
            }
            return changepoints

        if self.cfg.changepoints != ChangepointMode.DETECT:
            raise ValueError(f"Unsupported changepoint mode: {self.cfg.changepoints}")

        if scorer is None or variables is None:
            raise ValueError("DETECT changepoints requires scorer and variables.")

        if self.cfg.changepoint_method != ChangepointMethod.PELT:
            raise ValueError(f"Unsupported changepoint method: {self.cfg.changepoint_method}")

        if self.cfg.changepoint_scope == ChangepointScope.GLOBAL:
            return self._detect_global(
                X=X,
                panel=time_grid,
                graph=graph,
                scorer=scorer,
                variables=variables,
            )

        if self.cfg.changepoint_scope == ChangepointScope.PER_CONTEXT:
            return self._detect_per_context(
                X=X,
                panel=time_grid,
                graph=graph,
                scorer=scorer,
                variables=variables,
            )

        raise ValueError(f"Unsupported changepoint scope: {self.cfg.changepoint_scope}")

    def _detect_global(
        self,
        *,
        X: pd.DataFrame | None,
        panel: TimeGrid | None,
        graph: nx.DiGraph | None,
        scorer,
        variables: list[str],
    ) -> list[int]:
        if panel is not None:
            signal = scorer.residual_time_grid(
                panel,
                graph=graph,
                variables=variables,
            )
            n_raw_samples = len(panel.first_dataset())
        else:
            if X is None:
                raise ValueError("Either X or panel must be provided.")

            signal = scorer.residual(
                X,
                graph=graph,
                variables=variables,
            )
            n_raw_samples = len(X)

        changepoints, diagnostics = self._detect_from_signal(
            signal=signal,
            n_raw_samples=n_raw_samples,
            raw_offset=int(scorer.tau_max),
        )

        self.changepoints_by_context_ = None
        self.diagnostics_ = {
            "mode": "detect",
            "scope": ChangepointScope.GLOBAL.value,
            "changepoints": changepoints,
            "global": diagnostics,
        }
        return changepoints

    def _detect_per_context(
        self,
        *,
        X: pd.DataFrame | None,
        panel: TimeGrid | None,
        graph: nx.DiGraph | None,
        scorer,
        variables: list[str],
    ) -> list[int]:
        if panel is None:
            if X is None:
                raise ValueError("Either X or panel must be provided.")

            panel = TimeGrid(
                datasets={0: X},
                variables=[str(c) for c in X.columns],
                context_col=None,
            )

        by_context: dict[Any, list[int]] = {}
        diagnostics_by_context: dict[Any, dict[str, Any]] = {}

        for dataset_id in panel.dataset_ids:
            X_context = panel.datasets[dataset_id]

            signal = scorer.residual(
                X_context,
                graph=graph,
                variables=variables,
            )

            changepoints, diagnostics = self._detect_from_signal(
                signal=signal,
                n_raw_samples=len(X_context),
                raw_offset=int(scorer.tau_max),
            )

            by_context[dataset_id] = changepoints
            diagnostics_by_context[dataset_id] = diagnostics

        union_changepoints = sorted({cp for changepoints in by_context.values() for cp in changepoints})

        self.changepoints_by_context_ = by_context
        self.diagnostics_ = {
            "mode": "detect",
            "scope": ChangepointScope.PER_CONTEXT.value,
            "changepoints": union_changepoints,
            "by_context": by_context,
            "context_diagnostics": diagnostics_by_context,
            "note": (
                "Per-context changepoints are detected separately and their union "
                "is used as the downstream regime grid."
            ),
        }
        return union_changepoints

    def _detect_from_signal(
        self,
        *,
        signal: np.ndarray,
        n_raw_samples: int,
        raw_offset: int,
    ) -> tuple[list[int], dict[str, Any]]:
        signal = self._as_2d_signal(signal)

        penalty = self._resolve_pelt_penalty(signal)

        design_cps = self._detect_pelt_rbf(
            signal=signal,
            min_size=self.cfg.d_min,
            penalty=penalty,
        )

        raw_cps = [int(cp + raw_offset) for cp in design_cps]
        raw_cps = [cp for cp in raw_cps if 0 < cp < n_raw_samples]

        diagnostics = {
            "signal_shape": tuple(signal.shape),
            "selected_penalty": float(penalty),
            "design_changepoints": design_cps,
            "raw_changepoints": raw_cps,
            "raw_offset": raw_offset,
        }

        return raw_cps, diagnostics

    def _detect_pelt_rbf(
        self,
        *,
        signal: np.ndarray,
        min_size: int,
        penalty: float,
    ) -> list[int]:
        """
        Detect changepoints using ruptures.Pelt(model="rbf").

        ruptures returns segment endpoints and includes the final endpoint.
        We return only internal changepoints in signal/design coordinates.
        """
        rpt = _require_rpt()
        data = self._as_2d_signal(signal)

        algo = rpt.Pelt(
            model="rbf",
            min_size=int(min_size),
            jump=1,
        ).fit(data)

        bkps = algo.predict(pen=float(penalty))
        return [int(b) for b in bkps if 0 < b < data.shape[0]]

    def _as_2d_signal(self, signal: np.ndarray) -> np.ndarray:
        arr = np.asarray(signal, dtype=float)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D residual signal, got shape {arr.shape}")

        if arr.shape[0] == 0:
            return arr

        # Replace non-finite values without removing rows, because row index maps
        # directly to time index.
        for j in range(arr.shape[1]):
            col = arr[:, j]
            finite = np.isfinite(col)
            fill = float(np.mean(col[finite])) if np.any(finite) else 0.0
            arr[:, j] = np.where(finite, col, fill)

        mean = np.mean(arr, axis=0, keepdims=True)
        std = np.std(arr, axis=0, keepdims=True)
        std[std <= 1e-12] = 1.0

        return (arr - mean) / std

    def _resolve_pelt_penalty(self, signal: np.ndarray) -> float:
        penalty = self.cfg.pelt_penalty

        if isinstance(penalty, int | float):
            if penalty <= 0:
                raise ValueError("pelt_penalty must be positive.")
            return float(penalty)

        if not isinstance(penalty, str):
            raise TypeError(f"Unsupported pelt_penalty type: {type(penalty)!r}")

        penalty = penalty.lower()

        if penalty == "bic":
            return self._bic_penalty(signal)

        if penalty == "mbic":
            return self._mbic_penalty(signal)

        if penalty == "auto":
            return self._auto_penalty(signal)

        raise ValueError("pelt_penalty must be a positive number or one of " "{'bic', 'mbic', 'auto'}.")

    def _bic_penalty(self, signal: np.ndarray) -> float:
        n = max(int(signal.shape[0]), 2)
        return float(np.log(n))

    def _mbic_penalty(self, signal: np.ndarray) -> float:
        n = max(int(signal.shape[0]), 3)

        return float(np.log(n) + 2.0 * np.log(np.log(n)))

    def _auto_penalty(self, signal: np.ndarray) -> float:
        n = max(int(signal.shape[0]), 2)

        # At least d_min apart, so this is a rough upper bound on sensible cps.
        max_cps = max(1, n // max(int(self.cfg.d_min), 1) - 1)

        candidate_penalties = np.geomspace(0.25, 10.0, num=16)

        best_penalty = self._bic_penalty(signal)
        best_score = float("inf")

        for penalty in candidate_penalties:
            cps = self._detect_pelt_rbf(
                signal=signal,
                min_size=self.cfg.d_min,
                penalty=float(penalty),
            )
            n_cps = len(cps)

            if n_cps == 0:
                continue

            # Prefer a small number of changepoints, but avoid the empty solution.
            oversegmentation = max(0, n_cps - max_cps)
            score = oversegmentation * 10.0 + n_cps

            if score < best_score:
                best_score = score
                best_penalty = float(penalty)

        return float(best_penalty)
