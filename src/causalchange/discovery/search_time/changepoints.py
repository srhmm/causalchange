from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from causalchange.config.cc_config import ChangepointMethod, ChangepointMode, ChangepointScope, SpaceTimeConfig
from causalchange.discovery.search_time.base import TimePanel


class SpaceTimeChangepointDetection:
    def __init__(self, cfg: SpaceTimeConfig):
        self.cfg = cfg

    def detect(
        self,
        X: pd.DataFrame | None = None,
        *,
        panel: TimePanel | None = None,
        graph: nx.DiGraph | None = None,
        scorer=None,
        variables: list[str] | None = None,
    ) -> list[int]:
        if self.cfg.changepoints == ChangepointMode.NONE:
            return []

        if self.cfg.changepoints == ChangepointMode.FIXED:
            return list(self.cfg.fixed_changepoints)

        if self.cfg.changepoints != ChangepointMode.DETECT:
            raise ValueError(f"Unsupported changepoint mode: {self.cfg.changepoints}")

        if self.cfg.changepoint_scope == ChangepointScope.PER_CONTEXT:
            raise NotImplementedError("Per-context changepoints are not implemented yet.")

        if scorer is None or variables is None:
            raise ValueError("DETECT changepoints requires scorer and variables.")

        if panel is not None:
            signal = scorer.residual_signal_panel(
                panel,
                graph=graph,
                variables=variables,
            )
            n_raw_samples = len(panel.first_dataset())
        else:
            if X is None:
                raise ValueError("Either X or panel must be provided.")

            signal = scorer.residual_signal(
                X,
                graph=graph,
                variables=variables,
            )
            n_raw_samples = len(X)

        if self.cfg.changepoint_method == ChangepointMethod.PELT:
            design_cps = self._detect_pelt_rbf(
                signal=signal,
                min_size=self.cfg.d_min,
                penalty=self.cfg.pelt_penalty,
            )
        else:
            raise ValueError(f"Unsupported changepoint method: {self.cfg.changepoint_method}")

        # residual signal index 0 corresponds to raw time tau_max
        raw_offset = int(scorer.tau_max)
        raw_cps = [int(cp + raw_offset) for cp in design_cps]

        return [cp for cp in raw_cps if 0 < cp < n_raw_samples]

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
        try:
            import ruptures as rpt
        except ImportError as exc:
            raise ImportError(
                "ChangepointMethod.PELT requires the optional dependency 'ruptures'. "
                "Install it with `pip install ruptures`."
            ) from exc

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


def changepoints_to_intervals(
    n_samples: int,
    changepoints: list[int],
) -> list[tuple[int, int]]:
    """
    Convert changepoints into half-open intervals.

    Example:
        n_samples=100, changepoints=[30, 70]
        -> [(0, 30), (30, 70), (70, 100)]
    """
    cps = sorted(int(cp) for cp in changepoints)

    if any(cp <= 0 or cp >= n_samples for cp in cps):
        raise ValueError(f"changepoints must lie strictly inside [0, {n_samples}), got {changepoints}")

    if len(set(cps)) != len(cps):
        raise ValueError(f"changepoints must be unique, got {changepoints}")

    bounds = [0, *cps, int(n_samples)]
    return list(zip(bounds[:-1], bounds[1:], strict=False))
