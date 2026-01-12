from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional

import numpy as np
import pandas as pd

from causalchange.config.cc_config import CausalChangeConfig


@dataclass(frozen=True)
class AggregationResult:
    total: float
    diagnostics: dict[str, Any]


def _colname(node: Any) -> str:
    # supports either str (tabular) or (var, lag) temporal nodes
    if isinstance(node, tuple) and len(node) == 2:
        v, lag = node
        return f"{v}_lag{lag}"
    return str(node)


class ChainAggregator:
    """
    CHAIN aggregation score = sum_ctx score_ctx(df_ctx)  +/- lambda_inv * invariance_penalty
    where penalty is MMD on pooled OLS residual distributions across contexts.
    """

    def __init__(
        self,
        *,
            cfg: CausalChangeConfig,
    ):
        self.lambda_inv = 1.0
        self.mmd_max_samples = 200
        self.mmd_gamma = None
        self.mmd_compare_to = "pooled"#pairwise
        self.higher_is_better = cfg.higher_is_better

        seed = 42
        self._rng = np.random.default_rng(seed)
        self._pooled_cache: dict[tuple[str, tuple[str, ...]], tuple[dict[Hashable, np.ndarray], np.ndarray]] = {}
        self._mmd_cache: dict[tuple[Any, ...], float] = {}

    def aggregate(
        self,
        *,
        contexts: dict[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple,
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> AggregationResult:
        ctx_ids = list(contexts.keys())
        if not ctx_ids:
            return AggregationResult(total=0.0, diagnostics={})

        # fit term
        fit = 0.0
        for c in ctx_ids:
            fit += float(score_ctx(contexts[c]))

        if self.lambda_inv <= 0.0 or len(ctx_ids) <= 1:
            return AggregationResult(total=float(fit), diagnostics={"fit": float(fit), "penalty": 0.0})

        pen = float(self._invariance_penalty(contexts, effect, parents))
        if self.higher_is_better:
            total = float(fit - self.lambda_inv * pen)
        else:
            total = float(fit + self.lambda_inv * pen)

        return AggregationResult(
            total=total,
            diagnostics={"fit": float(fit), "penalty": float(pen), "lambda_inv": self.lambda_inv},
        )

    def _invariance_penalty(self, contexts: dict[Hashable, pd.DataFrame], effect: Any, parents: tuple) -> float:
        eff = _colname(effect)
        par = tuple(sorted(_colname(p) for p in parents))

        key = (eff, par, self.mmd_compare_to, self.mmd_max_samples, self.mmd_gamma)
        if key in self._mmd_cache:
            return float(self._mmd_cache[key])

        residuals_by_c, pooled_resid = self._pooled_residuals_cached(contexts, eff, par)

        pooled_resid_s = self._subsample_1d(pooled_resid, self.mmd_max_samples)
        if pooled_resid_s.size < 5 or len(residuals_by_c) <= 1:
            self._mmd_cache[key] = 0.0
            return 0.0

        if self.mmd_compare_to == "pairwise":
            cs = list(residuals_by_c.keys())
            pen = 0.0
            for i in range(len(cs)):
                ri = self._subsample_1d(residuals_by_c[cs[i]], self.mmd_max_samples)
                if ri.size < 5:
                    continue
                for j in range(i + 1, len(cs)):
                    rj = self._subsample_1d(residuals_by_c[cs[j]], self.mmd_max_samples)
                    if rj.size < 5:
                        continue
                    pen += self._mmd2_rbf(ri, rj, gamma=self.mmd_gamma)
        else:
            pen = 0.0
            for r in residuals_by_c.values():
                r_s = self._subsample_1d(r, self.mmd_max_samples)
                if r_s.size < 5:
                    continue
                pen += self._mmd2_rbf(r_s, pooled_resid_s, gamma=self.mmd_gamma)

        pen = float(pen)
        self._mmd_cache[key] = pen
        return pen

    def _pooled_residuals_cached(
        self,
        contexts: dict[Hashable, pd.DataFrame],
        effect: str,
        parents: tuple[str, ...],
    ):
        key = (effect, tuple(sorted(parents)))
        if key in self._pooled_cache:
            return self._pooled_cache[key]

        residuals_by_c, pooled_resid = self._pooled_residuals(contexts, effect, parents)
        residuals_by_c = {c: self._normalize_residuals(r) for c, r in residuals_by_c.items()}
        pooled_resid = self._normalize_residuals(pooled_resid)

        self._pooled_cache[key] = (residuals_by_c, pooled_resid)
        return residuals_by_c, pooled_resid

    def _pooled_residuals(
        self,
        contexts: dict[Hashable, pd.DataFrame],
        effect: str,
        parents: tuple[str, ...],
    ):
        ys = []
        Xps = []
        by_context: dict[Hashable, tuple[Optional[np.ndarray], np.ndarray]] = {}

        for c, df in contexts.items():
            y = df[effect].to_numpy(dtype=float)
            if len(parents) == 0:
                Xp = None
            else:
                Xp = df[list(parents)].to_numpy(dtype=float)
            ys.append(y)
            if Xp is not None:
                Xps.append(Xp)
            by_context[c] = (Xp, y)

        y_all = np.concatenate(ys, axis=0)

        if len(parents) == 0:
            mu = float(np.mean(y_all))
            pooled_resid = y_all - mu
            resid_by_c = {c: y - mu for c, (_, y) in by_context.items()}
            return resid_by_c, pooled_resid

        X_all = np.concatenate(Xps, axis=0)
        X_all_i = np.column_stack([np.ones((X_all.shape[0], 1)), X_all])
        beta, *_ = np.linalg.lstsq(X_all_i, y_all, rcond=None)

        pooled_pred = X_all_i @ beta
        pooled_resid = y_all - pooled_pred

        resid_by_c: dict[Hashable, np.ndarray] = {}
        for c, (Xp, y) in by_context.items():
            assert Xp is not None
            Xp_i = np.column_stack([np.ones((Xp.shape[0], 1)), Xp])
            resid_by_c[c] = y - (Xp_i @ beta)

        return resid_by_c, pooled_resid

    def _subsample_1d(self, x: np.ndarray, max_n: int) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size <= max_n:
            return x
        idx = self._rng.choice(x.size, size=max_n, replace=False)
        return x[idx]

    def _mmd2_rbf(self, x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if x.shape[0] == 0 or y.shape[0] == 0:
            return 0.0
        if gamma is None:
            gamma = self._median_heuristic_gamma_1d(np.vstack([x, y]))

        Kxx = self._rbf_kernel(x, x, gamma)
        Kyy = self._rbf_kernel(y, y, gamma)
        Kxy = self._rbf_kernel(x, y, gamma)
        return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())

    def _rbf_kernel(self, a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
        aa = np.sum(a * a, axis=1, keepdims=True)
        bb = np.sum(b * b, axis=1, keepdims=True).T
        d2 = aa + bb - 2.0 * (a @ b.T)
        return np.exp(-gamma * d2)

    def _median_heuristic_gamma_1d(self, z: np.ndarray) -> float:
        n = z.shape[0]
        if n <= 2:
            return 1.0
        m = min(n, 300)
        idx = self._rng.choice(n, size=m, replace=False)
        zz = z[idx]
        d = np.abs(zz - zz.T)
        med = np.median(d[d > 0])
        if not np.isfinite(med) or med <= 0:
            return 1.0
        sigma = float(med)
        return 1.0 / (2.0 * sigma * sigma)

    def _normalize_residuals(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        mu = float(r.mean())
        sd = float(r.std())
        if sd < 1e-8:
            return r - mu
        return (r - mu) / sd
