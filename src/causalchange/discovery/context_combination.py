from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from causalchange.config.causal_change_config import CausalChangeConfigTabular
from causalchange.utils.union_find import union_find_components


from __future__ import annotations

from collections.abc import Hashable, Iterable

def _util_union_find_components(
    nodes: list[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
) -> list[list[Hashable]]:
    parent = {x: x for x in nodes}
    rank = {x: 0 for x in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in edges:
        union(a, b)

    comps: dict[Hashable, list[Hashable]] = {}
    for x in nodes:
        rx = find(x)
        comps.setdefault(rx, []).append(x)

    return list(comps.values())



@dataclass(frozen=True)
class ContextCombinationResult:
    total: float
    diagnostics: dict[str, Any]


class SkipCombination:
    """for single context"""

    def aggregate(
        self,
        *,
        contexts: dict[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple[Any, ...],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> ContextCombinationResult:
        if len(contexts) != 1:
            raise ValueError(
                f"NoAggregation expects exactly one context, got {len(contexts)}. "
                "Use ContextAggregation.CHAIN or ContextAggregation.LINC for multi-context data."
            )

        ctx, df = next(iter(contexts.items()))
        score = float(score_ctx(df))

        return ContextCombinationResult(
            total=score,
            diagnostics={
                "mode": "none",
                "context": ctx,
                "effect": effect,
                "parents": parents,
                "score": score,
            },
        )


@dataclass(frozen=True)
class ContextCombinationParams:
    method: Literal["components", "agglomerative"] = "components"
    gain_threshold: float = 0.0


class LINCContextCombination:
    def __init__(self, *, grouping: ContextCombinationParams, higher_is_better: bool):
        self.grouping = grouping
        self.higher_is_better = bool(higher_is_better)
        self.last_gain_matrix: np.ndarray | None = None
        self.last_gain_contexts: tuple[Hashable, ...] | None = None

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return (new_score - old_score) if self.higher_is_better else (old_score - new_score)

    def score_significant(self, gain: float) -> bool:
        return gain > float(self.grouping.gain_threshold)

    def aggregate(
        self,
        *,
        contexts: dict[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple,
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> ContextCombinationResult:
        # effect/parents are unused for LINC (scoring is encapsulated by score_ctx)
        ctx_ids = list(contexts.keys())
        n = len(ctx_ids)
        if n == 0:
            return ContextCombinationResult(total=0.0, diagnostics={})

        # per-context
        ctx_scores: dict[Hashable, float] = {c: float(score_ctx(contexts[c])) for c in ctx_ids}

        if n == 1:
            return ContextCombinationResult(
                total=float(ctx_scores[ctx_ids[0]]),
                diagnostics={
                    "groups": [frozenset([ctx_ids[0]])],
                    "ctx_scores": ctx_scores,
                },
            )

        if self.grouping.method == "agglomerative":
            total, diag = self._agglomerative(ctx_ids, contexts, score_ctx)
            return ContextCombinationResult(total=float(total), diagnostics=diag)

        # components method
        gain = np.zeros((n, n), dtype=float)
        edges: list[tuple[Hashable, Hashable]] = []

        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = ctx_ids[i], ctx_ids[j]
                pooled = pd.concat([contexts[ci], contexts[cj]], axis=0, ignore_index=True)
                pooled_score = float(score_ctx(pooled))
                g = float(self.transition_gain(ctx_scores[ci] + ctx_scores[cj], pooled_score))
                gain[i, j] = gain[j, i] = g
                if self.score_significant(g):
                    edges.append((ci, cj))

        self.last_gain_matrix = gain
        self.last_gain_contexts = tuple(ctx_ids)

        components = _util_union_find_components(ctx_ids, edges)

        total = 0.0
        for comp in components:
            pooled = pd.concat([contexts[c] for c in comp], axis=0, ignore_index=True)
            total += float(score_ctx(pooled))

        diag = {
            "method": "components",
            "gain_matrix": gain,
            "gain_contexts": tuple(ctx_ids),
            "edges": edges,
            "groups": [frozenset(c) for c in components],
            "ctx_scores": ctx_scores,
        }
        return ContextCombinationResult(total=float(total), diagnostics=diag)

    def _agglomerative(self, ctx_ids, contexts, score_ctx):
        groups = [frozenset([c]) for c in ctx_ids]

        def group_score(g):
            pooled = pd.concat([contexts[c] for c in g], axis=0, ignore_index=True)
            return float(score_ctx(pooled))

        scores = {g: group_score(g) for g in groups}

        while True:
            best_gain = float("-inf")
            best_pair = None
            best_score = None

            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    gi, gj = groups[i], groups[j]
                    merged = gi | gj
                    s_merged = group_score(merged)
                    g = float(self.transition_gain(scores[gi] + scores[gj], s_merged))
                    if g > best_gain:
                        best_gain = g
                        best_pair = (i, j, merged)
                        best_score = s_merged

            if best_pair is None or not self.score_significant(best_gain):
                break

            i, j, merged = best_pair
            gi, gj = groups[i], groups[j]

            groups = [g for k, g in enumerate(groups) if k not in (i, j)]
            groups.append(merged)

            scores.pop(gi)
            scores.pop(gj)
            scores[merged] = float(best_score)

        return float(sum(scores.values())), {
            "method": "agglomerative",
            "groups": list(scores.keys()),
            "scores": dict(scores),
        }


def _colname(node: Any) -> str:
    # supports either str (tabular) or (var, lag) temporal nodes
    if isinstance(node, tuple) and len(node) == 2:
        v, lag = node
        return f"{v}_lag{lag}"
    return str(node)


class CHAINContextCombination:
    """
    CHAIN idea w score = sum_ctx score_ctx(df_ctx)  +/- lambda_inv * invariance_penalty
    where penalty is MMD on pooled OLS residual distributions across contexts
    """

    def __init__(
        self,
        *,
        cfg: CausalChangeConfigTabular,
    ):
        self.lambda_inv = 1.0
        self.mmd_max_samples = 200
        self.mmd_gamma = None
        self.mmd_compare_to = "pooled"  # pairwise
        self.higher_is_better = bool(cfg.score_type.higher_is_better())

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
    ) -> ContextCombinationResult:
        ctx_ids = list(contexts.keys())
        if not ctx_ids:
            return ContextCombinationResult(total=0.0, diagnostics={})

        # fit term
        fit = 0.0
        for c in ctx_ids:
            fit += float(score_ctx(contexts[c]))

        if self.lambda_inv <= 0.0 or len(ctx_ids) <= 1:
            return ContextCombinationResult(total=float(fit), diagnostics={"fit": float(fit), "penalty": 0.0})

        pen = float(self._invariance_penalty(contexts, effect, parents))
        if self.higher_is_better:
            total = float(fit - self.lambda_inv * pen)
        else:
            total = float(fit + self.lambda_inv * pen)

        return ContextCombinationResult(
            total=total,
            diagnostics={
                "fit": float(fit),
                "penalty": float(pen),
                "lambda_inv": self.lambda_inv,
            },
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
        by_context: dict[Hashable, tuple[np.ndarray | None, np.ndarray]] = {}

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
