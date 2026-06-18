from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from causalchange.core.results import ContextCombinationResult, LincMixtureResult, LincTargetMixtureResult
from causalchange.core.types import ContextCombinationKwargs, StatisticalTestingMethod, TabularMechanismClusteringMethod


class SkipCombination:
    """for single context"""

    def combine_contexts(
        self,
        *,
        contexts: Mapping[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple[Any, ...],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> ContextCombinationResult:
        assert (
            len(contexts) == 1
        ), f"expect one context, got {len(contexts)}. use TabularContextMethod.LINC for context data"

        ctx, df = next(iter(contexts.items()))
        score = float(score_ctx(df))

        return ContextCombinationResult(
            total=score,
            diagnostics={"mode": "none", "context": ctx, "effect": effect, "parents": parents, "score": score},
        )


class LINCContextCombination:
    def __init__(
        self,
        *,
        grouping: ContextCombinationKwargs,
        gain_threshold: float,
        higher_is_better: bool,
        mechanism_clustering_method: TabularMechanismClusteringMethod = TabularMechanismClusteringMethod.SCORE_MERGE,
        testing_method: StatisticalTestingMethod = StatisticalTestingMethod.SKIP,
        mechanism_test_alpha: float = 0.05,
        mechanism_clustering_n_clusters: int | None = None,
        mechanism_clustering_distance_threshold: float | None = None,
        seed: int = 42,
    ):
        self.grouping = grouping
        self.gain_threshold = float(gain_threshold)
        self.higher_is_better = bool(higher_is_better)
        self.mechanism_clustering_method = TabularMechanismClusteringMethod(mechanism_clustering_method)
        self.testing_method = StatisticalTestingMethod(testing_method)
        self.mechanism_test_alpha = float(mechanism_test_alpha)
        self.mechanism_clustering_n_clusters = mechanism_clustering_n_clusters
        self.mechanism_clustering_distance_threshold = mechanism_clustering_distance_threshold
        self._rng = np.random.default_rng(seed)
        self._results_by_mechanism: dict[tuple[Any, tuple[Any, ...]], ContextCombinationResult] = {}
        self.last_gain_matrix: np.ndarray | None = None
        self.last_gain_contexts: tuple[Hashable, ...] | None = None

    def combine_contexts(
        self,
        *,
        contexts: Mapping[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple[Any, ...],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> ContextCombinationResult:
        if not contexts:
            raise ValueError("LINCContextCombination requires at least one context.")

        if self.mechanism_clustering_method == TabularMechanismClusteringMethod.SCORE_MERGE:
            result = self._combine_score_merge(contexts=contexts, score_ctx=score_ctx)
        elif self.mechanism_clustering_method == TabularMechanismClusteringMethod.TESTING:
            result = self._combine_testing(
                contexts=contexts,
                effect=effect,
                parents=parents,
                score_ctx=score_ctx,
            )
        elif self.mechanism_clustering_method == TabularMechanismClusteringMethod.CLUSTERING:
            result = self._combine_clustering(
                contexts=contexts,
                effect=effect,
                parents=parents,
                score_ctx=score_ctx,
            )
        else:
            raise ValueError(f"Unknown mechanism clustering method: {self.mechanism_clustering_method!r}")

        return self._record_result(effect=effect, parents=parents, result=result)

    def final_linc_components(self, graph) -> LincMixtureResult:
        components: dict[Any, LincTargetMixtureResult] = {}
        failures: list[dict[str, Any]] = []

        for target in graph.nodes():
            parents = tuple(sorted(graph.predecessors(target), key=repr))
            combo = self._results_by_mechanism.get(self._mechanism_key(target, parents))

            assert combo is not None, "missing cached local score"

            groups_raw = combo.diagnostics.get("groups")

            assert groups_raw is not None
            groups = [frozenset(g) for g in groups_raw]

            components[target] = LincTargetMixtureResult(
                target=target,
                parents=parents,
                groups=groups,
                labels_by_context=self._labels_from_groups(groups),
                score=float(combo.total),
                n_components=len(groups),
                diagnostics=dict(combo.diagnostics),
            )

        return LincMixtureResult(
            target_components=components,
            diagnostics={
                "mode": "linc_final_graph_context_partitions",
                "method": self.mechanism_clustering_method.value,
                "failures": failures,
                "n_cached_mechanisms": len(self._results_by_mechanism),
            },
        )

    def _combine_score_merge(
        self,
        *,
        contexts: Mapping[Hashable, pd.DataFrame],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> ContextCombinationResult:
        ctx_ids = list(contexts)
        ctx_scores = {c: float(score_ctx(contexts[c])) for c in ctx_ids}

        if len(ctx_ids) == 1:
            ctx = ctx_ids[0]
            return ContextCombinationResult(
                total=ctx_scores[ctx],
                diagnostics={
                    "method": "score-merge-single-context",
                    "groups": [frozenset([ctx])],
                    "ctx_scores": ctx_scores,
                },
            )

        if self.grouping == ContextCombinationKwargs.AGGLOMERATIVE:
            groups, diag = self._score_merge_agglomerative(ctx_ids, contexts, score_ctx, ctx_scores)
        else:
            groups, diag = self._score_merge_components(ctx_ids, contexts, score_ctx, ctx_scores)

        total, group_scores = self._score_groups(groups=groups, contexts=contexts, score_ctx=score_ctx)
        diag["group_scores"] = group_scores
        return ContextCombinationResult(total=total, diagnostics=diag)

    def _combine_testing(
        self,
        *,
        contexts: Mapping[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple[Any, ...],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> ContextCombinationResult:
        ctx_ids = list(contexts)
        ctx_scores = {c: float(score_ctx(contexts[c])) for c in ctx_ids}

        if len(ctx_ids) == 1:
            groups = [frozenset([ctx_ids[0]])]
            total, group_scores = self._score_groups(groups=groups, contexts=contexts, score_ctx=score_ctx)
            return ContextCombinationResult(
                total=total,
                diagnostics={
                    "method": "statistical-testing-single-context",
                    "groups": groups,
                    "ctx_scores": ctx_scores,
                    "group_scores": group_scores,
                    "tests": [],
                },
            )

        if self.testing_method == StatisticalTestingMethod.SKIP:
            groups = [frozenset([c]) for c in ctx_ids]
            total, group_scores = self._score_groups(groups=groups, contexts=contexts, score_ctx=score_ctx)
            return ContextCombinationResult(
                total=total,
                diagnostics={
                    "method": "statistical-testing-skipped",
                    "groups": groups,
                    "ctx_scores": ctx_scores,
                    "group_scores": group_scores,
                    "tests": [],
                },
            )

        edges: list[tuple[Hashable, Hashable]] = []
        tests: list[dict[str, Any]] = []
        for i, ci in enumerate(ctx_ids):
            for cj in ctx_ids[i + 1 :]:
                pvalue, statistic = self._residual_mmd_test(
                    contexts[ci],
                    contexts[cj],
                    effect=effect,
                    parents=parents,
                )
                same = pvalue >= self.mechanism_test_alpha
                if same:
                    edges.append((ci, cj))
                tests.append({"left": ci, "right": cj, "same": same, "pvalue": pvalue, "statistic": statistic})

        groups = [frozenset(group) for group in _util_union_find_components(ctx_ids, edges)]
        total, group_scores = self._score_groups(groups=groups, contexts=contexts, score_ctx=score_ctx)
        return ContextCombinationResult(
            total=total,
            diagnostics={
                "method": "statistical-testing",
                "testing_method": self.testing_method.value,
                "alpha": self.mechanism_test_alpha,
                "groups": groups,
                "ctx_scores": ctx_scores,
                "group_scores": group_scores,
                "tests": tests,
            },
        )

    def _combine_clustering(
        self,
        *,
        contexts: Mapping[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple[Any, ...],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> ContextCombinationResult:
        ctx_ids = list(contexts)
        ctx_scores = {c: float(score_ctx(contexts[c])) for c in ctx_ids}
        features = np.asarray(
            [
                self._mechanism_signature(contexts[c], effect=effect, parents=parents, score=ctx_scores[c])
                for c in ctx_ids
            ],
            dtype=float,
        )
        labels = self._cluster_features(features)
        groups = self._groups_from_labels(ctx_ids, labels)
        total, group_scores = self._score_groups(groups=groups, contexts=contexts, score_ctx=score_ctx)
        return ContextCombinationResult(
            total=total,
            diagnostics={
                "method": "mechanism-clustering",
                "groups": groups,
                "labels_by_context": {co: int(lbl) for co, lbl in zip(ctx_ids, labels, strict=True)},
                "ctx_scores": ctx_scores,
                "group_scores": group_scores,
                "features": features,
                "n_clusters": len(groups),
                "distance_threshold": self.mechanism_clustering_distance_threshold,
            },
        )

    def _score_merge_agglomerative(
        self,
        ctx_ids: list[Hashable],
        contexts: Mapping[Hashable, pd.DataFrame],
        score_ctx: Callable[[pd.DataFrame], float],
        ctx_scores: dict[Hashable, float],
    ) -> tuple[list[frozenset[Any]], dict[str, Any]]:
        groups = [frozenset([c]) for c in ctx_ids]
        scores = {g: ctx_scores[next(iter(g))] for g in groups}
        merge_history: list[dict[str, Any]] = []
        stop_info: dict[str, Any] = {}

        while True:
            best_gain = float("-inf")
            best_pair: tuple[int, int, frozenset[Any]] | None = None
            best_score: float | None = None

            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    merged = groups[i] | groups[j]
                    merged_score = self._score_group(merged, contexts, score_ctx)
                    gain = self.transition_gain(scores[groups[i]] + scores[groups[j]], merged_score)
                    if gain > best_gain:
                        best_gain = float(gain)
                        best_pair = (i, j, merged)
                        best_score = float(merged_score)

            if best_pair is None or not self.score_significant(best_gain):
                stop_info = {"stop_best_gain": best_gain, "gain_threshold": self.gain_threshold}
                if best_pair is not None:
                    i, j, merged = best_pair
                    stop_info["stop_best_pair"] = (sorted(groups[i]), sorted(groups[j]), sorted(merged))
                    stop_info["stop_best_score"] = best_score
                break

            i, j, merged = best_pair
            left, right = groups[i], groups[j]
            merge_history.append(
                {"left": left, "right": right, "merged": merged, "gain": best_gain, "score": best_score}
            )
            groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [merged]
            scores.pop(left)
            scores.pop(right)
            scores[merged] = float(best_score)

        return groups, {
            "method": "score-merge-agglomerative",
            "groups": groups,
            "ctx_scores": ctx_scores,
            "merge_history": merge_history,
            **stop_info,
        }

    def _score_merge_components(
        self,
        ctx_ids: list[Hashable],
        contexts: Mapping[Hashable, pd.DataFrame],
        score_ctx: Callable[[pd.DataFrame], float],
        ctx_scores: dict[Hashable, float],
    ) -> tuple[list[frozenset[Any]], dict[str, Any]]:
        n = len(ctx_ids)
        gain = np.zeros((n, n), dtype=float)
        edges: list[tuple[Hashable, Hashable]] = []

        for i, ci in enumerate(ctx_ids):
            for j, cj in enumerate(ctx_ids[i + 1 :], start=i + 1):
                merged_score = self._score_group(frozenset([ci, cj]), contexts, score_ctx)
                gain[i, j] = gain[j, i] = self.transition_gain(ctx_scores[ci] + ctx_scores[cj], merged_score)
                if self.score_significant(gain[i, j]):
                    edges.append((ci, cj))

        self.last_gain_matrix = gain
        self.last_gain_contexts = tuple(ctx_ids)
        groups = [frozenset(group) for group in _util_union_find_components(ctx_ids, edges)]
        return groups, {
            "method": "score-merge-components",
            "groups": groups,
            "ctx_scores": ctx_scores,
            "gain_matrix": gain,
            "gain_contexts": tuple(ctx_ids),
            "edges": edges,
            "gain_threshold": self.gain_threshold,
        }

    def _record_result(
        self,
        *,
        effect: Any,
        parents: tuple[Any, ...],
        result: ContextCombinationResult,
    ) -> ContextCombinationResult:
        parents_t = tuple(sorted(parents, key=repr))
        recorded = ContextCombinationResult(
            total=float(result.total),
            diagnostics={**dict(result.diagnostics), "effect": effect, "parents": parents_t},
        )
        self._results_by_mechanism[self._mechanism_key(effect, parents_t)] = recorded
        return recorded

    def _score_group(
        self,
        group: frozenset[Any],
        contexts: Mapping[Hashable, pd.DataFrame],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> float:
        return float(score_ctx(pd.concat([contexts[c] for c in group], axis=0, ignore_index=True)))

    def _score_groups(
        self,
        *,
        groups: list[frozenset[Any]],
        contexts: Mapping[Hashable, pd.DataFrame],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> tuple[float, dict[frozenset[Any], float]]:
        group_scores = {group: self._score_group(group, contexts, score_ctx) for group in groups}
        return float(sum(group_scores.values())), group_scores

    def _residual_mmd_test(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        *,
        effect: Any,
        parents: tuple[Any, ...],
        n_perm: int = 99,
        max_samples: int = 200,
    ) -> tuple[float, float]:
        r_left, r_right = self._pooled_residuals(left, right, effect=effect, parents=parents)
        r_left = self._subsample(r_left, max_samples)
        r_right = self._subsample(r_right, max_samples)
        if len(r_left) < 5 or len(r_right) < 5:
            return 1.0, 0.0

        observed = self._mmd2(r_left, r_right)
        pooled = np.concatenate([r_left, r_right])
        n_left = len(r_left)
        count = 1
        for _ in range(n_perm):
            perm = self._rng.permutation(len(pooled))
            stat = self._mmd2(pooled[perm[:n_left]], pooled[perm[n_left:]])
            count += int(stat >= observed)
        return float(count / (n_perm + 1)), float(observed)

    def _pooled_residuals(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        *,
        effect: Any,
        parents: tuple[Any, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        y_left = left[_colname(effect)].to_numpy(dtype=float)
        y_right = right[_colname(effect)].to_numpy(dtype=float)

        if not parents:
            mean = float(np.mean(np.concatenate([y_left, y_right])))
            return y_left - mean, y_right - mean

        X_left = left[[_colname(p) for p in parents]].to_numpy(dtype=float)
        X_right = right[[_colname(p) for p in parents]].to_numpy(dtype=float)
        X = np.vstack([X_left, X_right])
        y = np.concatenate([y_left, y_right])
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
        r_left = y_left - np.column_stack([np.ones(len(X_left)), X_left]) @ beta
        r_right = y_right - np.column_stack([np.ones(len(X_right)), X_right]) @ beta
        return r_left, r_right

    def _mechanism_signature(
        self,
        df: pd.DataFrame,
        *,
        effect: Any,
        parents: tuple[Any, ...],
        score: float,
    ) -> list[float]:
        y = df[_colname(effect)].to_numpy(dtype=float)
        if not parents:
            resid = y - float(np.mean(y))
            return [float(np.mean(y)), float(np.std(y)), float(np.mean(resid)), float(np.std(resid)), float(score)]

        X = df[[_colname(p) for p in parents]].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
        resid = y - np.column_stack([np.ones(len(X)), X]) @ beta
        return [
            *[float(v) for v in beta],
            float(np.mean(resid)),
            float(np.std(resid)),
            float(np.mean(y)),
            float(np.std(y)),
            float(score),
        ]

    def _cluster_features(self, features: np.ndarray) -> np.ndarray:
        n = features.shape[0]
        if n <= 1 or np.allclose(features, features[0]):
            return np.zeros(n, dtype=int)

        if self.mechanism_clustering_n_clusters is None and self.mechanism_clustering_distance_threshold is None:
            return np.zeros(n, dtype=int)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = StandardScaler().fit_transform(features)
        model = AgglomerativeClustering(
            n_clusters=None
            if self.mechanism_clustering_distance_threshold is not None
            else self.mechanism_clustering_n_clusters,
            distance_threshold=self.mechanism_clustering_distance_threshold,
        )
        return model.fit_predict(features).astype(int)

    @staticmethod
    def _mechanism_key(effect: Any, parents: tuple[Any, ...]) -> tuple[Any, tuple[Any, ...]]:
        return effect, tuple(sorted(parents, key=repr))

    @staticmethod
    def _labels_from_groups(groups: list[frozenset[Any]]) -> dict[Any, int]:
        return {context_id: label for label, group in enumerate(groups) for context_id in group}

    @staticmethod
    def _groups_from_labels(items: list[Hashable], labels: Iterable[int]) -> list[frozenset[Any]]:
        groups: dict[int, list[Hashable]] = {}
        for item, label in zip(items, labels, strict=True):
            groups.setdefault(int(label), []).append(item)
        return [frozenset(groups[label]) for label in sorted(groups)]

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return (new_score - old_score) if self.higher_is_better else (old_score - new_score)

    def score_significant(self, gain: float) -> bool:
        return float(gain) > self.gain_threshold

    def _subsample(self, x: np.ndarray, max_samples: int) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if len(x) <= max_samples:
            return x
        return x[self._rng.choice(len(x), size=max_samples, replace=False)]

    def _mmd2(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        z = np.vstack([x, y])
        scale = np.median(np.abs(z - z.T))
        gamma = 1.0 if not np.isfinite(scale) or scale <= 0 else 1.0 / (2.0 * scale * scale)
        return float(
            self._rbf(x, x, gamma).mean() + self._rbf(y, y, gamma).mean() - 2.0 * self._rbf(x, y, gamma).mean()
        )

    @staticmethod
    def _rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
        d2 = np.sum(x * x, axis=1, keepdims=True) + np.sum(y * y, axis=1, keepdims=True).T - 2.0 * x @ y.T
        return np.exp(-gamma * d2)


def _colname(node: Any) -> str:
    if isinstance(node, tuple) and len(node) == 2:
        var, lag = node
        return f"{var}_lag{lag}"
    return str(node)


def _util_union_find_components(
    nodes: list[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
) -> list[list[Hashable]]:
    parent = {x: x for x in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups: dict[Hashable, list[Hashable]] = {}
    for node in nodes:
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


class CHAINContextCombination:
    """
    CHAIN idea w score = sum_ctx score_ctx(df_ctx)  +/- lambda_inv * invariance_penalty
    where penalty is MMD on pooled OLS residual distributions across contexts
    """

    def __init__(
        self,
        *,
        higher_is_better: bool,
        seed: int = 42,
        lambda_inv: float = 1.0,
    ):
        self.lambda_inv = lambda_inv
        self.higher_is_better = higher_is_better
        self._rng = np.random.default_rng(seed)
        self.mmd_max_samples = 200
        self.mmd_gamma = None
        self.mmd_compare_to = "pooled"  # pairwise

        self.seed = seed
        self._pooled_cache: dict[tuple[str, tuple[str, ...]], tuple[dict[Hashable, np.ndarray], np.ndarray]] = {}
        self._mmd_cache: dict[tuple[Any, ...], float] = {}

    def combine_contexts(
        self,
        *,
        contexts: Mapping[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple[Any, ...],
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

    def _invariance_penalty(
        self,
        contexts: Mapping[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple[Any, ...],
    ) -> float:
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
