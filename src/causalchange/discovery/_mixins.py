from __future__ import annotations

import pandas as pd
import numpy as np

from typing import Any, Callable, Hashable, Iterable, Literal ,Protocol, runtime_checkable, Optional, Sequence, cast
from dataclasses import dataclass

from causalchange._cc_types import ScoreType, MixingType
from causalchange._cc_types import DataMode
from causalchange.scoring.edge_score import EdgeScore


Node = tuple[str, int]
LocalScoreFn = Callable[[str, Sequence[str]], float]
ScoreFn = Callable[[str, tuple[str, ...]], float]

@runtime_checkable
class _HasTauMax(Protocol):
    tau_max: int


@runtime_checkable
class _TabularScoreHost(Protocol):
    data_mode: DataMode
    score_type: ScoreType
    mixing_type: MixingType
    score_params: dict[str, Any]
    score_higher_better: bool | None

    def _init_score(self, X: pd.DataFrame) -> None: ...
    def _score(self, effect, parents) -> float: ...
    def _transition_gain(self, old_score: float, new_score: float) -> float: ...
    def _score_is_better(self, a: float, b: float) -> bool: ...
    def _score_significant(self, gain: float) -> bool: ...


@runtime_checkable
class _AutoRegressiveHost(_TabularScoreHost, _HasTauMax, Protocol):
    def _ar_build_design(self, X: pd.DataFrame) -> pd.DataFrame: ...


@runtime_checkable
class _SpaceTimeHost(_HasTauMax, Protocol):
    data_mode: DataMode

    def _domain_allowed_edge(self, u: Node, v: Node) -> bool: ...
    def _addition_gain(self, u: Node, v: Node, dag_current) -> float: ...
    def _score_significant(self, gain: float) -> bool: ...
    def _score_is_better(self, a: float, b: float) -> bool: ...
    def _score(self, effect: Node, parents: Sequence[Node]) -> float: ...
    def _transition_gain(self, old_score: float, new_score: float) -> float: ...


@runtime_checkable
class _LincHost(Protocol):
    data_mode: DataMode
    score_higher_better: bool
    grouping: "LINCGroupingParams"
    _X_context: dict[Hashable, pd.DataFrame]

    def _init_contexts(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def _init_score(self, X: pd.DataFrame) -> None: ...
    def _score(self, effect, parents) -> float: ...
    def fit(self, X: pd.DataFrame, **kwargs): ...


class TabularDomainMixin:
    """
    Task: organises data and graph nodes for continuous domain
    """
    def _domain_prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def _domain_nodes(self, X: pd.DataFrame) -> list[str]:
        return list(X.columns)

    def _domain_candidates(self, X: pd.DataFrame) -> list[str]:
        return self._domain_nodes(X)

    def _domain_parent_candidates(self, child: str, remaining: list[str]) -> list[str]:
        return [p for p in remaining if p != child]

    def _domain_allowed_edge(self, u: str, v: str) -> bool:
        return u != v


class TemporalDomainMixin:
    """
    Task: organises data and graph nodes for temporal domain
    """
    tau_max: int = 1
    allow_instantaneous: bool = True

    def __init__(self, *args, tau_max: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if tau_max <= 0:
            raise ValueError("tau_max must be a positive integer.")
        self.tau_max = int(tau_max)

    def _domain_prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def _domain_nodes(self, X: pd.DataFrame) -> list[Node]:
        vars_ = list(X.columns)
        return [(v, lag) for lag in range(0, self.tau_max + 1) for v in vars_]

    def _domain_candidates(self, X: pd.DataFrame) -> list[Node]:
        return [(v, 0) for v in list(X.columns)]

    def _domain_parent_candidates(self, child: Node, remaining_lag0: list[Node]) -> list[Node]:
        v_child, lag_child = child
        if lag_child != 0:
            raise ValueError("This design assumes only lag-0 nodes are scored as effects.")

        parents: list[Node] = []

        if self.allow_instantaneous:
            parents.extend([p for p in remaining_lag0 if p != child])

        vars_ = [v for (v, _) in remaining_lag0]
        for lag in range(1, self.tau_max + 1):
            parents.extend([(v, lag) for v in vars_])

        return parents

    def _domain_allowed_edge(self, u: Node, v: Node) -> bool:
        (_, lag_u) = u
        (_, lag_v) = v
        if lag_v != 0:
            return False
        return True



class TabularScoreMixin:
    """
    Task: handles scoring for continuous domain (wraps around EdgeScore)
    """
    edges_state: EdgeScore
    data_mode: DataMode
    score_type:score_type
    mixing_type:MixingType
    score_params: dict[str, Any]
    score_higher_better: bool = False

    def _init_score(self, X: pd.DataFrame) -> None:
        assert hasattr(self, "data_mode")
        assert hasattr(self, "score_type")
        assert hasattr(self, "mixing_type")

        # Optional guard: this mixin is for IID-like scoring backends
        assert self.data_mode in (DataMode.IID, DataMode.CONTEXTS, DataMode.TIME, DataMode.TIME_CONTEXTS), self.data_mode

        X_np = X.to_numpy(dtype=float)

        self.edges_state = EdgeScore(
            X_np,
            data_mode=self.data_mode,
            score_type=self.score_type,
            mixing_type=self.mixing_type,
            **self.score_params,
        )

        # causalchange convention: typically minimize (MDL/penalty-like). Override if needed.
        #if self.score_higher_better is None:
        #    self.score_higher_better = False

        self._col_index = {c: i for i, c in enumerate(X.columns)}

    def _score(self, effect: str, parents: Sequence[str]) -> float:
        j = self._col_index[effect]
        pa = [self._col_index[p] for p in parents]
        return float(self.edges_state.score_edge(j, pa, ret_full_result=False))

    def _transition_gain(self, old_score: float, new_score: float) -> float:
        # Return "improvement": positive means better.
        assert self.score_higher_better is not None
        return (new_score - old_score) if self.score_higher_better else (old_score - new_score)

    def _score_is_better(self, a: float, b: float) -> bool:
        # IMPORTANT: gains are always compared the same way.
        return a > b

    def _score_significant(self, gain: float) -> bool:
        return gain > 0

class AutoRegressiveScoreMixin(TabularScoreMixin):
    """
    Task: handles scoring for temporal domain (wraps around EdgeScore)
    """
    _ar_df: Optional[pd.DataFrame] = None
    _node_to_col: dict[Node, str]
    tau_max: int

    def _host(self) -> _AutoRegressiveHost:
        return cast(_AutoRegressiveHost, self)

    def _ar_col(self, node: Node) -> str:
        v, lag = node
        return f"{v}_lag{lag}"

    def _ar_build_design(self, X: pd.DataFrame) -> pd.DataFrame:
        host = self._host()
        tau = int(host.tau_max)
        if tau <= 0:
            raise ValueError("tau_max must be positive for autoregressive scoring.")

        cols: dict[str, pd.Series] = {}
        for v in X.columns:
            for lag in range(0, tau + 1):
                cols[self._ar_col((v, lag))] = X[v].shift(lag)

        Z = pd.DataFrame(cols)
        Z = Z.iloc[tau:].copy()
        Z.reset_index(drop=True, inplace=True)
        return Z

    def _init_score(self, X: pd.DataFrame) -> None:
        host = self._host()
        assert host.data_mode in (DataMode.TIME, DataMode.TIME_CONTEXTS), host.data_mode

        Z = self._ar_build_design(X)

        self._node_to_col = {
            (v, lag): self._ar_col((v, lag))
            for v in X.columns
            for lag in range(0, int(host.tau_max) + 1)
        }
        self._ar_df = Z

        super()._init_score(Z)

    def _score(self, effect: Node, parents: Sequence[Node]) -> float:
        eff = self._node_to_col[effect]
        par = [self._node_to_col[p] for p in parents]
        return super()._score(eff, par)


class ContextScoreMixin(TabularScoreMixin):
    """
    Task: handles scoring for multi-context domain (wraps around EdgeScore)
    """
    context_col: str = "context"

    _X_context: dict[Hashable, pd.DataFrame]

    def _init_contexts(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.context_col not in X.columns:
            raise ValueError(f"context_col '{self.context_col}' not found")

        self._X_context = {}
        for ctx, g in X.groupby(self.context_col):
            self._X_context[ctx] = g.drop(columns=[self.context_col]).copy()

        return X.drop(columns=[self.context_col]).copy()

class SpaceTimeMixin:
    """Task: implements logic for SpaceTime (whatever is needed on top of IID DAG search logic)"""

    def _host(self) -> _SpaceTimeHost:
        return cast(_SpaceTimeHost, self)

    def _st_lagged_nodes_for_source(self, source: Node) -> list[Node]:
        host = self._host()
        var, lag = source
        if lag != 0:
            raise ValueError("SpaceTime expects lag-0 sources.")
        return [(var, l) for l in range(1, int(host.tau_max) + 1)]

    def _add_outgoing_edges(self, source, remaining, dag_current, X0):
        host = self._host()
        assert host.data_mode in (DataMode.TIME, DataMode.TIME_CONTEXTS), host.data_mode

        added_edges = []
        meta = []

        _, src_lag = source
        if src_lag != 0:
            raise ValueError("SpaceTime expects lag-0 sources.")

        lagged_sources = self._st_lagged_nodes_for_source(source)

        for target0 in remaining:
            if target0 == source:
                continue

            # instantaneous: (Xi,0)->(Xj,0)
            if host._domain_allowed_edge(source, target0):
                gain = host._addition_gain(source, target0, dag_current, X0)
                sig = host._score_significant(gain)
                meta.append(
                    {"from": source, "to": target0, "gain": float(gain), "kind": "inst", "significant": bool(sig)}
                )
                if sig:
                    dag_current.add_edge(source, target0)
                    added_edges.append({"from": source, "to": target0, "gain": float(gain), "kind": "inst"})

            # lagged: (Xi,lag)->(Xj,0) for lag=1..tau_max, same Xi
            for ls in lagged_sources:
                if host._domain_allowed_edge(ls, target0):
                    gain_l = host._addition_gain(ls, target0, dag_current, X0)
                    sig_l = host._score_significant(gain_l)
                    meta.append(
                        {"from": ls, "to": target0, "gain": float(gain_l), "kind": "lag", "significant": bool(sig_l)}
                    )
                    if sig_l:
                        dag_current.add_edge(ls, target0)
                        added_edges.append({"from": ls, "to": target0, "gain": float(gain_l), "kind": "lag"})

        return added_edges, meta

    def _find_removable_edge(self, parents, child, dag_current, X0):
        host = self._host()

        old_score = host._score(child, parents)

        best_parent = None
        best_gain = float("-inf")
        candidate_stats = []

        for parent in parents:
            new_parents = [p for p in parents if p != parent]
            new_score = host._score(child, new_parents)
            gain_remove = host._transition_gain(old_score, new_score)
            candidate_stats.append((parent, float(gain_remove)))

            if host._score_significant(gain_remove) and (
                best_parent is None or host._score_is_better(gain_remove, best_gain)
            ):
                best_gain = float(gain_remove)
                best_parent = parent

        if best_parent is None:
            return False, None, float("inf"), candidate_stats

        return True, best_parent, float(best_gain), candidate_stats

@dataclass(frozen=True)
class LINCGroupingParams:
    method: Literal["components", "agglomerative"] = "components"
    gain_threshold: float = 0.0


def _union_find_components(
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


class LINCMixin(ContextScoreMixin):
    """
    Task: implements logic of LINC (whatever is needed on top of single-dataset, iid scoring)
    """

    grouping: LINCGroupingParams = LINCGroupingParams()

    _last_gain_matrix: Optional[np.ndarray] = None
    _last_gain_contexts: Optional[tuple[Hashable, ...]] = None

    def _host(self) -> _LincHost:
        return cast(_LincHost, self)
 
    def __init__(self, *, context_col: str = "context", grouping=None, **kwargs):
        super().__init__(**kwargs)
        self.context_col = context_col
        if grouping is not None:
            self.grouping = grouping

    @staticmethod
    def _parents_to_tuple(parents) -> tuple:
        return tuple(parents) if parents is not None else tuple()

    def _transition_gain(self, old_score: float, new_score: float) -> float:
        host = self._host()
        return (new_score - old_score) if host.score_higher_better else (old_score - new_score)

    def _score_significant(self, gain: float) -> bool:
        host = self._host()
        return gain > host.grouping.gain_threshold

    def _score_components(self, effect, parents) -> float:
        host = self._host()

        # Optional guard
        assert host.data_mode in (DataMode.CONTEXTS, DataMode.TIME_CONTEXTS), host.data_mode

        parents_t = self._parents_to_tuple(parents)
        ctx_ids = list(host._X_context.keys())
        n = len(ctx_ids)

        if n == 0:
            return 0.0

        ctx_scores: dict[Hashable, float] = {}
        for c in ctx_ids:
            host._init_score(host._X_context[c])
            ctx_scores[c] = float(super()._score(effect, parents_t))

        if n == 1:
            return ctx_scores[ctx_ids[0]]

        gain = np.zeros((n, n), dtype=float)
        edges: list[tuple[Hashable, Hashable]] = []

        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = ctx_ids[i], ctx_ids[j]

                pooled = pd.concat([host._X_context[ci], host._X_context[cj]], axis=0, ignore_index=True)
                host._init_score(pooled)
                pooled_score = float(super()._score(effect, parents_t))

                g = self._transition_gain(ctx_scores[ci] + ctx_scores[cj], pooled_score)
                gain[i, j] = gain[j, i] = g

                if self._score_significant(g):
                    edges.append((ci, cj))

        self._last_gain_matrix = gain
        self._last_gain_contexts = tuple(ctx_ids)

        components = _union_find_components(ctx_ids, edges)

        total = 0.0
        for comp in components:
            pooled = pd.concat([host._X_context[c] for c in comp], axis=0, ignore_index=True)
            host._init_score(pooled)
            total += float(super()._score(effect, parents_t))

        return total

    def _score_agglomerative(self, effect, parents) -> float:
        host = self._host()

        parents_t = self._parents_to_tuple(parents)
        ctx_ids = list(host._X_context.keys())
        if not ctx_ids:
            return 0.0

        groups = [frozenset([c]) for c in ctx_ids]

        def group_score(g):
            pooled = pd.concat([host._X_context[c] for c in g], axis=0, ignore_index=True)
            host._init_score(pooled)
            return float(super()._score(effect, parents_t))

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
                    g = self._transition_gain(scores[gi] + scores[gj], s_merged)
                    if g > best_gain:
                        best_gain = g
                        best_pair = (i, j, merged)
                        best_score = s_merged

            if best_pair is None or not self._score_significant(best_gain):
                break

            i, j, merged = best_pair
            gi, gj = groups[i], groups[j]

            groups = [g for k, g in enumerate(groups) if k not in (i, j)]
            groups.append(merged)

            scores.pop(gi)
            scores.pop(gj)
            scores[merged] = best_score

        return float(sum(scores.values()))

    def _score(self, effect, parents) -> float:
        if self.grouping.method == "agglomerative":
            return self._score_agglomerative(effect, parents)
        else:
            return self._score_components(effect, parents)

    def fit(self, X: pd.DataFrame, **kwargs):
        host = self._host()

        X0 = host._init_contexts(X)
        host._init_score(X0)
        return super().fit(X0, **kwargs)





class CHAINMixin(ContextScoreMixin):
    """
    Task: CHAIN scoring (multi-context invariance penalty via MMD on pooled residuals).
    """

    lambda_inv: float = 1.0
    mmd_max_samples: int = 200
    mmd_gamma: float | None = None
    mmd_compare_to: str = "pooled"  # "pooled" or "pairwise"

    _rng: np.random.Generator
    _pooled_cache: dict[tuple[str, tuple[str, ...]], tuple[dict[Hashable, np.ndarray], np.ndarray]]
    _mmd_cache: dict[tuple[Any, ...], float]

    # store per-context EdgeScore + pooled EdgeScore
    _edges_by_ctx: dict[Hashable, EdgeScore]
    _edges_pooled: EdgeScore
    _col_index: dict[str, int]

    def __init__(
        self,
        *,
        context_col: str = "context",
        lambda_inv: float = 1.0,
        mmd_max_samples: int = 200,
        mmd_gamma: float | None = None,
        mmd_compare_to: str = "pooled",
        **kwargs,
    ):
        super().__init__(context_col=context_col, **kwargs)
        self.lambda_inv = float(lambda_inv)
        self.mmd_max_samples = int(mmd_max_samples)
        self.mmd_gamma = mmd_gamma
        self.mmd_compare_to = str(mmd_compare_to)
        self._rng = np.random.default_rng(0)
        self._pooled_cache = {}
        self._mmd_cache = {}
        self._edges_by_ctx = {}

    def _init_chain_score(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.data_mode in (DataMode.CONTEXTS, DataMode.TIME_CONTEXTS), self.data_mode

        X0 = self._init_contexts(X)  # set self._X_context
        self._col_index = {c: i for i, c in enumerate(X0.columns)}

        X0_np = X0.to_numpy(dtype=float)
        self._edges_pooled = EdgeScore(
            X0_np,
            data_mode=self.data_mode,
            score_type=self.score_type,
            mixing_type=self.mixing_type,
            **self.score_params,
        )

         # per-context EdgeScore
        self._edges_by_ctx = {}
        for ctx, dfc in self._X_context.items():
            Xc_np = dfc.to_numpy(dtype=float)
            self._edges_by_ctx[ctx] = EdgeScore(
                Xc_np,
                data_mode=self.data_mode,
                score_type=self.score_type,
                mixing_type=self.mixing_type,
                **self.score_params,
            )

        self._pooled_cache.clear()
        self._mmd_cache.clear()

        return X0

    def _score(self, effect: str, parents: Sequence[str]) -> float:
        # fit term: sum of context scores
        j = self._col_index[effect]
        pa = [self._col_index[p] for p in parents]

        fit = 0.0
        for es in self._edges_by_ctx.values():
            fit += float(es.score_edge(j, pa, ret_full_result=False))

        if self.lambda_inv <= 0.0 or len(self._edges_by_ctx) <= 1:
            return float(fit)

        pen = float(self._invariance_penalty(effect, parents))

        # IMPORTANT: sign depends on score_higher_better.
        # In causalchange you default score_higher_better=False (lower is better).
        assert self.score_higher_better is not None
        if self.score_higher_better:
            return float(fit - self.lambda_inv * pen)
        else:
            return float(fit + self.lambda_inv * pen)

    def _invariance_penalty(self, effect: str, parents: Sequence[str]) -> float:
        parents_key = tuple(sorted(str(p) for p in parents))
        key = (
            str(effect),
            parents_key,
            self.mmd_compare_to,
            int(self.mmd_max_samples),
            None if self.mmd_gamma is None else float(self.mmd_gamma),
        )
        if key in self._mmd_cache:
            return float(self._mmd_cache[key])

        residuals_by_c, pooled_resid = self._pooled_residuals_cached(effect, parents)

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

    def _pooled_residuals_cached(self, effect: str, parents: Sequence[str]):
        parents_key = tuple(sorted(str(p) for p in parents))
        key = (str(effect), parents_key)
        if key in self._pooled_cache:
            return self._pooled_cache[key]

        residuals_by_c, pooled_resid = self._pooled_residuals(effect, parents)
        residuals_by_c = {c: self._normalize_residuals(r) for c, r in residuals_by_c.items()}
        pooled_resid = self._normalize_residuals(pooled_resid)

        self._pooled_cache[key] = (residuals_by_c, pooled_resid)
        return residuals_by_c, pooled_resid

    def _pooled_residuals(self, effect: str, parents: Sequence[str]):
        # pooled OLS on concatenated contexts (like your old CHAIN)
        ys = []
        Xps = []
        by_context: dict[Hashable, tuple[Optional[np.ndarray], np.ndarray]] = {}

        for c, df in self._X_context.items():
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

    def fit(self, X: pd.DataFrame, **kwargs):
        X0 = self._init_chain_score(X)

        # prevent base mixins/search from rebuilding score state again
        orig_init = getattr(self, "_init_score", None)
        try:
            self._init_score = lambda _X: None  # type: ignore[assignment]
            return super().fit(X0, **kwargs)
        finally:
            if callable(orig_init):
                self._init_score = orig_init  # type: ignore[assignment]
