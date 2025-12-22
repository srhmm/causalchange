from __future__ import annotations

from typing import Sequence

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Literal, Optional


from causalchange._cc_types import DataMode, ScoreType, MixingType
from causalchange.scoring.edge_score import EdgeScore
from typing import Protocol, runtime_checkable, cast

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
    score_higher_better: bool | None = None

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
        if self.score_higher_better is None:
            self.score_higher_better = False

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

    def _add_outgoing_edges(self, source, remaining, dag_current):
        host = self._host()

        # Optional guard
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

            if host._domain_allowed_edge(source, target0):
                gain = host._addition_gain(source, target0, dag_current)
                sig = host._score_significant(gain)
                meta.append({"from": source, "to": target0, "gain": float(gain), "kind": "inst", "significant": bool(sig)})
                if sig:
                    dag_current.add_edge(source, target0)
                    added_edges.append({"from": source, "to": target0, "gain": float(gain), "kind": "inst"})

            for ls in lagged_sources:
                if host._domain_allowed_edge(ls, target0):
                    gain_l = host._addition_gain(ls, target0, dag_current)
                    sig_l = host._score_significant(gain_l)
                    meta.append({"from": ls, "to": target0, "gain": float(gain_l), "kind": "lag", "significant": bool(sig_l)})
                    if sig_l:
                        dag_current.add_edge(ls, target0)
                        added_edges.append({"from": ls, "to": target0, "gain": float(gain_l), "kind": "lag"})

        return added_edges, meta

    def _find_removable_edge(self, parents, child, dag_current):
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
