from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from pgmpy.base import DAG
Node = tuple[str, int]
LocalScoreFn = Callable[[str, Sequence[str]], float]
from typing import Optional, Sequence, Protocol, runtime_checkable, cast, Any

@runtime_checkable
class _DomainHost(Protocol):
    def _domain_prepare(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def _domain_nodes(self, X: pd.DataFrame) -> Sequence[Any]: ...
    def _domain_candidates(self, X: pd.DataFrame) -> list[Any]: ...
    def _domain_allowed_edge(self, u: Any, v: Any) -> bool: ...


@runtime_checkable
class _ScoreHost(Protocol):
    def _init_score(self, X: pd.DataFrame) -> None: ...
    def _score(self, effect: Any, parents: Sequence[Any]) -> float: ...
    def _transition_gain(self, old_score: float, new_score: float) -> float: ...
    def _score_significant(self, gain: float) -> bool: ...
    def _score_is_better(self, a: float, b: float) -> bool: ...


@runtime_checkable
class _SearchHost(_DomainHost, _ScoreHost, Protocol):
    pass
    #expert_knowledge: Optional[ExpertKnowledge]



class TOPICSearch:
    """
    Search logic extracted from TOPIC-style topological search.

    Requires domain hooks:
      - _domain_prepare
      - _domain_candidates
      - _domain_parent_candidates
      - _domain_allowed_edge

    Requires scoring hooks:
      - _init_score
      - _score
      - _transition_gain
      - _score_significant
      - _score_is_better
    """
    return_type: str = "dag"
    significance_level: float = 0.05

    def _host(self) -> _SearchHost:
        return cast(_SearchHost, self)

    def fit(self, X: pd.DataFrame, **kwargs):
        return self._fit(X, **kwargs)

    def _fit(self, X: pd.DataFrame, **kwargs):
        host = self._host()

        X0 = host._domain_prepare(X)
        host._init_score(X0)

        dag_current = DAG()
        for n in host._domain_nodes(X0):
            dag_current.add_node(n)



        candidates = list(host._domain_candidates(X0))
        topological_order = []
        history = []

        n_nodes = len(candidates)
        it = 0
        while it < n_nodes:
            source, source_meta = self._next_node_in_topological_order(candidates, dag_current, X0)
            candidates.remove(source)
            topological_order.append(source)

            added_edges, outgoing_scores = self._add_outgoing_edges(source, candidates, dag_current, X0)
            pruned_edges, incoming_scores = self._remove_ingoing_edges(source, dag_current, X0)

            history.append(
                {
                    "iteration": it,
                    "source": source,
                    "topological_order": list(topological_order),
                    "remaining_candidates": list(candidates),
                    "source_selection": source_meta,
                    "added_edges": added_edges,
                    "pruned_edges": pruned_edges,
                    "outgoing_scores": outgoing_scores,
                    "incoming_scores": incoming_scores,
                }
            )
            it += 1

        self.causal_graph_ = dag_current
        self.topological_order_ = topological_order
        self.history_ = history
        return self.causal_graph_


    def _addition_gain(self, cause, effect, dag_current, X0):
        host = self._host()
        parents = list(dag_current.get_parents(effect))
        old_score = host._score(effect, parents)
        new_parents = list(parents) + [cause]
        new_score = host._score(effect, new_parents)
        return host._transition_gain(old_score, new_score)


    def _improvement_matrix(self, candidates, dag_current, X0) -> np.ndarray:
        host = self._host()
        n = len(candidates)
        imp = np.zeros((n, n), dtype=float)
        idx = {node: i for i, node in enumerate(candidates)}

        for cause in candidates:
            for effect in candidates:
                if cause == effect:
                    continue
                if not host._domain_allowed_edge(cause, effect):
                    continue
                g = self._addition_gain(cause, effect, dag_current, X0)
                imp[idx[cause], idx[effect]] = float(g)

        return imp


    def _next_node_in_topological_order(self, candidates, dag_current, X0):
        improvement = self._improvement_matrix(candidates, dag_current, X0)
        delta = improvement - improvement.T
        np.fill_diagonal(delta, -np.inf)

        incoming_pressure = np.max(delta, axis=0)
        source_idx = int(np.argmin(incoming_pressure))
        source = candidates[source_idx]

        meta = {
            "candidates": list(candidates),
            "improvement_matrix": improvement.tolist(),
            "delta_matrix": delta.tolist(),
            "source_idx": source_idx,
        }
        return source, meta


    def _add_outgoing_edges(self, source, remaining, dag_current, X0):
        host = self._host()
        added_edges = []
        meta = []

        for node in remaining:
            if node == source:
                continue
            if not host._domain_allowed_edge(source, node):
                continue
            gain = self._addition_gain(source, node, dag_current, X0)
            significant = host._score_significant(gain)

            meta.append({"from": source, "to": node, "gain": float(gain), "significant": bool(significant)})

            if significant:
                dag_current.add_edge(source, node)
                added_edges.append({"from": source, "to": node, "gain": float(gain)})

        return added_edges, meta

    def _remove_ingoing_edges(self, source, dag_current, X0):
        pruned_edges = []
        meta = []

        parents = list(dag_current.get_parents(source))
        while parents:
            removed_found, removed_parent, best_gain, cand = self._find_removable_edge(parents, source, dag_current, X0)
            for parent, diff in cand:
                meta.append({"from": parent, "to": source, "diff": float(diff)})

            if removed_parent is None:
                break

            dag_current.remove_edge(removed_parent, source)
            parents.remove(removed_parent)
            pruned_edges.append({"from": removed_parent, "to": source, "diff": float(best_gain)})

        return pruned_edges, meta

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


class GLOBESearch:
    def fit(self, X: pd.DataFrame, **kwargs):
        raise NotImplementedError("GLOBE search logic not implemented yet.")

