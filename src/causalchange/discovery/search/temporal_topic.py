from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

Node = tuple[str, int]
ScoreFunction = Callable[[Node, tuple[Node, ...]], float]
AllowedEdge = Callable[[Node, Node], bool]


@dataclass
class TemporalDAGSearchResult:
    graph: nx.DiGraph
    topological_order: list[str]
    history: list[dict[str, Any]]


class TemporalTopicSearch:
    def __init__(self, *, scoring):
        self.transition_gain = scoring.transition_gain
        self.score_significant = scoring.score_significant
        self.score_is_better = scoring.score_is_better

    def run(
        self,
        *,
        variables: Sequence[str],
        tau_max: int,
        allowed_edge: AllowedEdge,
        score_fun: ScoreFunction,
    ) -> TemporalDAGSearchResult:
        variables = list(map(str, variables))

        g = nx.DiGraph()
        nodes = [(v, lag) for lag in range(0, tau_max + 1) for v in variables]
        g.add_nodes_from(nodes)

        remaining = list(variables)
        order: list[str] = []
        history: list[dict[str, Any]] = []

        # First add significant lagged parents. They are temporally ordered by construction.
        lagged_added = self._add_lagged_edges(
            graph=g,
            variables=variables,
            tau_max=tau_max,
            allowed_edge=allowed_edge,
            score_fun=score_fun,
        )

        it = 0
        while remaining:
            source, source_meta = self._next_instantaneous_node(
                remaining=remaining,
                graph=g,
                allowed_edge=allowed_edge,
                score_fun=score_fun,
            )

            remaining.remove(source)
            order.append(source)

            added_edges, outgoing_scores = self._add_instantaneous_outgoing_edges(
                source=source,
                remaining=remaining,
                graph=g,
                allowed_edge=allowed_edge,
                score_fun=score_fun,
            )

            pruned_edges, incoming_scores = self._remove_ingoing_edges(
                source=(source, 0),
                graph=g,
                score_fun=score_fun,
            )

            history.append(
                {
                    "iteration": it,
                    "source": source,
                    "topological_order": list(order),
                    "remaining": list(remaining),
                    "source_selection": source_meta,
                    "added_edges": added_edges,
                    "pruned_edges": pruned_edges,
                    "outgoing_scores": outgoing_scores,
                    "incoming_scores": incoming_scores,
                }
            )

            it += 1

        history.insert(0, {"phase": "lagged_edges", "added_edges": lagged_added})
        return TemporalDAGSearchResult(graph=g, topological_order=order, history=history)

    def _addition_gain(
        self,
        cause: Node,
        effect: Node,
        graph: nx.DiGraph,
        score_fun: ScoreFunction,
    ) -> float:
        parents = tuple(graph.predecessors(effect))
        old_score = float(score_fun(effect, parents))
        new_score = float(score_fun(effect, parents + (cause,)))
        return float(self.transition_gain(old_score, new_score))

    def _add_lagged_edges(
        self,
        *,
        graph: nx.DiGraph,
        variables: Sequence[str],
        tau_max: int,
        allowed_edge: AllowedEdge,
        score_fun: ScoreFunction,
    ) -> list[dict[str, Any]]:
        added = []

        for effect_var in variables:
            effect = (effect_var, 0)

            for lag in range(1, tau_max + 1):
                for cause_var in variables:
                    cause = (cause_var, lag)

                    if not allowed_edge(cause, effect):
                        continue

                    gain = self._addition_gain(cause, effect, graph, score_fun)
                    if self.score_significant(gain):
                        graph.add_edge(cause, effect)
                        added.append(
                            {
                                "from": cause,
                                "to": effect,
                                "gain": float(gain),
                            }
                        )

        return added

    def _improvement_matrix(
        self,
        *,
        remaining: Sequence[str],
        graph: nx.DiGraph,
        allowed_edge: AllowedEdge,
        score_fun: ScoreFunction,
    ) -> np.ndarray:
        n = len(remaining)
        imp = np.zeros((n, n), dtype=float)
        idx = {v: i for i, v in enumerate(remaining)}

        for cause_var in remaining:
            for effect_var in remaining:
                if cause_var == effect_var:
                    continue

                cause = (cause_var, 0)
                effect = (effect_var, 0)

                if not allowed_edge(cause, effect):
                    continue

                gain = self._addition_gain(cause, effect, graph, score_fun)
                imp[idx[cause_var], idx[effect_var]] = float(gain)

        return imp

    def _next_instantaneous_node(
        self,
        *,
        remaining: Sequence[str],
        graph: nx.DiGraph,
        allowed_edge: AllowedEdge,
        score_fun: ScoreFunction,
    ):
        improvement = self._improvement_matrix(
            remaining=remaining,
            graph=graph,
            allowed_edge=allowed_edge,
            score_fun=score_fun,
        )

        delta = improvement - improvement.T
        np.fill_diagonal(delta, -np.inf)

        incoming_pressure = np.max(delta, axis=0)
        source_idx = int(np.argmin(incoming_pressure))
        source = list(remaining)[source_idx]

        meta = {
            "remaining": list(remaining),
            "improvement_matrix": improvement.tolist(),
            "delta_matrix": delta.tolist(),
            "source_idx": source_idx,
            "incoming_pressure": incoming_pressure.tolist(),
        }
        return source, meta

    def _add_instantaneous_outgoing_edges(
        self,
        *,
        source: str,
        remaining: Sequence[str],
        graph: nx.DiGraph,
        allowed_edge: AllowedEdge,
        score_fun: ScoreFunction,
    ):
        added_edges = []
        meta = []

        cause = (source, 0)

        for effect_var in remaining:
            effect = (effect_var, 0)

            if not allowed_edge(cause, effect):
                continue

            gain = self._addition_gain(cause, effect, graph, score_fun)
            significant = bool(self.score_significant(gain))

            meta.append(
                {
                    "from": cause,
                    "to": effect,
                    "gain": float(gain),
                    "significant": significant,
                }
            )

            if significant:
                graph.add_edge(cause, effect)
                added_edges.append(
                    {
                        "from": cause,
                        "to": effect,
                        "gain": float(gain),
                    }
                )

        return added_edges, meta

    def _remove_ingoing_edges(self, *, source: Node, graph: nx.DiGraph, score_fun: ScoreFunction):
        pruned_edges = []
        meta = []

        parents = list(graph.predecessors(source))
        while parents:
            removed_parent, best_gain, candidates = self._find_removable_edge(
                parents=parents,
                child=source,
                score_fun=score_fun,
            )

            for parent, diff in candidates:
                meta.append({"from": parent, "to": source, "diff": float(diff)})

            if removed_parent is None:
                break

            graph.remove_edge(removed_parent, source)
            parents.remove(removed_parent)
            pruned_edges.append(
                {
                    "from": removed_parent,
                    "to": source,
                    "diff": float(best_gain),
                }
            )

        return pruned_edges, meta

    def _find_removable_edge(self, *, parents: list[Node], child: Node, score_fun: ScoreFunction):
        old_score = float(score_fun(child, tuple(parents)))

        best_parent = None
        best_gain = float("-inf")
        candidate_stats = []

        for parent in parents:
            new_parents = tuple(p for p in parents if p != parent)
            new_score = float(score_fun(child, new_parents))
            gain_remove = float(self.transition_gain(old_score, new_score))
            candidate_stats.append((parent, float(gain_remove)))

            if self.score_significant(gain_remove) and (
                best_parent is None or self.score_is_better(gain_remove, best_gain)
            ):
                best_parent = parent
                best_gain = float(gain_remove)

        return best_parent, best_gain, candidate_stats
