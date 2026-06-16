from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np

from causalchange.core.results import GraphSearchResult
from causalchange.domain.temporal import TemporalAllowedEdge, TemporalNode, TemporalScoreFunction


class GraphSearchTemporalGreedy:
    """
    Edge-greedy search for a window causal graph.

    Lagged edges are always time-ordered. Instantaneous lag-0 edges are only
    added if they keep the lag-0 subgraph acyclic.
    """

    def __init__(self, *, scoring):
        self.transition_gain = scoring.transition_gain
        self.score_significant = scoring.score_significant
        self.gain_is_better = scoring.gain_is_better

    def run(
        self,
        *,
        variables: Sequence[str],
        tau_max: int,
        allowed_edge: TemporalAllowedEdge,
        score_fun: TemporalScoreFunction,
    ) -> GraphSearchResult:
        variables = list(map(str, variables))

        graph = nx.DiGraph()
        nodes = [(v, lag) for lag in range(0, tau_max + 1) for v in variables]
        graph.add_nodes_from(nodes)

        history: list[dict[str, Any]] = []

        forward_history = self._forward_phase(
            graph=graph,
            variables=variables,
            tau_max=tau_max,
            allowed_edge=allowed_edge,
            score_fun=score_fun,
        )

        backward_history = self._backward_phase(
            graph=graph,
            score_fun=score_fun,
        )

        history.extend(forward_history)
        history.extend(backward_history)

        topological_order = self._instantaneous_topological_order(graph, variables)

        return GraphSearchResult(
            graph=graph,
            topological_order=topological_order,
            history=history,
        )

    def _candidate_edges(
        self,
        *,
        variables: Sequence[str],
        tau_max: int,
        allowed_edge: TemporalAllowedEdge,
    ) -> list[tuple[TemporalNode, TemporalNode]]:
        edges: list[tuple[TemporalNode, TemporalNode]] = []

        # Lagged candidate edges: X(t-lag) -> Y(t)
        for effect_var in variables:
            effect = (effect_var, 0)
            for lag in range(1, tau_max + 1):
                for cause_var in variables:
                    cause = (cause_var, lag)
                    if allowed_edge(cause, effect):
                        edges.append((cause, effect))

        # Instantaneous candidate edges: X(t) -> Y(t), X != Y
        for cause_var in variables:
            for effect_var in variables:
                if cause_var == effect_var:
                    continue
                cause = (cause_var, 0)
                effect = (effect_var, 0)
                if allowed_edge(cause, effect):
                    edges.append((cause, effect))

        return edges

    def _forward_phase(
        self,
        *,
        graph: nx.DiGraph,
        variables: Sequence[str],
        tau_max: int,
        allowed_edge: TemporalAllowedEdge,
        score_fun: TemporalScoreFunction,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        candidates = self._candidate_edges(
            variables=variables,
            tau_max=tau_max,
            allowed_edge=allowed_edge,
        )

        step = 0
        while True:
            best_edge = None
            best_gain = float("-inf")
            scored_edges = []

            for cause, effect in candidates:
                if graph.has_edge(cause, effect):
                    continue

                if not self._can_add_edge(graph, cause, effect):
                    continue

                gain = self._addition_gain(cause, effect, graph, score_fun)
                scored_edges.append(
                    {
                        "from": cause,
                        "to": effect,
                        "gain": float(gain),
                    }
                )

                if self.score_significant(gain) and (best_edge is None or self.gain_is_better(gain, best_gain)):
                    best_edge = (cause, effect)
                    best_gain = float(gain)

            if best_edge is None:
                history.append(
                    {
                        "phase": "forward",
                        "step": step,
                        "action": "stop",
                        "scored_edges": scored_edges,
                    }
                )
                break

            cause, effect = best_edge
            graph.add_edge(cause, effect)

            history.append(
                {
                    "phase": "forward",
                    "step": step,
                    "action": "add",
                    "from": cause,
                    "to": effect,
                    "gain": float(best_gain),
                    "scored_edges": scored_edges,
                }
            )

            step += 1

        return history

    def _backward_phase(
        self,
        *,
        graph: nx.DiGraph,
        score_fun: TemporalScoreFunction,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []

        step = 0
        while True:
            best_edge = None
            best_gain = float("-inf")
            scored_edges = []

            for cause, effect in list(graph.edges):
                gain = self._removal_gain(cause, effect, graph, score_fun)
                scored_edges.append(
                    {
                        "from": cause,
                        "to": effect,
                        "gain": float(gain),
                    }
                )

                if self.score_significant(gain) and (best_edge is None or self.gain_is_better(gain, best_gain)):
                    best_edge = (cause, effect)
                    best_gain = float(gain)

            if best_edge is None:
                history.append(
                    {
                        "phase": "backward",
                        "step": step,
                        "action": "stop",
                        "scored_edges": scored_edges,
                    }
                )
                break

            cause, effect = best_edge
            graph.remove_edge(cause, effect)

            history.append(
                {
                    "phase": "backward",
                    "step": step,
                    "action": "remove",
                    "from": cause,
                    "to": effect,
                    "gain": float(best_gain),
                    "scored_edges": scored_edges,
                }
            )

            step += 1

        return history

    def _addition_gain(
        self,
        cause: TemporalNode,
        effect: TemporalNode,
        graph: nx.DiGraph,
        score_fun: TemporalScoreFunction,
    ) -> float:
        parents = tuple(graph.predecessors(effect))
        old_score = float(score_fun(effect, parents))
        new_score = float(score_fun(effect, parents + (cause,)))
        return float(self.transition_gain(old_score, new_score))

    def _removal_gain(
        self,
        cause: TemporalNode,
        effect: TemporalNode,
        graph: nx.DiGraph,
        score_fun: TemporalScoreFunction,
    ) -> float:
        parents = tuple(graph.predecessors(effect))
        if cause not in parents:
            return float("-inf")

        old_score = float(score_fun(effect, parents))
        new_parents = tuple(p for p in parents if p != cause)
        new_score = float(score_fun(effect, new_parents))
        return float(self.transition_gain(old_score, new_score))

    def _can_add_edge(self, graph: nx.DiGraph, cause: TemporalNode, effect: TemporalNode) -> bool:
        if cause == effect:
            return False

        # Lagged edges cannot create directed cycles in the window graph because
        # they point into lag-0 nodes from earlier lags.
        if cause[1] > 0:
            return True

        # Instantaneous edges must keep the lag-0 subgraph acyclic.
        if cause[1] == 0 and effect[1] == 0:
            graph.add_edge(cause, effect)
            try:
                ok = nx.is_directed_acyclic_graph(self._instantaneous_subgraph(graph))
            finally:
                graph.remove_edge(cause, effect)
            return ok

        return True

    def _instantaneous_subgraph(self, graph: nx.DiGraph) -> nx.DiGraph:
        lag0_nodes = [node for node in graph.nodes if isinstance(node, tuple) and node[1] == 0]
        return graph.subgraph(lag0_nodes).copy()

    def _instantaneous_topological_order(
        self,
        graph: nx.DiGraph,
        variables: Sequence[str],
    ) -> list[str]:
        lag0_nodes = [(v, 0) for v in variables]
        sub = graph.subgraph(lag0_nodes).copy()

        if not nx.is_directed_acyclic_graph(sub):
            return list(variables)

        return [v for v, lag in nx.topological_sort(sub) if lag == 0]


class GraphSearchTemporalTopological:
    def __init__(self, *, scoring):
        self.transition_gain = scoring.transition_gain
        self.score_significant = scoring.score_significant
        self.gain_is_better = scoring.gain_is_better

    def run(
        self,
        *,
        variables: Sequence[str],
        tau_max: int,
        allowed_edge: TemporalAllowedEdge,
        score_fun: TemporalScoreFunction,
    ) -> GraphSearchResult:
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
        return GraphSearchResult(graph=g, topological_order=order, history=history)

    def _addition_gain(
        self,
        cause: TemporalNode,
        effect: TemporalNode,
        graph: nx.DiGraph,
        score_fun: TemporalScoreFunction,
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
        allowed_edge: TemporalAllowedEdge,
        score_fun: TemporalScoreFunction,
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
        allowed_edge: TemporalAllowedEdge,
        score_fun: TemporalScoreFunction,
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
        allowed_edge: TemporalAllowedEdge,
        score_fun: TemporalScoreFunction,
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
        allowed_edge: TemporalAllowedEdge,
        score_fun: TemporalScoreFunction,
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

    def _remove_ingoing_edges(self, *, source: TemporalNode, graph: nx.DiGraph, score_fun: TemporalScoreFunction):
        pruned_edges = []
        meta = []

        parents = list(graph.predecessors(source))
        while parents:
            removed_parent, best_gain, candidates = self._find_removable_edge(
                parents=parents,
                child=source,
                score_fun=score_fun,
            )

            for parent, keep_gain, removable in candidates:
                meta.append(
                    {
                        "from": parent,
                        "to": source,
                        "keep_gain": float(keep_gain),
                        "removable": bool(removable),
                    }
                )
            if removed_parent is None:
                break

            graph.remove_edge(removed_parent, source)
            parents.remove(removed_parent)
            pruned_edges.append(
                {
                    "from": removed_parent,
                    "to": source,
                    "keep_gain": float(best_gain),
                }
            )

        return pruned_edges, meta

    def _find_removable_edge(
        self,
        *,
        parents: list[TemporalNode],
        child: TemporalNode,
        score_fun: TemporalScoreFunction,
    ):
        full_score = float(score_fun(child, tuple(parents)))

        best_parent = None
        weakest_keep_gain = float("inf")
        candidate_stats = []

        for parent in parents:
            reduced_parents = tuple(p for p in parents if p != parent)
            reduced_score = float(score_fun(child, reduced_parents))

            # Improvement from adding this parent back to the reduced parent set.
            keep_gain = float(self.transition_gain(reduced_score, full_score))
            removable = not self.score_significant(keep_gain)

            candidate_stats.append((parent, keep_gain, removable))

            if removable and keep_gain < weakest_keep_gain:
                weakest_keep_gain = keep_gain
                best_parent = parent

        if best_parent is None:
            return None, float("inf"), candidate_stats

        return best_parent, float(weakest_keep_gain), candidate_stats
