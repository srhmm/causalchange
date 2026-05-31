from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

Node = tuple[str, int]
ScoreFunction = Callable[[Node, tuple[Node, ...]], float]
AllowedEdge = Callable[[Node, Node], bool]


@dataclass
class TemporalGlobeSearchResult:
    graph: nx.DiGraph
    topological_order: list[str]
    history: list[dict[str, Any]]


class TemporalGlobeSearch:
    """
    Edge-greedy search for a window causal graph.

    Lagged edges are always time-ordered. Instantaneous lag-0 edges are only
    added if they keep the lag-0 subgraph acyclic.
    """

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
    ) -> TemporalGlobeSearchResult:
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

        return TemporalGlobeSearchResult(
            graph=graph,
            topological_order=topological_order,
            history=history,
        )

    def _candidate_edges(
        self,
        *,
        variables: Sequence[str],
        tau_max: int,
        allowed_edge: AllowedEdge,
    ) -> list[tuple[Node, Node]]:
        edges: list[tuple[Node, Node]] = []

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
        allowed_edge: AllowedEdge,
        score_fun: ScoreFunction,
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

                if self.score_significant(gain) and (best_edge is None or self.score_is_better(gain, best_gain)):
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
        score_fun: ScoreFunction,
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

                if self.score_significant(gain) and (best_edge is None or self.score_is_better(gain, best_gain)):
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
        cause: Node,
        effect: Node,
        graph: nx.DiGraph,
        score_fun: ScoreFunction,
    ) -> float:
        parents = tuple(graph.predecessors(effect))
        old_score = float(score_fun(effect, parents))
        new_score = float(score_fun(effect, parents + (cause,)))
        return float(self.transition_gain(old_score, new_score))

    def _removal_gain(
        self,
        cause: Node,
        effect: Node,
        graph: nx.DiGraph,
        score_fun: ScoreFunction,
    ) -> float:
        parents = tuple(graph.predecessors(effect))
        if cause not in parents:
            return float("-inf")

        old_score = float(score_fun(effect, parents))
        new_parents = tuple(p for p in parents if p != cause)
        new_score = float(score_fun(effect, new_parents))
        return float(self.transition_gain(old_score, new_score))

    def _can_add_edge(self, graph: nx.DiGraph, cause: Node, effect: Node) -> bool:
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
