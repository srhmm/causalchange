from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np

from causalchange.core.protocols import TabularScoringProtocol
from causalchange.core.results import GraphSearchResult
from causalchange.domain.tabular import TabularAllowedEdge, TabularScoreFunction


class GraphSearchTabularGreedy:
    def __init__(self, *, scoring):
        self.scoring = scoring

    def run(self, *args, **kwargs):
        raise NotImplementedError("not implemented yet")


class GraphSearchTabularTopological:
    def __init__(self, *, scoring: TabularScoringProtocol):
        self.transition_gain = scoring.transition_gain
        self.score_significant = scoring.score_significant
        self.gain_is_better = scoring.gain_is_better

    def run(
        self,
        *,
        nodes: Sequence[Any],
        candidates: list[Any],
        allowed_edge: TabularAllowedEdge,
        score_fun: TabularScoreFunction,
    ) -> GraphSearchResult:
        g = nx.DiGraph()
        g.add_nodes_from(nodes)

        candidates = list(candidates)  # copy
        topological_order: list[Any] = []
        history: list[dict[str, Any]] = []

        n_nodes = len(candidates)
        it = 0
        while it < n_nodes:
            source, source_meta = self._next_node_in_topological_order(
                candidates=candidates,
                graph=g,
                allowed_edge=allowed_edge,
                score_fun=score_fun,
            )

            candidates.remove(source)
            topological_order.append(source)

            added_edges, outgoing_scores = self._add_outgoing_edges(
                source=source,
                remaining=candidates,
                graph=g,
                allowed_edge=allowed_edge,
                score_fun=score_fun,
            )
            pruned_edges, incoming_scores = self._remove_ingoing_edges(
                source=source,
                graph=g,
                score_fun=score_fun,
            )

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

        return GraphSearchResult(graph=g, topological_order=topological_order, history=history)

    def _addition_gain(self, cause, effect, graph: nx.DiGraph, score_oracle: TabularScoreFunction) -> float:
        parents = tuple(graph.predecessors(effect))
        old_score = float(score_oracle(effect, parents))
        new_score = float(score_oracle(effect, parents + (cause,)))
        return float(self.transition_gain(old_score, new_score))

    def _improvement_matrix(
        self,
        *,
        candidates: Sequence[Any],
        graph: nx.DiGraph,
        allowed_edge: TabularAllowedEdge,
        score_oracle: TabularScoreFunction,
    ) -> np.ndarray:
        n = len(candidates)
        imp = np.zeros((n, n), dtype=float)
        idx = {node: i for i, node in enumerate(candidates)}

        for cause in candidates:
            for effect in candidates:
                if cause == effect:
                    continue
                if not allowed_edge(cause, effect):
                    continue
                g = self._addition_gain(cause, effect, graph, score_oracle)
                imp[idx[cause], idx[effect]] = float(g)

        return imp

    def _next_node_in_topological_order(
        self,
        *,
        candidates: Sequence[Any],
        graph: nx.DiGraph,
        allowed_edge: TabularAllowedEdge,
        score_fun: TabularScoreFunction,
    ):
        improvement = self._improvement_matrix(
            candidates=candidates,
            graph=graph,
            allowed_edge=allowed_edge,
            score_oracle=score_fun,
        )
        delta = improvement - improvement.T
        np.fill_diagonal(delta, -np.inf)

        incoming_pressure = np.max(delta, axis=0)
        source_idx = int(np.argmin(incoming_pressure))
        source = list(candidates)[source_idx]

        meta = {
            "candidates": list(candidates),
            "improvement_matrix": improvement.tolist(),
            "delta_matrix": delta.tolist(),
            "source_idx": source_idx,
            "incoming_pressure": incoming_pressure.tolist(),
        }
        return source, meta

    def _add_outgoing_edges(
        self,
        *,
        source,
        remaining: Sequence[Any],
        graph: nx.DiGraph,
        allowed_edge: TabularAllowedEdge,
        score_fun: TabularScoreFunction,
    ):
        added_edges = []
        meta = []

        for node in remaining:
            if node == source:
                continue
            if not allowed_edge(source, node):
                continue

            gain = self._addition_gain(source, node, graph, score_fun)
            significant = bool(self.score_significant(gain))

            meta.append(
                {
                    "from": source,
                    "to": node,
                    "gain": float(gain),
                    "significant": significant,
                }
            )

            if significant:
                graph.add_edge(source, node)
                added_edges.append({"from": source, "to": node, "gain": float(gain)})

        return added_edges, meta

    def _remove_ingoing_edges(self, *, source, graph: nx.DiGraph, score_fun: TabularScoreFunction):
        pruned_edges = []
        meta = []

        parents = list(graph.predecessors(source))
        while parents:
            removed_found, removed_parent, best_gain, cand = self._find_removable_edge(
                parents=parents,
                child=source,
                graph=graph,
                score_oracle=score_fun,
            )

            for parent, keep_gain, removable in cand:
                meta.append(
                    {
                        "from": parent,
                        "to": source,
                        "keep_gain": float(keep_gain),
                        "removable": bool(removable),
                    }
                )

            if not removed_found:
                break

            graph.remove_edge(removed_parent, source)
            pruned_edges.append(
                {
                    "from": removed_parent,
                    "to": source,
                    "keep_gain": float(best_gain),
                }
            )
            parents = list(graph.predecessors(source))

        return pruned_edges, meta

    def _find_removable_edge(self, *, parents, child, graph: nx.DiGraph, score_oracle: TabularScoreFunction):
        full_score = float(score_oracle(child, tuple(parents)))

        best_parent = None
        weakest_keep_gain = float("inf")
        candidate_stats = []

        for parent in parents:
            reduced_parents = tuple(p for p in parents if p != parent)
            reduced_score = float(score_oracle(child, reduced_parents))

            # Improvement from adding this parent back to the reduced parent set.
            keep_gain = float(self.transition_gain(reduced_score, full_score))
            removable = not self.score_significant(keep_gain)

            candidate_stats.append((parent, keep_gain, removable))

            if removable and keep_gain < weakest_keep_gain:
                weakest_keep_gain = keep_gain
                best_parent = parent

        if best_parent is None:
            return False, None, float("inf"), candidate_stats

        return True, best_parent, float(weakest_keep_gain), candidate_stats