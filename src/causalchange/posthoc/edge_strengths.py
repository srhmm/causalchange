from __future__ import annotations

from collections.abc import Callable
from typing import Any

import networkx as nx

from causalchange.core.results import PostProcessingResult


def compute_edge_strengths(
    graph: nx.DiGraph,
    *,
    local_score: Callable[[Any, tuple[Any, ...]], float],
    transition_gain: Callable[[float, float], float],
) -> dict[tuple[Any, Any], float]:
    """compute pair-wise edge strengths relative to a given graph

    For each parent -> effect, the strength is the score gain from adding
    parent to the discovered parent set of effect
    """

    strengths: dict[tuple[Any, Any], float] = {}

    for parent, effect in graph.edges():
        parents = tuple(graph.predecessors(effect))

        score_with = float(local_score(effect, parents))
        score_without = float(
            local_score(
                effect,
                tuple(p for p in parents if p != parent),
            )
        )

        strengths[(parent, effect)] = float(transition_gain(score_without, score_with))

    return strengths


def compute_postprocessing_result(
    graph: nx.DiGraph,
    *,
    local_score: Callable[[Any, tuple[Any, ...]], float],
    transition_gain: Callable[[float, float], float],
) -> PostProcessingResult:
    edge_strengths = compute_edge_strengths(
        graph,
        local_score=local_score,
        transition_gain=transition_gain,
    )

    return PostProcessingResult(
        edge_strengths=edge_strengths,
        diagnostics={
            "edge_strengths": {
                "n_edges": len(edge_strengths),
            }
        },
    )
