from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx


def graph_from_index_links(
    links: Mapping[int, Sequence[tuple[tuple[int, int], float, Any]]],
    *,
    variables: Sequence[str],
    tau_max: int,
) -> nx.DiGraph:
    """Build a temporal graph from index-based reference links.

    ``links[target_idx]`` contains entries ``((parent_idx, lag), weight, meta)``.
    The returned graph uses nodes ``(variable, lag)`` and edges into ``(target, 0)``.
    """
    graph = nx.DiGraph()
    for var in variables:
        for lag in range(int(tau_max) + 1):
            graph.add_node((str(var), lag))

    for target_idx, parent_entries in links.items():
        effect = (str(variables[int(target_idx)]), 0)
        for (parent_idx, lag), _weight, _meta in parent_entries:
            parent = (str(variables[int(parent_idx)]), int(lag))
            graph.add_edge(parent, effect)

    return graph
