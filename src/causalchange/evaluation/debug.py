from __future__ import annotations

from typing import Any

import networkx as nx


class TruthOracle:
    def __init__(self, true_graph: nx.DiGraph | None = None):
        self.true_graph = true_graph

    def edge_label(self, u: Any, v: Any) -> str:
        if self.true_graph is None:
            return ""

        if self.true_graph.has_edge(u, v):
            return "causal"

        if self.true_graph.has_edge(v, u):
            return "rev"

        return "spurious"
