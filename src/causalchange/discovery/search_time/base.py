from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class SpaceTimePartitions:
    contexts: dict[Any, int]
    regimes: dict[Any, int]
    diagnostics: dict[str, Any]


@dataclass
class SpaceTimeResult:
    graph: nx.DiGraph
    changepoints: list[int]
    partitions: SpaceTimePartitions
    history: list[dict[str, Any]]
