from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass
class GlobeSearchResult:
    graph: nx.DiGraph
    history: list[dict[str, Any]]


class GlobeSearch: ...