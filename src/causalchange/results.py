from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import networkx as nx
import numpy as np
import pandas as pd

Node = tuple[str, int]


@dataclass(frozen=True)
class SpaceTimePartitions:
    """
    contexts[target][dataset_id] = context_cluster_id
    regimes[target][regime_id] = regime_cluster_id
    """

    contexts: dict[str, dict[Any, int]]
    regimes: dict[str, dict[int, int]]
    diagnostics: dict[str, Any]


@dataclass
class TemporalResult:
    graph: Any
    changepoints: list[int]
    partitions: SpaceTimePartitions

    topological_order: list[str] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    edge_strengths: dict[tuple[Any, Any], float] = field(default_factory=dict)

    # For ChangepointScope.PER_CONTEXT, changepoints is the union of changepoints_by_context
    changepoints_by_context: dict[Any, list[int]] | None = None
    changepoint_diagnostics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
