from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from causalchange.domain.temporal import TemporalNode


@dataclass(frozen=True)
class SpaceTimeGridClusters:
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
    grid_clusters: SpaceTimeGridClusters

    topological_order: list[str] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    edge_strengths: dict[tuple[Any, Any], float] = field(default_factory=dict)

    # For ChangepointScope.PER_CONTEXT, changepoints is the union of changepoints_by_context
    changepoints_by_context: dict[Any, list[int]] | None = None
    changepoint_diagnostics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MechanismScoreRecord:
    scope: str
    target: str
    effect: TemporalNode
    parents: tuple[TemporalNode, ...]
    n_parents: int
    score: float
    dataset_id: Any | None = None
    interval_id: int | None = None
    interval_start: int | None = None
    interval_stop: int | None = None
    n_samples: int | None = None


@dataclass(frozen=True)
class EdgeContributionRecord:
    scope: str
    parent: TemporalNode
    target: str
    effect: TemporalNode
    full_parent_set: tuple[TemporalNode, ...]
    n_parents: int
    full_score: float
    reduced_score: float
    raw_gain: float
    positive_gain: float
    dataset_id: Any | None = None
    interval_id: int | None = None
    interval_start: int | None = None
    interval_stop: int | None = None
    n_samples: int | None = None


@dataclass(frozen=True)
class ContextCombinationResult:
    total: float
    diagnostics: dict[str, Any]


@dataclass
class GraphSearchTemporalResult:
    graph: nx.DiGraph
    topological_order: list[str]
    history: list[dict[str, Any]]


@dataclass
class GraphSearchTabularResult:
    graph: nx.DiGraph
    topological_order: list[Any]
    history: list[dict[str, Any]]
    edge_strengths: dict[tuple[Any, Any], float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
