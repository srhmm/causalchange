from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from causalchange.domain.temporal import TemporalNode

# %% Components/algos


@dataclass
class GraphSearchResult:
    """returned by a graph search algo"""

    graph: nx.DiGraph
    topological_order: list[Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangepointResult:
    """returned by changepoint detection algo"""

    changepoints: list[int] = field(default_factory=list)
    changepoints_by_context: dict[Any, list[int]] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextCombinationResult:
    """returned by context combination algo"""

    total: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GridCell:
    """One cell in the context × time-interval grid"""

    dataset_id: Any
    interval_id: int


@dataclass
class SCMClusteringResult:
    """
    Mechanism clusters over context × time-interval cells

    cell_clusters[target][GridCell(dataset_id, interval_id)] = mechanism_cluster_id
    """

    cell_clusters: dict[str, dict[GridCell, int]]
    intervals_by_context: dict[Any, list[tuple[int, int]]]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def cluster_for(
        self,
        *,
        target: str,
        dataset_id: Any,
        interval_id: int,
    ) -> int:
        return self.cell_clusters[target][GridCell(dataset_id, interval_id)]

    def cells_for_cluster(
        self,
        *,
        target: str,
        cluster_id: int,
    ) -> list[GridCell]:
        return [cell for cell, label in self.cell_clusters.get(target, {}).items() if label == cluster_id]


@dataclass
class CMMTargetMixtureResult:
    """Mixture components for one final local mechanism target | parents."""

    target: Any
    parents: tuple[Any, ...]
    labels: list[int]
    responsibilities: list[list[float]]
    component_weights: list[float] = field(default_factory=list)
    score: float | None = None
    n_components: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CMMMixtureResult:
    """Mixture components learned by CMM under the final graph."""

    target_components: dict[Any, CMMTargetMixtureResult] = field(default_factory=dict)
    global_labels: list[int] | None = None
    global_responsibilities: list[list[float]] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def labels_for(self, target: Any) -> list[int]:
        return self.target_components[target].labels

    def responsibilities_for(self, target: Any) -> list[list[float]]:
        return self.target_components[target].responsibilities

    def parents_for(self, target: Any) -> tuple[Any, ...]:
        return self.target_components[target].parents


@dataclass
class PostProcessingResult:
    """Returned by postprocessing steps."""

    edge_strengths: dict[tuple[Any, Any], float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


# %% records
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


# %% CausalChange


@dataclass
class CausalChangeResult:
    """Returned by a discovery engine."""

    graph_search: GraphSearchResult
    postprocessing: PostProcessingResult = field(default_factory=PostProcessingResult)
    history: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def graph(self) -> nx.DiGraph:
        return self.graph_search.graph

    @property
    def topological_order(self) -> list[Any] | None:
        return self.graph_search.topological_order

    @property
    def edge_strengths(self) -> dict[tuple[Any, Any], float]:
        return self.postprocessing.edge_strengths


# %% Engines
@dataclass
class TabularResult(CausalChangeResult):
    """returned by the tabular engine"""

    mechanism_mixture: CMMMixtureResult | None = None

    @property
    def mixture_components(self) -> CMMMixtureResult | None:
        return self.mechanism_mixture


@dataclass
class TemporalResult(CausalChangeResult):
    """returned by the temporal engine"""

    changepoint: ChangepointResult = field(default_factory=ChangepointResult)
    mechanism_clustering: SCMClusteringResult | None = None

    @property
    def changepoints(self) -> list[int]:
        return self.changepoint.changepoints

    @property
    def changepoints_by_context(self) -> dict[Any, list[int]] | None:
        return self.changepoint.changepoints_by_context

    @property
    def changepoint_diagnostics(self) -> dict[str, Any]:
        return self.changepoint.diagnostics

    @property
    def grid_clusters(self) -> SCMClusteringResult | None:
        return self.mechanism_clustering
