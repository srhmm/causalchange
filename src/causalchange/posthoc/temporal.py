from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import pandas as pd

from results import Node, SpaceTimePartitions
 


if TYPE_CHECKING:
    from engines.temporal import TemporalDiscoveryEngine

ScoreScope = Literal["global", "windows"]


@dataclass(frozen=True)
class MechanismScoreRecord:
    scope: str
    target: str
    effect: Node
    parents: tuple[Node, ...]
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
    parent: Node
    target: str
    effect: Node
    full_parent_set: tuple[Node, ...]
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


def compute_mechanism_scores(
    engine: TemporalDiscoveryEngine,
    *,
    graph: nx.DiGraph | None = None,
    scope: ScoreScope = "global",
    changepoints: list[int] | None = None,
) -> pd.DataFrame:
    graph = _resolve_graph(engine, graph)

    if scope == "global":
        records = _mechanism_scores_global(engine, graph)
    elif scope == "windows":
        records = _mechanism_scores_windows(engine, graph, changepoints=changepoints)
    else:
        raise ValueError(f"Unknown score scope: {scope!r}")

    return pd.DataFrame([record.__dict__ for record in records])


def compute_edge_contributions(
    engine: TemporalDiscoveryEngine,
    *,
    graph: nx.DiGraph | None = None,
    scope: ScoreScope = "global",
    changepoints: list[int] | None = None,
) -> pd.DataFrame:
    graph = _resolve_graph(engine, graph)

    if scope == "global":
        records = _edge_contributions_global(engine, graph)
    elif scope == "windows":
        records = _edge_contributions_windows(engine, graph, changepoints=changepoints)
    else:
        raise ValueError(f"Unknown score scope: {scope!r}")

    return pd.DataFrame([record.__dict__ for record in records])


def fixed_partition_for_graph(
    *,
    graph: nx.DiGraph,
    dataset_ids: Iterable[Any],
    n_intervals: int,
) -> SpaceTimePartitions:
    """Create singleton/no-change partitions for a fixed graph.

    Useful for post-hoc scoring from saved graph/changepoints when you do not want
    to rerun context/regime partitioning.
    """
    targets = sorted({str(effect[0]) for effect in _effect_nodes(graph)})

    return SpaceTimePartitions(
        contexts={target: {dataset_id: 0 for dataset_id in dataset_ids} for target in targets},
        regimes={target: {interval_id: interval_id for interval_id in range(n_intervals)} for target in targets},
        diagnostics={"mode": "fixed_posthoc"},
    )


def changepoints_to_intervals(
    *,
    n_samples: int,
    changepoints: list[int],
) -> list[tuple[int, int]]:
    cps = sorted(int(cp) for cp in changepoints if 0 < int(cp) < n_samples)
    boundaries = [0, *cps, int(n_samples)]

    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(len(boundaries) - 1)]


def _resolve_graph(
    engine: TemporalDiscoveryEngine,
    graph: nx.DiGraph | None,
) -> nx.DiGraph:
    if graph is not None:
        return graph

    if engine.result_ is not None:
        return engine.result_.graph

    raise RuntimeError("No graph provided and engine has no result_. Fit/discover first or pass graph=...")


def _effect_nodes(graph: nx.DiGraph) -> list[Node]:
    nodes: list[Node] = []

    for node in graph.nodes():
        if _is_temporal_node(node) and int(node[1]) == 0:
            nodes.append((str(node[0]), int(node[1])))

    return sorted(nodes, key=repr)


def _parents_for_effect(
    graph: nx.DiGraph,
    effect: Node,
) -> tuple[Node, ...]:
    parents: list[Node] = []

    for parent in graph.predecessors(effect):
        if not _is_temporal_node(parent):
            continue
        parents.append((str(parent[0]), int(parent[1])))

    return tuple(sorted(parents, key=repr))


def _is_temporal_node(node: Any) -> bool:
    return isinstance(node, tuple) and len(node) == 2


def _mechanism_scores_global(
    engine: TemporalDiscoveryEngine,
    graph: nx.DiGraph,
) -> list[MechanismScoreRecord]:
    records: list[MechanismScoreRecord] = []

    for effect in _effect_nodes(graph):
        parents = _parents_for_effect(graph, effect)
        score = float(engine.score_edge(effect, parents))

        records.append(
            MechanismScoreRecord(
                scope="global",
                target=str(effect[0]),
                effect=effect,
                parents=parents,
                n_parents=len(parents),
                score=score,
            )
        )

    return records


def _edge_contributions_global(
    engine: TemporalDiscoveryEngine,
    graph: nx.DiGraph,
) -> list[EdgeContributionRecord]:
    records: list[EdgeContributionRecord] = []

    for effect in _effect_nodes(graph):
        parents = _parents_for_effect(graph, effect)
        full_score = float(engine.score_edge(effect, parents))

        for parent in parents:
            reduced_parents = tuple(p for p in parents if p != parent)
            reduced_score = float(engine.score_edge(effect, reduced_parents))
            raw_gain = _score_gain(engine, reduced_score=reduced_score, full_score=full_score)

            records.append(
                EdgeContributionRecord(
                    scope="global",
                    parent=parent,
                    target=str(effect[0]),
                    effect=effect,
                    full_parent_set=parents,
                    n_parents=len(parents),
                    full_score=full_score,
                    reduced_score=reduced_score,
                    raw_gain=raw_gain,
                    positive_gain=max(raw_gain, 0.0),
                )
            )

    return records


def _mechanism_scores_windows(
    engine: TemporalDiscoveryEngine,
    graph: nx.DiGraph,
    *,
    changepoints: list[int] | None,
) -> list[MechanismScoreRecord]:
    panel = _require_panel(engine)
    cps = list(engine.changepoints_ if changepoints is None else changepoints)
    n_samples = len(panel.first_dataset())
    intervals = changepoints_to_intervals(n_samples=n_samples, changepoints=cps)

    records: list[MechanismScoreRecord] = []

    for dataset_id, X_context in panel.datasets.items():
        for interval_id, (start, stop) in enumerate(intervals):
            X_window = _windowed_dataframe(
                X_context,
                start=start,
                stop=stop,
                tau_max=engine.cfg.spacetime.tau_max,
            )

            for effect in _effect_nodes(graph):
                parents = _parents_for_effect(graph, effect)
                score = float(engine.scorer.score_edge(X_window, effect, parents))

                records.append(
                    MechanismScoreRecord(
                        scope="windows",
                        dataset_id=dataset_id,
                        interval_id=interval_id,
                        interval_start=start,
                        interval_stop=stop,
                        n_samples=_effective_n_samples(X_window, engine.cfg.spacetime.tau_max),
                        target=str(effect[0]),
                        effect=effect,
                        parents=parents,
                        n_parents=len(parents),
                        score=score,
                    )
                )

    return records


def _edge_contributions_windows(
    engine: TemporalDiscoveryEngine,
    graph: nx.DiGraph,
    *,
    changepoints: list[int] | None,
) -> list[EdgeContributionRecord]:
    panel = _require_panel(engine)
    cps = list(engine.changepoints_ if changepoints is None else changepoints)
    n_samples = len(panel.first_dataset())
    intervals = changepoints_to_intervals(n_samples=n_samples, changepoints=cps)

    records: list[EdgeContributionRecord] = []

    for dataset_id, X_context in panel.datasets.items():
        for interval_id, (start, stop) in enumerate(intervals):
            X_window = _windowed_dataframe(
                X_context,
                start=start,
                stop=stop,
                tau_max=engine.cfg.spacetime.tau_max,
            )
            n_eff = _effective_n_samples(X_window, engine.cfg.spacetime.tau_max)

            for effect in _effect_nodes(graph):
                parents = _parents_for_effect(graph, effect)
                full_score = float(engine.scorer.score_edge(X_window, effect, parents))

                for parent in parents:
                    reduced_parents = tuple(p for p in parents if p != parent)
                    reduced_score = float(engine.scorer.score_edge(X_window, effect, reduced_parents))
                    raw_gain = _score_gain(
                        engine,
                        reduced_score=reduced_score,
                        full_score=full_score,
                    )

                    records.append(
                        EdgeContributionRecord(
                            scope="windows",
                            dataset_id=dataset_id,
                            interval_id=interval_id,
                            interval_start=start,
                            interval_stop=stop,
                            n_samples=n_eff,
                            parent=parent,
                            target=str(effect[0]),
                            effect=effect,
                            full_parent_set=parents,
                            n_parents=len(parents),
                            full_score=full_score,
                            reduced_score=reduced_score,
                            raw_gain=raw_gain,
                            positive_gain=max(raw_gain, 0.0),
                        )
                    )

    return records


def _require_panel(engine: TemporalDiscoveryEngine):
    if engine.panel_ is None:
        raise RuntimeError("Engine has no panel_. Call fit(X) first.")
    return engine.panel_


def _windowed_dataframe(
    X: pd.DataFrame,
    *,
    start: int,
    stop: int,
    tau_max: int,
) -> pd.DataFrame:
    """Return a slice suitable for temporal scoring.

    We include up to tau_max rows before the interval so lagged parents for the
    first scored row inside the interval are available.
    """
    prefix_start = max(0, int(start) - int(tau_max))
    return X.iloc[prefix_start : int(stop)].reset_index(drop=True)


def _effective_n_samples(
    X: pd.DataFrame,
    tau_max: int,
) -> int:
    return max(0, int(len(X)) - int(tau_max))


def _score_gain(
    engine: TemporalDiscoveryEngine,
    *,
    reduced_score: float,
    full_score: float,
) -> float:
    """Return the gain from adding the parent"""
    try:
        return float(engine.scorer.transition_gain(reduced_score, full_score))
    except Exception:
        return float(reduced_score - full_score)
