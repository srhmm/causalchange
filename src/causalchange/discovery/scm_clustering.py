from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from causalchange.config.causal_change_config import CausalChangeConfigTemporal
from causalchange.core.results import GridCell, SCMClusteringResult
from causalchange.core.types import MechanismClusteringMethod, StatisticalTestingMethod
from causalchange.domain.temporal import (
    TemporalNode,
    TimeGrid,
    util_changepoints_to_intervals,
)
from causalchange.scoring.statistical_tests import SCMEqualityTestKCI
from causalchange.scoring.tabular import SCMScoreTabular


class SCMClustering(ABC):
    """Abstract base for SCM mechanism clustering."""

    @abstractmethod
    def fit_predict(self, *args: Any, **kwargs: Any) -> SCMClusteringResult: ...


class TabularSCMClustering(SCMClustering):
    """Placeholder for future tabular/mixture SCM clustering."""

    def fit_predict(self, *args: Any, **kwargs: Any) -> SCMClusteringResult:
        raise NotImplementedError("Tabular SCM clustering is not implemented yet.")


class BaseTemporalSCMClustering(SCMClustering):
    """Shared logic for temporal SCM clustering over context x interval grid cells."""

    def __init__(self, cfg: CausalChangeConfigTemporal):
        self.cfg = cfg

    @abstractmethod
    def fit_predict(
        self,
        X: pd.DataFrame | None = None,
        *,
        panel: TimeGrid | None = None,
        graph=None,
        changepoints: list[int] | None = None,
        changepoints_by_context: dict[Any, list[int]] | None = None,
        scorer=None,
    ) -> SCMClusteringResult: ...

    def _coerce_panel(
        self,
        X: pd.DataFrame | None,
        panel: TimeGrid | None,
    ) -> TimeGrid:
        if panel is not None:
            return panel

        if X is None:
            raise ValueError("Either X or panel must be provided.")

        return TimeGrid(
            datasets={0: X},
            variables=[str(c) for c in X.columns],
            context_col=None,
        )

    def _intervals_by_context(
            self,
            *,
            panel: TimeGrid,
            changepoints: list[int] | None,
            changepoints_by_context: dict[Any, list[int]] | None,
    ) -> dict[Any, list[tuple[int, int]]]:
        if changepoints_by_context is not None:
            return {
                dataset_id: util_changepoints_to_intervals(
                    len(panel.datasets[dataset_id]),
                    list(changepoints_by_context.get(dataset_id, [])),
                )
                for dataset_id in panel.dataset_ids
            }

        if changepoints is not None:
            global_changepoints = list(changepoints)
            return {
                dataset_id: util_changepoints_to_intervals(
                    len(panel.datasets[dataset_id]),
                    global_changepoints,
                )
                for dataset_id in panel.dataset_ids
            }

        if not self.cfg.clustering_scope.detects_regimes():
            return {
                dataset_id: [(0, len(X_context))]
                for dataset_id, X_context in panel.datasets.items()
            }
        return {
            dataset_id: util_changepoints_to_intervals(
                len(panel.datasets[dataset_id]),
                [],
            )
            for dataset_id in panel.dataset_ids
        }

    def _grid_cells(
        self,
        intervals_by_context: dict[Any, list[tuple[int, int]]],
    ) -> list[GridCell]:
        return [
            GridCell(dataset_id=dataset_id, interval_id=interval_id)
            for dataset_id, intervals in intervals_by_context.items()
            for interval_id in range(len(intervals))
        ]

    def _trivial_result(
        self,
        *,
        panel: TimeGrid,
        intervals_by_context: dict[Any, list[tuple[int, int]]],
        mode: str,
        extra_diagnostics: dict[str, Any] | None = None,
    ) -> SCMClusteringResult:
        cells = self._grid_cells(intervals_by_context)

        cell_clusters = {target: {cell: 0 for cell in cells} for target in panel.variables}

        diagnostics: dict[str, Any] = {
            "mode": mode,
            "n_cells": len(cells),
            "n_contexts": panel.n_contexts,
            "intervals_by_context": intervals_by_context,
        }

        if extra_diagnostics:
            diagnostics.update(extra_diagnostics)

        return SCMClusteringResult(
            cell_clusters=cell_clusters,
            intervals_by_context=intervals_by_context,
            diagnostics=diagnostics,
        )

    def _parents_for_target(self, graph, target: str) -> list[TemporalNode]:
        if graph is None:
            return []

        effect = (target, 0)

        if effect not in graph:
            return []

        parents: list[TemporalNode] = []

        for parent in graph.predecessors(effect):
            if not isinstance(parent, tuple) or len(parent) != 2:
                continue

            parent_var, lag = parent
            parents.append((str(parent_var), int(lag)))

        return sorted(parents, key=repr)

    def _parent_cols(self, parents: list[TemporalNode]) -> list[str]:
        return [f"parent_{idx}" for idx, _ in enumerate(parents)]

    def _sample_for_cell(
        self,
        *,
        panel: TimeGrid,
        cell: GridCell,
        intervals_by_context: dict[Any, list[tuple[int, int]]],
        target: str,
        parents: list[TemporalNode],
    ) -> pd.DataFrame:
        X = panel.datasets[cell.dataset_id]
        interval = intervals_by_context[cell.dataset_id][cell.interval_id]

        return self._sample_for_interval(
            X=X,
            target=target,
            parents=parents,
            interval=interval,
        )

    def _sample_for_interval(
        self,
        *,
        X: pd.DataFrame,
        target: str,
        parents: list[TemporalNode],
        interval: tuple[int, int],
    ) -> pd.DataFrame:
        start, stop = interval
        max_lag = max([lag for _, lag in parents], default=0)
        first_t = max(start, max_lag, self.cfg.tau_max)

        rows: list[dict[str, float]] = []

        for t in range(first_t, stop):
            row: dict[str, float] = {}

            for idx, (parent_var, lag) in enumerate(parents):
                row[f"parent_{idx}"] = float(X[str(parent_var)].iloc[t - int(lag)])

            row["target"] = float(X[str(target)].iloc[t])
            rows.append(row)

        columns = [*self._parent_cols(parents), "target"]
        return pd.DataFrame(rows, columns=columns)

    def _pair_key(self, a: Any, b: Any) -> frozenset[Any]:
        if a == b:
            raise ValueError("Pair keys require two distinct nodes.")

        return frozenset((a, b))

    def _cluster_from_pairwise_tests(
        self,
        *,
        nodes: list[Any],
        same_pairs: set[frozenset[Any]],
        different_pairs: set[frozenset[Any]],
        pvalues: dict[frozenset[Any], float],
    ) -> dict[Any, int]:
        clusters: list[set[Any]] = [{node} for node in nodes]

        while True:
            best_pair: tuple[int, int] | None = None
            best_strength = float("-inf")

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    left = clusters[i]
                    right = clusters[j]

                    if not self._can_merge_clusters(
                        left=left,
                        right=right,
                        same_pairs=same_pairs,
                        different_pairs=different_pairs,
                    ):
                        continue

                    strength = self._merge_strength(
                        left=left,
                        right=right,
                        pvalues=pvalues,
                    )

                    if strength > best_strength:
                        best_strength = strength
                        best_pair = (i, j)

            if best_pair is None:
                break

            i, j = best_pair
            clusters[i] = clusters[i] | clusters[j]
            del clusters[j]

        labels: dict[Any, int] = {}

        for label, cluster in enumerate(clusters):
            for node in cluster:
                labels[node] = label

        return labels

    def _can_merge_clusters(
        self,
        *,
        left: set[Any],
        right: set[Any],
        same_pairs: set[frozenset[Any]],
        different_pairs: set[frozenset[Any]],
    ) -> bool:
        for a in left:
            for b in right:
                pair = self._pair_key(a, b)

                if pair in different_pairs:
                    return False

                if pair not in same_pairs:
                    return False

        return True

    def _merge_strength(
        self,
        *,
        left: set[Any],
        right: set[Any],
        pvalues: dict[frozenset[Any], float],
    ) -> float:
        values = [float(pvalues.get(self._pair_key(a, b), 0.0)) for a in left for b in right]

        if not values:
            return 0.0

        return float(sum(values) / len(values))


class TemporalSCMSkipClustering(BaseTemporalSCMClustering):
    """Skip temporal SCM clustering."""

    def fit_predict(
        self,
        X: pd.DataFrame | None = None,
        *,
        panel: TimeGrid | None = None,
        graph=None,
        changepoints: list[int] | None = None,
        changepoints_by_context: dict[Any, list[int]] | None = None,
        scorer=None,
    ) -> SCMClusteringResult:
        panel = self._coerce_panel(X, panel)

        intervals_by_context = self._intervals_by_context(
            panel=panel,
            changepoints=changepoints,
            changepoints_by_context=changepoints_by_context,
        )

        return self._trivial_result(
            panel=panel,
            intervals_by_context=intervals_by_context,
            mode="skip",
        )


class TemporalSCMPairwiseTesting(BaseTemporalSCMClustering):
    """Cluster temporal SCM grid cells by pairwise mechanism equality tests."""

    def __init__(self, cfg: CausalChangeConfigTemporal):
        super().__init__(cfg)

        self.equality_test = SCMEqualityTestKCI(
            alpha=cfg.mechanism_test_alpha,
            min_samples=max(5, min(cfg.d_min, 10)),
        )

    def fit_predict(
        self,
        X: pd.DataFrame | None = None,
        *,
        panel: TimeGrid | None = None,
        graph=None,
        changepoints: list[int] | None = None,
        changepoints_by_context: dict[Any, list[int]] | None = None,
        scorer=None,
    ) -> SCMClusteringResult:
        panel = self._coerce_panel(X, panel)

        intervals_by_context = self._intervals_by_context(
            panel=panel,
            changepoints=changepoints,
            changepoints_by_context=changepoints_by_context,
        )

        cells = self._grid_cells(intervals_by_context)

        diagnostics: dict[str, Any] = {
            "mode": "pairwise_testing",
            "changepoints": list(changepoints or []),
            "intervals_by_context": intervals_by_context,
            "n_cells": len(cells),
            "n_contexts": panel.n_contexts,
            "tests": [],
        }

        if self.cfg.testing_method == StatisticalTestingMethod.SKIP:
            return self._trivial_result(
                panel=panel,
                intervals_by_context=intervals_by_context,
                mode="testing_skipped",
                extra_diagnostics=diagnostics,
            )

        cell_clusters: dict[str, dict[GridCell, int]] = {}

        for target in panel.variables:
            parents = self._parents_for_target(graph, target)
            parent_cols = self._parent_cols(parents)

            same_pairs: set[frozenset[Any]] = set()
            different_pairs: set[frozenset[Any]] = set()
            pvalues: dict[frozenset[Any], float] = {}

            for i, cell_a in enumerate(cells):
                for cell_b in cells[i + 1 :]:
                    sample_a = self._sample_for_cell(
                        panel=panel,
                        cell=cell_a,
                        intervals_by_context=intervals_by_context,
                        target=target,
                        parents=parents,
                    )
                    sample_b = self._sample_for_cell(
                        panel=panel,
                        cell=cell_b,
                        intervals_by_context=intervals_by_context,
                        target=target,
                        parents=parents,
                    )

                    pair = self._pair_key(cell_a, cell_b)

                    if sample_a.empty or sample_b.empty:
                        same = False
                        pvalue = 0.0
                        method = "empty_sample"
                    else:
                        result = self.equality_test.same_mechanism(
                            sample_a=sample_a,
                            sample_b=sample_b,
                            target_col="target",
                            parent_cols=parent_cols,
                        )
                        same = bool(result.same)
                        pvalue = float(result.pvalue)
                        method = result.method

                    pvalues[pair] = pvalue

                    if same:
                        same_pairs.add(pair)
                    else:
                        different_pairs.add(pair)

                    diagnostics["tests"].append(
                        {
                            "target": target,
                            "cell_a": cell_a,
                            "cell_b": cell_b,
                            "pvalue": pvalue,
                            "same": same,
                            "method": method,
                        }
                    )

            cell_clusters[target] = self._cluster_from_pairwise_tests(
                nodes=cells,
                same_pairs=same_pairs,
                different_pairs=different_pairs,
                pvalues=pvalues,
            )

        return SCMClusteringResult(
            cell_clusters=cell_clusters,
            intervals_by_context=intervals_by_context,
            diagnostics=diagnostics,
        )


class TemporalSCMEdgeStrengthClustering(BaseTemporalSCMClustering):
    """Cluster temporal SCM grid cells by edge-strength feature vectors."""

    def fit_predict(
        self,
        X: pd.DataFrame | None = None,
        *,
        panel: TimeGrid | None = None,
        graph=None,
        changepoints: list[int] | None = None,
        changepoints_by_context: dict[Any, list[int]] | None = None,
        scorer=None,
    ) -> SCMClusteringResult:
        if scorer is None:
            raise ValueError("scorer is required for edge-strength clustering.")

        panel = self._coerce_panel(X, panel)

        intervals_by_context = self._intervals_by_context(
            panel=panel,
            changepoints=changepoints,
            changepoints_by_context=changepoints_by_context,
        )

        cells = self._grid_cells(intervals_by_context)

        diagnostics: dict[str, Any] = {
            "mode": "edge_strength_clustering",
            "changepoints": list(changepoints or []),
            "intervals_by_context": intervals_by_context,
            "n_cells": len(cells),
            "n_contexts": panel.n_contexts,
            "features": {},
            "failures": [],
        }

        if graph is None or graph.number_of_edges() == 0:
            return self._trivial_result(
                panel=panel,
                intervals_by_context=intervals_by_context,
                mode="edge_strength_clustering_no_graph",
                extra_diagnostics=diagnostics,
            )

        cell_clusters: dict[str, dict[GridCell, int]] = {}
        tabular_scorer = SCMScoreTabular(self.cfg)

        for target in panel.variables:
            parents = self._parents_for_target(graph, target)

            if not parents:
                cell_clusters[target] = {cell: 0 for cell in cells}
                diagnostics["features"][target] = {
                    "n_features": 0,
                    "reason": "no_parents",
                }
                continue

            feature_rows = []

            for cell in cells:
                sample = self._sample_for_cell(
                    panel=panel,
                    cell=cell,
                    intervals_by_context=intervals_by_context,
                    target=target,
                    parents=parents,
                )

                features = self._edge_strength_features(
                    sample=sample,
                    parents=parents,
                    tabular_scorer=tabular_scorer,
                    transition_gain=scorer.transition_gain,
                    diagnostics=diagnostics,
                    target=target,
                    cell=cell,
                )

                feature_rows.append(features)

            X_features = np.asarray(feature_rows, dtype=float)

            labels = self._cluster_feature_matrix(
                items=cells,
                X_features=X_features,
            )

            cell_clusters[target] = labels

            diagnostics["features"][target] = {
                "n_features": int(X_features.shape[1]) if X_features.ndim == 2 else 0,
                "n_cells": len(cells),
                "n_clusters": len(set(labels.values())),
                "parents": parents,
            }

        return SCMClusteringResult(
            cell_clusters=cell_clusters,
            intervals_by_context=intervals_by_context,
            diagnostics=diagnostics,
        )

    def _edge_strength_features(
        self,
        *,
        sample: pd.DataFrame,
        parents: list[TemporalNode],
        tabular_scorer: SCMScoreTabular,
        transition_gain,
        diagnostics: dict[str, Any],
        target: str,
        cell: GridCell,
    ) -> list[float]:
        parent_cols = self._parent_cols(parents)

        if sample.empty:
            return [0.0 for _ in parent_cols]

        try:
            full_score = float(tabular_scorer.local_score(sample, "target", parent_cols))
        except Exception as exc:
            diagnostics["failures"].append(
                {
                    "target": target,
                    "cell": cell,
                    "stage": "full_score",
                    "error": repr(exc),
                }
            )
            return [0.0 for _ in parent_cols]

        features: list[float] = []

        for parent_col in parent_cols:
            reduced_cols = [col for col in parent_cols if col != parent_col]

            try:
                reduced_score = float(tabular_scorer.local_score(sample, "target", reduced_cols))
                gain = float(transition_gain(reduced_score, full_score))
                features.append(max(gain, 0.0))
            except Exception as exc:
                diagnostics["failures"].append(
                    {
                        "target": target,
                        "cell": cell,
                        "stage": "reduced_score",
                        "parent_col": parent_col,
                        "error": repr(exc),
                    }
                )
                features.append(0.0)

        return features

    def _cluster_feature_matrix(
        self,
        *,
        items: list[GridCell],
        X_features: np.ndarray,
    ) -> dict[GridCell, int]:
        if len(items) == 0:
            return {}

        if len(items) == 1:
            return {items[0]: 0}

        if X_features.size == 0 or X_features.shape[1] == 0:
            return {item: 0 for item in items}

        X_features = np.nan_to_num(
            X_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if np.allclose(X_features, X_features[0]):
            return {item: 0 for item in items}

        X_scaled = StandardScaler().fit_transform(X_features)

        n_clusters = getattr(self.cfg, "mechanism_clustering_n_clusters", None)
        distance_threshold = getattr(self.cfg, "mechanism_clustering_distance_threshold", None)

        if n_clusters is None and distance_threshold is None:
            n_clusters = min(3, len(items))

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
        )

        labels = model.fit_predict(X_scaled)

        return {item: int(label) for item, label in zip(items, labels, strict=True)}


class TemporalSCMClustering(SCMClustering):
    """Dispatcher for temporal SCM clustering methods."""

    def __init__(self, cfg: CausalChangeConfigTemporal):
        if cfg.clustering_method == MechanismClusteringMethod.SKIP:
            self.impl: BaseTemporalSCMClustering = TemporalSCMSkipClustering(cfg)
        elif cfg.clustering_method == MechanismClusteringMethod.TESTING:
            self.impl = TemporalSCMPairwiseTesting(cfg)
        elif cfg.clustering_method == MechanismClusteringMethod.CLUSTERING:
            self.impl = TemporalSCMEdgeStrengthClustering(cfg)
        else:
            raise ValueError(f"Unsupported clustering_method: {cfg.clustering_method}")

    def fit_predict(
        self,
        X: pd.DataFrame | None = None,
        *,
        panel: TimeGrid | None = None,
        graph=None,
        changepoints: list[int] | None = None,
        changepoints_by_context: dict[Any, list[int]] | None = None,
        scorer=None,
    ) -> SCMClusteringResult:
        return self.impl.fit_predict(
            X=X,
            panel=panel,
            graph=graph,
            changepoints=changepoints,
            changepoints_by_context=changepoints_by_context,
            scorer=scorer,
        )


# Backward-compatible alias for older tests/imports.
SpaceTimeClustering = TemporalSCMClustering
