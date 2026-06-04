"""Temporal discovery engine, coordinates changepoints, scm clustering, graph search, and scoring."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from causalchange.core.protocols import (
    ChangepointDetectionProtocol,
    MechanismClusteringProtocol,
    TemporalDomainProtocol,
    TemporalScoringProtocol,
    TemporalSearchProtocol,
)
from causalchange.core.results import ChangepointResult, SCMClusteringResult, TemporalResult
from causalchange.core.types import (
    ChangepointMode,
    ChangepointScope,
    DataMode,
    MechanismClusteringScope,
    PostprocessingMode,
)
from causalchange.domain.temporal import TimeGrid
from causalchange.engines.base import BaseDiscoveryEngine


class TemporalDiscoveryEngine(
    BaseDiscoveryEngine[TemporalDomainProtocol, TemporalScoringProtocol, TemporalSearchProtocol]
):
    """shows lower-level control flow for temporal causal discovery."""

    def __init__(
        self,
        *,
        data_mode: DataMode,
        domain: TemporalDomainProtocol,
        scoring: TemporalScoringProtocol,
        search: TemporalSearchProtocol,
        changepoint_detection: ChangepointDetectionProtocol,
        scm_clustering: MechanismClusteringProtocol,
        clustering_scope: MechanismClusteringScope,
        context_col: str,
        tau_max: int,
        changepoint_mode: ChangepointMode,
        changepoint_scope: ChangepointScope,
        max_iter: int,
        diagnostics: dict[str, Any] | None = None,
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
    ):
        super().__init__(
            data_mode=data_mode,
            domain=domain,
            scoring=scoring,
            search=search,
            postprocessing_mode=postprocessing_mode,
        )
        self.changepoint_mode = changepoint_mode
        self.changepoint_detection = changepoint_detection
        self.changepoint_scope = changepoint_scope
        self.scm_clustering = scm_clustering
        self.clustering_scope = clustering_scope

        self.context_col = context_col
        self.tau_max = tau_max
        self.max_iter = max_iter
        self.diagnostics = dict(diagnostics or {})

        self.X0_: pd.DataFrame | None = None
        self.changepoints_: list[int] = []
        self.changepoints_by_context_: dict[Any, list[int]] | None = None
        self.changepoint_diagnostics_: dict[str, Any] = {}
        self.panel_: TimeGrid | None = None
        self.partitions_: SCMClusteringResult | None = None
        self.result_: TemporalResult | None = None
        self._score_cache: dict[tuple, float] = {}

    def fit(self, X: pd.DataFrame) -> TemporalDiscoveryEngine:
        self.panel_ = self._make_panel(X)

        self.scoring.fit_panel(self.panel_)

        X0 = self.panel_.first_dataset()
        self.X0_ = X0

        return self

    def local_score(self, effect, parents) -> float:
        if self.panel_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        parents = tuple(sorted(parents, key=repr))
        cache_key = (
            effect,
            parents,
            tuple(self.changepoints_),
            self._partition_key(),
        )

        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        if self.partitions_ is not None:
            value = float(
                self.scoring.local_score_grid(
                    panel=self.panel_,
                    effect=effect,
                    parents=parents,
                    partitions=self.partitions_,
                )
            )
        else:
            if self.X0_ is None:
                raise RuntimeError("Engine not fitted. Call fit() first.")

            value = float(self.scoring.local_score(self.X0_, effect, parents))

        self._score_cache[cache_key] = value
        return value

    def _run_discovery(self) -> TemporalResult:
        if self.X0_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        variables = self.domain.variables(self.X0_)
        needs_iteration = self.changepoint_mode == ChangepointMode.DETECT or self.clustering_scope in [
            MechanismClusteringScope.REGIMES,
            MechanismClusteringScope.CONTEXTS,
            MechanismClusteringScope.REGIMES_CONTEXTS,
        ]

        max_iter = self.max_iter if needs_iteration else 1
        graph = None
        previous_key = None
        all_history: list[dict[str, Any]] = []

        final_search_result = None

        for iteration in range(max_iter):
            t0 = perf_counter()

            t_cp0 = perf_counter()
            self.changepoints_ = self.changepoint_detection.detect(
                time_grid=self.panel_,
                graph=graph,
                scorer=self.scoring,
                variables=variables,
            )
            t_cp = perf_counter() - t_cp0

            if self.changepoint_scope == ChangepointScope.PER_CONTEXT:
                self.changepoints_by_context_ = self.changepoint_detection.changepoints_by_context_
            else:
                self.changepoints_by_context_ = None

            self.changepoint_diagnostics_ = self.changepoint_detection.diagnostics_

            self.scoring.set_time_windows(
                n_raw_samples=len(self.panel_.first_dataset()),
                changepoints=self.changepoints_,
            )

            t_part0 = perf_counter()
            self.partitions_ = self.scm_clustering.fit_predict(
                panel=self.panel_,
                graph=graph,
                changepoints=self.changepoints_,
                changepoints_by_context=self.changepoints_by_context_,
                scorer=self.scoring,
            )
            t_part = perf_counter() - t_part0

            self._score_cache = {}

            t_search0 = perf_counter()
            search_result = self.search.run(
                variables=variables,
                tau_max=self.tau_max,
                allowed_edge=self.domain.allowed_edge,
                score_fun=self.local_score,
            )
            t_search = perf_counter() - t_search0

            iteration_time = perf_counter() - t0

            final_search_result = search_result
            graph = search_result.graph

            key = (
                tuple(self.changepoints_),
                tuple(sorted(graph.edges())),
                self._partition_key(),
            )

            all_history.append(
                {
                    "iteration": iteration,
                    "changepoints": list(self.changepoints_),
                    "n_edges": graph.number_of_edges(),
                    "search_history": search_result.history,
                    "partition_diagnostics": self.partitions_.diagnostics,
                    "timing": {
                        "changepoints": t_cp,
                        "partitioning": t_part,
                        "search": t_search,
                        "iteration_total": iteration_time,
                        "score_cache_size": len(self._score_cache),
                    },
                }
            )

            if key == previous_key:
                break

            previous_key = key

        if final_search_result is None:
            raise RuntimeError("Temporal discovery failed to produce a graph search result.")

        # Final mechanism clustering is post-processing under the final graph and
        # final changepoints. This is the result exposed as result.grid_clusters.
        t_final_part0 = perf_counter()
        self.partitions_ = self.scm_clustering.fit_predict(
            panel=self.panel_,
            graph=final_search_result.graph,
            changepoints=self.changepoints_,
            changepoints_by_context=self.changepoints_by_context_,
            scorer=self.scoring,
        )
        final_partition_time = perf_counter() - t_final_part0
        self._score_cache = {}

        if all_history:
            all_history[-1]["final_partition_diagnostics"] = self.partitions_.diagnostics
            all_history[-1]["timing"]["final_partitioning"] = final_partition_time

        changepoint_result = ChangepointResult(
            changepoints=self.changepoints_,
            changepoints_by_context=self.changepoints_by_context_,
            diagnostics=self.changepoint_diagnostics_,
        )

        result = TemporalResult(
            graph_search=final_search_result,
            changepoint=changepoint_result,
            mechanism_clustering=self.partitions_,
            history=all_history,
            diagnostics={
                "n_iterations": len(all_history),
                "score_cache_size": len(self._score_cache),
            },
        )
        self.result_ = result
        return result

    def _partition_key(self) -> tuple | None:
        if self.partitions_ is None:
            return None

        cell_key = tuple(
            (
                target,
                tuple(
                    sorted(
                        (
                            repr(cell.dataset_id),
                            cell.interval_id,
                            cluster_id,
                        )
                        for cell, cluster_id in mapping.items()
                    )
                ),
            )
            for target, mapping in sorted(self.partitions_.cell_clusters.items())
        )

        interval_key = tuple(
            (
                repr(dataset_id),
                tuple(intervals),
            )
            for dataset_id, intervals in sorted(
                self.partitions_.intervals_by_context.items(),
                key=lambda item: repr(item[0]),
            )
        )

        return cell_key, interval_key

    def _make_panel(self, X: pd.DataFrame) -> TimeGrid:
        if self.data_mode == DataMode.TIME:
            X0 = self.domain.prepare_X(X)
            return TimeGrid(
                datasets={0: X0},
                variables=self.domain.variables(X0),
                context_col=None,
            )

        if self.data_mode == DataMode.TIME_CONTEXTS:
            return self._make_context_panel(X)

        raise ValueError(f"TemporalDiscoveryEngine expects temporal data, got {self.data_mode=}")

    def _make_context_panel(self, X: pd.DataFrame) -> TimeGrid:
        context_col = self.context_col

        if context_col not in X.columns:
            raise ValueError(
                f"data_mode={self.data_mode.value} requires context column "
                f"{context_col!r}, but it was not found in X.columns."
            )

        datasets: dict[Any, pd.DataFrame] = {}

        for context_id, X_context in X.groupby(context_col, sort=False):
            X_context = X_context.drop(columns=[context_col])
            X_context = self.domain.prepare_X(X_context)
            X_context = X_context.reset_index(drop=True)
            datasets[context_id] = X_context

        if not datasets:
            raise ValueError("No contexts found.")

        first_id = next(iter(datasets))
        variables = self.domain.variables(datasets[first_id])

        for context_id, X_context in datasets.items():
            current_variables = self.domain.variables(X_context)
            if current_variables != variables:
                raise ValueError(
                    "All time-series contexts must have the same variables in the same order. "
                    f"Expected {variables}, got {current_variables} for context {context_id!r}."
                )

        return TimeGrid(
            datasets=datasets,
            variables=variables,
            context_col=context_col,
        )
