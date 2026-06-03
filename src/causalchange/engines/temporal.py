from __future__ import annotations

import pandas as pd
from time import perf_counter
from typing import Any

from causalchange.config.causal_change_config import CausalChangeConfigTabular, ChangepointMode, ChangepointScope, DataMode
from causalchange.results import TemporalResult
from causalchange.domain.temporal import TimeGrid
from causalchange.posthoc.temporal import compute_edge_contributions, compute_mechanism_scores
from causalchange.scoring.temporal import SCMScoreTemporal


class TemporalDiscoveryEngine:
    """ shows lower-level control flow for temporal causal discovery. """
    def __init__(
        self,
        *,
        data_mode: DataMode,
        domain,
        scoring: SCMScoreTemporal,
        search,
        changepoint_detection,
        scm_clustering,
        cfg: CausalChangeConfigTabular,
    ):
        self.data_mode = data_mode
        self.domain = domain
        self.scorer = scoring
        self.search = search
        self.changepoint_detection = changepoint_detection
        self.partitioning = scm_clustering
        self.cfg = cfg

        self.X0_: pd.DataFrame | None = None
        self.changepoints_: list[int] = []
        self.changepoints_by_context_: dict[Any, list[int]] | None = None
        self.changepoint_diagnostics_: dict[str, Any] = {}
        self.panel_: TimeGrid | None = None
        self.partitions_ = None
        self.result_: TemporalResult | None = None
        self._score_cache: dict[tuple, float] = {}

    def fit(self, X: pd.DataFrame) -> TemporalDiscoveryEngine:
        self.panel_ = self._make_panel(X)

        self.scorer.fit_panel(self.panel_)

        X0 = self.panel_.first_dataset()
        self.X0_ = X0

        return self

    def score_edge(self, effect, parents) -> float:
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
                self.scorer.score_edge_panel(
                    panel=self.panel_,
                    effect=effect,
                    parents=parents,
                    partitions=self.partitions_,
                )
            )
        else:
            if self.X0_ is None:
                raise RuntimeError("Engine not fitted. Call fit() first.")

            value = float(self.scorer.score_edge(self.X0_, effect, parents))

        self._score_cache[cache_key] = value
        return value

    def _partition_key(self) -> tuple | None:
        if self.partitions_ is None:
            return None

        context_key = tuple(
            (target, tuple(sorted(mapping.items()))) for target, mapping in sorted(self.partitions_.contexts.items())
        )

        regime_key = tuple(
            (target, tuple(sorted(mapping.items()))) for target, mapping in sorted(self.partitions_.regimes.items())
        )

        return context_key, regime_key

    def discover(self) -> TemporalResult:
        if self.X0_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        assert self.cfg.spacetime is not None

        variables = self.domain.variables(self.X0_)
        needs_iteration = (
            self.cfg.spacetime.changepoints == ChangepointMode.DETECT
            or self.cfg.spacetime.detect_contexts
            or self.cfg.spacetime.detect_regimes
        )

        max_iter = self.cfg.spacetime.max_iter if needs_iteration else 1
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
                scorer=self.scorer,
                variables=variables,
            )
            t_cp = perf_counter() - t_cp0
            if self.cfg.spacetime.changepoint_scope == ChangepointScope.PER_CONTEXT:
                self.changepoints_by_context_ = self.changepoint_detection.changepoints_by_context_
            else:
                self.changepoints_by_context_ = None

            self.changepoint_diagnostics_ = self.changepoint_detection.diagnostics_

            self.scorer.set_time_windows(
                n_raw_samples=len(self.panel_.first_dataset()),
                changepoints=self.changepoints_,
            )

            t_part0 = perf_counter()

            self.partitions_ = self.partitioning.fit_predict(
                panel=self.panel_,
                graph=graph,
                changepoints=self.changepoints_,
            )
            t_part = perf_counter() - t_part0

            self._score_cache = {}
            t_search0 = perf_counter()
            search_result = self.search.run(
                variables=variables,
                tau_max=self.cfg.spacetime.tau_max,
                allowed_edge=self.domain.allowed_edge,
                score_fun=self.score_edge,
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

        if final_search_result is None or self.partitions_ is None:
            raise RuntimeError("SpaceTime discovery failed to produce a result.")

        edge_strengths = self._compute_edge_strengths(final_search_result.graph)
        result = TemporalResult(
            graph=final_search_result.graph,
            topological_order=final_search_result.topological_order,
            changepoints=self.changepoints_,
            partitions=self.partitions_,
            changepoints_by_context=self.changepoints_by_context_,
            changepoint_diagnostics=self.changepoint_diagnostics_,
            edge_strengths=edge_strengths,
            diagnostics={
                "data_mode": self.data_mode.value,
                "graph_search": self.cfg.graph_search.value,
                "score_type": str(self.cfg.score_type),
            },
        )

        self.result_ = result
        return result

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
        context_col = self.cfg.context_col

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

    def _compute_edge_strengths(self, graph) -> dict[tuple[Any, Any], float]:
        strengths: dict[tuple[Any, Any], float] = {}

        for edge in graph.edges():
            parent, effect = edge

            parents = list(graph.predecessors(effect))
            score_with = self.score_edge(effect, tuple(parents))

            parents_without = [p for p in parents if p != parent]
            score_without = self.score_edge(effect, tuple(parents_without))

            # MDL scores are lower-is-better, so positive means the edge helps.
            strengths[edge] = float(score_without - score_with)

        return strengths

    def mechanism_scores(
        self,
        *,
        graph=None,
        scope="global",
        changepoints: list[int] | None = None,
    ):
        return compute_mechanism_scores(
            self,
            graph=graph,
            scope=scope,
            changepoints=changepoints,
        )

    def edge_contributions(
        self,
        *,
        graph=None,
        scope="global",
        changepoints: list[int] | None = None,
    ):
        return compute_edge_contributions(
            self,
            graph=graph,
            scope=scope,
            changepoints=changepoints,
        )
