from __future__ import annotations

from typing import Any

import pandas as pd

from causalchange.config.cc_config import CausalChangeConfig, ChangepointMode, ChangepointScope
from causalchange.config.cc_types import DataMode
from causalchange.discovery.search_time.base import SpaceTimeResult, SpaceTimeScoring, TimePanel


class SpaceTimeEngine:
    def __init__(
        self,
        *,
        data_mode: DataMode,
        domain,
        scoring: SpaceTimeScoring,
        search,
        changepoint_detection,
        partitioning,
        cfg: CausalChangeConfig,
    ):
        self.data_mode = data_mode
        self.domain = domain
        self.scorer = scoring
        self.search = search
        self.changepoint_detection = changepoint_detection
        self.partitioning = partitioning
        self.cfg = cfg

        self.X0_: pd.DataFrame | None = None
        self.changepoints_: list[int] = []
        self.changepoints_by_context_: dict[Any, list[int]] | None = None
        self.changepoint_diagnostics_: dict[str, Any] = {}
        self.panel_: TimePanel | None = None
        self.partitions_ = None
        self.result_: SpaceTimeResult | None = None

    def fit(self, X: pd.DataFrame) -> SpaceTimeEngine:
        self.panel_ = self._make_panel(X)

        self.scorer.fit_panel(self.panel_)

        X0 = self.panel_.first_dataset()
        self.X0_ = X0

        return self

    def score_edge(self, effect, parents) -> float:
        if self.panel_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        if self.partitions_ is not None:
            return float(
                self.scorer.score_edge_panel(
                    panel=self.panel_,
                    effect=effect,
                    parents=tuple(parents),
                    partitions=self.partitions_,
                )
            )

        if self.X0_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        return float(self.scorer.score_edge(self.X0_, effect, tuple(parents)))

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

    def discover(self) -> SpaceTimeResult:
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
            self.changepoints_ = self.changepoint_detection.detect(
                panel=self.panel_,
                graph=graph,
                scorer=self.scorer,
                variables=variables,
            )
            if self.cfg.spacetime.changepoint_scope == ChangepointScope.PER_CONTEXT:
                self.changepoints_by_context_ = self.changepoint_detection.changepoints_by_context_
            else:
                self.changepoints_by_context_ = None

            self.changepoint_diagnostics_ = self.changepoint_detection.diagnostics_
            self.scorer.set_time_windows(
                n_raw_samples=len(self.panel_.first_dataset()),
                changepoints=self.changepoints_,
            )
            self.partitions_ = self.partitioning.fit_predict(
                panel=self.panel_,
                graph=graph,
                changepoints=self.changepoints_,
            )

            search_result = self.search.run(
                variables=variables,
                tau_max=self.cfg.spacetime.tau_max,
                allowed_edge=self.domain.allowed_edge,
                score_fun=self.score_edge,
            )

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
                }
            )

            if key == previous_key:
                break

            previous_key = key

        if final_search_result is None or self.partitions_ is None:
            raise RuntimeError("SpaceTime discovery failed to produce a result.")

        result = SpaceTimeResult(
            graph=final_search_result.graph,
            topological_order=final_search_result.topological_order,
            changepoints=self.changepoints_,
            partitions=self.partitions_,
            changepoints_by_context=self.changepoints_by_context_,
            changepoint_diagnostics=self.changepoint_diagnostics_,
        )

        self.result_ = result
        return result

    def _make_panel(self, X: pd.DataFrame) -> TimePanel:
        if self.data_mode == DataMode.TIME:
            X0 = self.domain.prepare_X(X)
            return TimePanel(
                datasets={0: X0},
                variables=self.domain.variables(X0),
                context_col=None,
            )

        if self.data_mode == DataMode.TIME_CONTEXTS:
            return self._make_context_panel(X)

        raise ValueError(f"SpaceTimeEngine expects temporal data, got {self.data_mode=}")

    def _make_context_panel(self, X: pd.DataFrame) -> TimePanel:
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

        return TimePanel(
            datasets=datasets,
            variables=variables,
            context_col=context_col,
        )
