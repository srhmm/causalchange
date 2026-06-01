from __future__ import annotations

from typing import Any

import pandas as pd

from causalchange.config.cc_config import PartitioningMethod, SpaceTimeConfig
from causalchange.discovery.search_time.base import Node, SpaceTimePartitions, TimePanel
from causalchange.discovery.search_time.changepoints import changepoints_to_intervals
from causalchange.discovery.search_time.mechanism_tests import KCIMechanismEqualityTest


class SpaceTimePartitioning:
    """
    Partition contexts and time regimes for SPACETIME

        contexts[target][dataset_id] = context_cluster_id
        regimes[target][regime_id] = regime_cluster_id
    """

    def __init__(self, cfg: SpaceTimeConfig):
        self.cfg = cfg
        self.equality_test = KCIMechanismEqualityTest(
            alpha=cfg.mechanism_test_alpha,
            min_samples=max(5, min(cfg.d_min, 10)),
        )

    def fit_predict(
        self,
        X: pd.DataFrame | None = None,
        *,
        panel: TimePanel | None = None,
        graph=None,
        changepoints: list[int] | None = None,
    ) -> SpaceTimePartitions:
        if panel is None:
            if X is None:
                raise ValueError("Either X or panel must be provided.")

            panel = TimePanel(
                datasets={0: X},
                variables=[str(c) for c in X.columns],
                context_col=None,
            )

        changepoints = list(changepoints or [])
        n_samples = len(panel.first_dataset())
        intervals = changepoints_to_intervals(n_samples, changepoints)

        contexts = self._initial_context_partitions(panel)
        regimes = self._initial_regime_partitions(
            variables=panel.variables,
            n_regimes=len(intervals),
        )

        diagnostics: dict[str, Any] = {
            "mode": "initial",
            "detect_contexts": self.cfg.detect_contexts,
            "detect_regimes": self.cfg.detect_regimes,
            "changepoints": changepoints,
            "intervals": intervals,
            "n_contexts": panel.n_contexts,
            "n_regimes": len(intervals),
            "tests": [],
        }

        if self.cfg.partitioning_method == PartitioningMethod.NONE:
            diagnostics["mode"] = "none"
            return SpaceTimePartitions(
                contexts=contexts,
                regimes=regimes,
                diagnostics=diagnostics,
            )

        if self.cfg.detect_regimes:
            regimes, regime_tests = self._partition_regimes(
                panel=panel,
                graph=graph,
                intervals=intervals,
            )
            diagnostics["tests"].extend(regime_tests)

        if self.cfg.detect_contexts:
            contexts, context_tests = self._partition_contexts(
                panel=panel,
                graph=graph,
                intervals=intervals,
            )
            diagnostics["tests"].extend(context_tests)

        diagnostics["mode"] = "kernel" if diagnostics["tests"] else "initial"

        return SpaceTimePartitions(
            contexts=contexts,
            regimes=regimes,
            diagnostics=diagnostics,
        )

    def _initial_context_partitions(
        self,
        panel: TimePanel,
    ) -> dict[str, dict[Any, int]]:
        if not self.cfg.detect_contexts:
            return {target: {dataset_id: 0 for dataset_id in panel.dataset_ids} for target in panel.variables}

        return {
            target: {dataset_id: context_idx for context_idx, dataset_id in enumerate(panel.dataset_ids)}
            for target in panel.variables
        }

    def _initial_regime_partitions(
        self,
        *,
        variables: list[str],
        n_regimes: int,
    ) -> dict[str, dict[int, int]]:
        return {target: {regime_id: regime_id for regime_id in range(n_regimes)} for target in variables}

    def _partition_regimes(
        self,
        *,
        panel: TimePanel,
        graph,
        intervals: list[tuple[int, int]],
    ) -> tuple[dict[str, dict[int, int]], list[dict[str, Any]]]:
        regimes: dict[str, dict[int, int]] = {}
        tests_list: list[dict[str, Any]] = []

        regime_ids = list(range(len(intervals)))

        for target in panel.variables:
            parents = self._parents_for_target(graph, target)
            parent_cols = self._parent_cols(parents)

            same_pairs: set[frozenset[Any]] = set()
            different_pairs: set[frozenset[Any]] = set()
            pvalues: dict[frozenset[Any], float] = {}

            for r1 in regime_ids:
                for r2 in regime_ids:
                    if r2 <= r1:
                        continue

                    sample_1 = self._pooled_interval_sample(
                        panel=panel,
                        target=target,
                        parents=parents,
                        interval=intervals[r1],
                    )
                    sample_2 = self._pooled_interval_sample(
                        panel=panel,
                        target=target,
                        parents=parents,
                        interval=intervals[r2],
                    )

                    result = self.equality_test.same_mechanism(
                        sample_a=sample_1,
                        sample_b=sample_2,
                        target_col="target",
                        parent_cols=parent_cols,
                    )

                    pair = self._pair_key(r1, r2)
                    pvalues[pair] = result.pvalue

                    if result.same:
                        same_pairs.add(pair)
                    else:
                        different_pairs.add(pair)

                    tests_list.append(
                        {
                            "kind": "regime",
                            "target": target,
                            "regime_a": r1,
                            "regime_b": r2,
                            "pvalue": result.pvalue,
                            "same": result.same,
                            "method": result.method,
                        }
                    )

            regimes[target] = self._cluster_from_pairwise_tests(
                nodes=regime_ids,
                same_pairs=same_pairs,
                different_pairs=different_pairs,
                pvalues=pvalues,
            )

        return regimes, tests_list

    def _partition_contexts(
        self,
        *,
        panel: TimePanel,
        graph,
        intervals: list[tuple[int, int]],
    ) -> tuple[dict[str, dict[Any, int]], list[dict[str, Any]]]:
        contexts: dict[str, dict[Any, int]] = {}
        tests: list[dict[str, Any]] = []

        dataset_ids = panel.dataset_ids

        for target in panel.variables:
            parents = self._parents_for_target(graph, target)
            parent_cols = self._parent_cols(parents)

            same_pairs: set[frozenset[Any]] = set()
            different_pairs: set[frozenset[Any]] = set()
            pvalues: dict[frozenset[Any], float] = {}

            for i, ctx_a in enumerate(dataset_ids):
                for ctx_b in dataset_ids[i + 1 :]:
                    pair_results = []

                    for regime_id, interval in enumerate(intervals):
                        sample_a = self._sample_for_interval(
                            X=panel.datasets[ctx_a],
                            target=target,
                            parents=parents,
                            interval=interval,
                        )
                        sample_b = self._sample_for_interval(
                            X=panel.datasets[ctx_b],
                            target=target,
                            parents=parents,
                            interval=interval,
                        )

                        result = self.equality_test.same_mechanism(
                            sample_a=sample_a,
                            sample_b=sample_b,
                            target_col="target",
                            parent_cols=parent_cols,
                        )

                        pair_results.append(result)

                        tests.append(
                            {
                                "kind": "context",
                                "target": target,
                                "context_a": ctx_a,
                                "context_b": ctx_b,
                                "regime": regime_id,
                                "pvalue": result.pvalue,
                                "same": result.same,
                                "method": result.method,
                            }
                        )

                    pair = self._pair_key(ctx_a, ctx_b)

                    # Contexts are merged only if they are compatible in every regime.
                    # A single detected difference is treated as a hard cannot-link.
                    if pair_results and all(result.same for result in pair_results):
                        same_pairs.add(pair)
                        pvalues[pair] = min(result.pvalue for result in pair_results)
                    else:
                        different_pairs.add(pair)
                        pvalues[pair] = min(
                            (result.pvalue for result in pair_results),
                            default=0.0,
                        )

            contexts[target] = self._cluster_from_pairwise_tests(
                nodes=dataset_ids,
                same_pairs=same_pairs,
                different_pairs=different_pairs,
                pvalues=pvalues,
            )

        return contexts, tests

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
        """
        Cluster nodes using complete-link equality and hard cannot-link constraints.

        A merge is allowed only if every cross-pair between the two clusters was
        judged equal, and no cross-pair was judged different.

        This avoids transitive merges of the form:
            A same B, B same C, but A different C.
        """
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
        values = []

        for a in left:
            for b in right:
                values.append(float(pvalues.get(self._pair_key(a, b), 0.0)))

        if not values:
            return 0.0

        return float(sum(values) / len(values))

    def _pooled_interval_sample(
        self,
        *,
        panel: TimePanel,
        target: str,
        parents: list[Node],
        interval: tuple[int, int],
    ) -> pd.DataFrame:
        samples = [
            self._sample_for_interval(
                X=X_context,
                target=target,
                parents=parents,
                interval=interval,
            )
            for X_context in panel.datasets.values()
        ]

        samples = [sample for sample in samples if not sample.empty]

        if not samples:
            return pd.DataFrame(columns=[*self._parent_cols(parents), "target"])

        return pd.concat(samples, axis=0, ignore_index=True)

    def _sample_for_interval(
        self,
        *,
        X: pd.DataFrame,
        target: str,
        parents: list[Node],
        interval: tuple[int, int],
    ) -> pd.DataFrame:
        start, stop = interval
        max_lag = max([lag for _, lag in parents], default=0)
        first_t = max(start, max_lag, self.cfg.tau_max)

        rows = []

        for t in range(first_t, stop):
            row: dict[str, float] = {}

            for idx, (parent_var, lag) in enumerate(parents):
                row[f"parent_{idx}"] = float(X[parent_var].iloc[t - lag])

            row["target"] = float(X[target].iloc[t])
            rows.append(row)

        columns = [*self._parent_cols(parents), "target"]
        return pd.DataFrame(rows, columns=columns)

    def _parents_for_target(self, graph, target: str) -> list[Node]:
        if graph is None:
            return []

        effect = (target, 0)

        if effect not in graph:
            return []

        parents = []

        for parent in graph.predecessors(effect):
            if not isinstance(parent, tuple) or len(parent) != 2:
                continue

            parent_var, lag = parent
            parents.append((str(parent_var), int(lag)))

        return parents

    def _parent_cols(self, parents: list[Node]) -> list[str]:
        return [f"parent_{idx}" for idx, _ in enumerate(parents)]
