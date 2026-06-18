from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import networkx as nx
import pandas as pd

from causalchange import SpaceTime
from causalchange.causal_change import CausalChange
from causalchange.posthoc.temporal import compute_edge_contributions, compute_mechanism_scores
from experiments.common.data_types import (
    PanelDataset,
    PosthocTables,
    SpaceTimeExperimentConfig,
    SpaceTimeExperimentRun,
)
from experiments.common.preprocessing import build_context_dataframe, preprocess_panel_dataset


def _public_changepoint_mode(value: str) -> str:
    if value == "none":
        return "skip"
    if value in {"detect", "fixed"}:
        return value
    raise ValueError(f"Unknown changepoint mode: {value!r}")


def _public_changepoint_scope(changepoint_mode: str, data_mode: str) -> str:
    if changepoint_mode == "skip":
        return "skip"
    # Real-world experiments currently use a common changepoint grid across contexts.
    # Switch to "per-context" here if a future experiment needs context-specific changepoints.
    return "global"


def _public_changepoint_method(changepoint_mode: str) -> str:
    return "pelt" if changepoint_mode == "detect" else "skip"


def _public_clustering_scope(config: SpaceTimeExperimentConfig, data_mode: str) -> str:
    if config.detect_contexts and config.detect_regimes:
        if data_mode != "time-contexts":
            return "regimes"
        return "regimes-contexts"
    if config.detect_contexts:
        if data_mode != "time-contexts":
            return "skip"
        return "contexts"
    if config.detect_regimes:
        return "regimes"
    return "skip"


def _public_clustering_method(config: SpaceTimeExperimentConfig, clustering_scope: str) -> str:
    if clustering_scope == "skip" or config.mechanism_clustering == "skip":
        return "skip"
    if config.mechanism_clustering == "testing":
        return "statistical-testing"
    if config.mechanism_clustering == "edge-strengths":
        return "mechanism-clustering"
    raise ValueError(f"Unknown mechanism_clustering: {config.mechanism_clustering!r}")


def _public_testing_method(clustering_method: str) -> str:
    return "kernel" if clustering_method == "statistical-testing" else "skip"


def dataframe_for_spacetime(
    dataset: PanelDataset,
    *,
    config: SpaceTimeExperimentConfig,
) -> tuple[pd.DataFrame, str]:
    """Return the dataframe and public SpaceTime data_mode string."""
    dataset.validate()

    if dataset.n_contexts() == 1:
        context_id = dataset.context_ids()[0]
        return dataset.panel[context_id].loc[:, list(dataset.variables)].reset_index(drop=True), "time"

    frame = build_context_dataframe(
        dataset.panel,
        context_col=config.context_col,
        variables=dataset.variables,
    )
    return frame, "time-contexts"


def make_spacetime_estimator(
    *,
    data_mode: str,
    config: SpaceTimeExperimentConfig,
) -> SpaceTime:
    """Build the public SpaceTime wrapper used by the real-world experiments."""
    changepoint_mode = _public_changepoint_mode(config.changepoints)
    clustering_scope = _public_clustering_scope(config, data_mode)
    clustering_method = _public_clustering_method(config, clustering_scope)

    return SpaceTime(
        data_mode=data_mode,
        score_type=config.score_type,
        context_col=config.context_col,
        tau_max=config.tau_max,
        changepoint_mode=changepoint_mode,
        changepoint_scope=_public_changepoint_scope(changepoint_mode, data_mode),
        changepoint_method=_public_changepoint_method(changepoint_mode),
        fixed_changepoints=list(config.fixed_changepoints) if changepoint_mode == "fixed" else None,
        clustering_scope=clustering_scope,
        clustering_method=clustering_method,
        testing_method=_public_testing_method(clustering_method),
        d_min=config.d_min,
        max_iter=config.max_iter,
        pelt_penalty=config.pelt_penalty,
        mechanism_test_alpha=config.mechanism_test_alpha,
    )


def fit_spacetime(
    dataset: PanelDataset,
    *,
    config: SpaceTimeExperimentConfig,
    preprocess: bool = True,
    graph_for_posthoc: nx.DiGraph | None = None,
    changepoints_for_posthoc: list[int] | None = None,
    compute_global_scores: bool = True,
    compute_window_scores: bool = True,
) -> SpaceTimeExperimentRun:
    """Preprocess, fit SpaceTime, and compute post-hoc score tables.

    The fitted estimator returns a context-regime grid in ``run.partitions``.
    For edge-strength clustering, each grid cell is represented by pairwise
    edge-strength features under the final graph and changepoints.
    """
    current = dataset

    if preprocess:
        current = preprocess_panel_dataset(
            dataset,
            fill_method=config.fill_method,
            standardize=config.standardize,
        )

    X, data_mode = dataframe_for_spacetime(current, config=config)
    estimator = make_spacetime_estimator(data_mode=data_mode, config=config)
    estimator.fit(X)

    posthoc = compute_posthoc_tables(
        estimator,
        graph=graph_for_posthoc,
        changepoints=changepoints_for_posthoc,
        compute_global_scores=compute_global_scores,
        compute_window_scores=compute_window_scores,
    )

    return SpaceTimeExperimentRun(
        dataset=current,
        config=config,
        estimator=estimator,
        graph=estimator.graph_,
        changepoints=list(estimator.changepoints_ or []),
        partitions=estimator.spacetime_components_,
        posthoc=posthoc,
    )


def compute_posthoc_tables(
    estimator: CausalChange,
    *,
    graph: nx.DiGraph | None = None,
    changepoints: list[int] | None = None,
    compute_global_scores: bool = True,
    compute_window_scores: bool = True,
) -> PosthocTables:
    mechanism_scores_global = None
    mechanism_scores_windows = None
    edge_contributions_global = None
    edge_contributions_windows = None

    engine = estimator.engine_
    if engine is None:
        raise RuntimeError("Estimator has no fitted engine.")

    if compute_global_scores:
        mechanism_scores_global = compute_mechanism_scores(
            engine,
            graph=graph,
            scope="global",
            changepoints=changepoints,
        )
        edge_contributions_global = compute_edge_contributions(
            engine,
            graph=graph,
            scope="global",
            changepoints=changepoints,
        )

    if compute_window_scores:
        mechanism_scores_windows = compute_mechanism_scores(
            engine,
            graph=graph,
            scope="windows",
            changepoints=changepoints,
        )
        edge_contributions_windows = compute_edge_contributions(
            engine,
            graph=graph,
            scope="windows",
            changepoints=changepoints,
        )

    return PosthocTables(
        mechanism_scores_global=mechanism_scores_global,
        mechanism_scores_windows=mechanism_scores_windows,
        edge_contributions_global=edge_contributions_global,
        edge_contributions_windows=edge_contributions_windows,
    )


def fit_spacetime_from_panel(
    panel: Mapping[Hashable, pd.DataFrame],
    *,
    variables: Sequence[str],
    name: str,
    config: SpaceTimeExperimentConfig,
    metadata: Mapping[str, object] | None = None,
    preprocess: bool = True,
) -> SpaceTimeExperimentRun:
    dataset = PanelDataset(
        name=name,
        panel=panel,
        variables=list(variables),
        context_col=config.context_col,
        metadata=dict(metadata or {}),
    )
    return fit_spacetime(dataset, config=config, preprocess=preprocess)
