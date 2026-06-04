from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import networkx as nx
import pandas as pd

from causalchange.causal_change import CausalChange
from causalchange.core.types import (
    ChangepointMethod,
    ChangepointMode,
    ChangepointScope,
    DataMode,
    GPType,
    GraphSearch,
    MechanismClusteringMethod,
    MechanismClusteringScope,
    ScoreType,
    StatisticalTestingMethod,
    TabularContextMethod,
)
from causalchange.posthoc.temporal import compute_edge_contributions, compute_mechanism_scores
from experiments.common.data_types import (
    PanelDataset,
    PosthocTables,
    SpaceTimeExperimentConfig,
    SpaceTimeExperimentRun,
)
from experiments.common.preprocessing import build_context_dataframe, preprocess_panel_dataset


def resolve_score_type(value: str) -> ScoreType | GPType:
    """Resolve experiment score names to CausalChange score enums."""
    if value in (GPType.EXACT.value, GPType.FOURIER.value):
        return GPType(value)
    return ScoreType(value)


def resolve_changepoint_mode(value: str) -> ChangepointMode:
    if value == "none":
        return ChangepointMode.SKIP
    if value == "detect":
        return ChangepointMode.DETECT
    if value == "fixed":
        return ChangepointMode.ORACLE
    raise ValueError(f"Unknown changepoint mode: {value!r}")


def changepoint_method_for_mode(mode: ChangepointMode) -> ChangepointMethod:
    return ChangepointMethod.PELT if mode == ChangepointMode.DETECT else ChangepointMethod.SKIP


def changepoint_scope_for_mode(mode: ChangepointMode) -> ChangepointScope:
    return ChangepointScope.GLOBAL if mode != ChangepointMode.SKIP else ChangepointScope.SKIP


def clustering_scope_from_config(config: SpaceTimeExperimentConfig) -> MechanismClusteringScope:
    if config.detect_contexts and config.detect_regimes:
        return MechanismClusteringScope.REGIMES_CONTEXTS
    if config.detect_contexts:
        return MechanismClusteringScope.CONTEXTS
    if config.detect_regimes:
        return MechanismClusteringScope.REGIMES
    return MechanismClusteringScope.SKIP


def dataframe_for_spacetime(
    dataset: PanelDataset,
    *,
    config: SpaceTimeExperimentConfig,
) -> tuple[pd.DataFrame, DataMode]:
    """Return the dataframe and data mode expected by CausalChange."""
    dataset.validate()

    if dataset.n_contexts() == 1:
        context_id = dataset.context_ids()[0]
        return dataset.panel[context_id].loc[:, list(dataset.variables)].reset_index(drop=True), DataMode.TIME

    frame = build_context_dataframe(
        dataset.panel,
        context_col=config.context_col,
        variables=dataset.variables,
    )
    return frame, DataMode.TIME_CONTEXTS


def make_spacetime_estimator(
    *,
    data_mode: DataMode,
    config: SpaceTimeExperimentConfig,
) -> CausalChange:
    fixed_changepoints = list(config.fixed_changepoints) if config.changepoints == "fixed" else None

    changepoint_mode = resolve_changepoint_mode(config.changepoints)
    clustering_scope = clustering_scope_from_config(config)
    clustering_method = (
        MechanismClusteringMethod.TESTING
        if clustering_scope != MechanismClusteringScope.SKIP
        else MechanismClusteringMethod.SKIP
    )
    testing_method = (
        StatisticalTestingMethod.KERNEL
        if clustering_method == MechanismClusteringMethod.TESTING
        else StatisticalTestingMethod.SKIP
    )

    return CausalChange(
        data_mode=data_mode,
        graph_search=GraphSearch.GLOBE,
        score_type=resolve_score_type(config.score_type),
        context_method=TabularContextMethod.SKIP,
        context_col=config.context_col if data_mode.is_context() else None,
        tau_max=config.tau_max,
        changepoint_mode=changepoint_mode,
        changepoint_scope=changepoint_scope_for_mode(changepoint_mode),
        changepoint_method=changepoint_method_for_mode(changepoint_mode),
        fixed_changepoints=fixed_changepoints,
        clustering_scope=clustering_scope,
        clustering_method=clustering_method,
        testing_method=testing_method,
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
    """Preprocess, fit SpaceTime, and compute post-hoc tables.

    ``graph_for_posthoc`` and ``changepoints_for_posthoc`` allow fixed-artifact
    post-hoc analysis after a discovery run. This keeps graph/changepoint choices
    explicit and reproducible.
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
        partitions=estimator.partitions_,
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

    if compute_global_scores:
        engine = estimator.engine_
        if engine is None:
            raise RuntimeError("Estimator has no fitted engine.")
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
        engine = estimator.engine_
        if engine is None:
            raise RuntimeError("Estimator has no fitted engine.")
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
