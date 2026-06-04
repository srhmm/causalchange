from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.common.data_types import SpaceTimeExperimentConfig
from experiments.common.paths import data_dir
from experiments.common.reference_graphs import graph_from_index_links

RIVER_VARIABLES = ["tavg", "prec", "Qobs"]
RIVER_ALL_VARIABLES = [
    "tavg",
    "tmin",
    "tmax",
    "prec",
    "rad",
    "Qobs",
    "QsimGlobal",
    "snowpackGlobal",
    "SWC005Global",
    "SMGlobal",
    "ETGlobal",
]
RIVER_NODES = ["T", "P", "Q"]

RIVER_MAIN_YEAR = "2010"
RIVER_N_DAYS = 365
RIVER_MISSING_THRESHOLD = 20
RIVER_MONTH_NAMES = ["Jan", "Fb", "Mr", "Ap", "My", "Jun", "Jl", "Au", "Sp", "Oc", "Nv", "Dc"]

RIVER_RESULT_LINKS = {
    0: [((0, 0), 1, None)],
    1: [((1, 1), 1, None)],
    2: [((1, 1), 1, None)],
}


@dataclass(frozen=True)
class RiverLoadConfig:
    data_root: Path | None = None
    basins_info_path: Path | None = None
    main_year: str = RIVER_MAIN_YEAR
    n_days: int = RIVER_N_DAYS
    missing_threshold: int = RIVER_MISSING_THRESHOLD
    relevant_variables: tuple[str, ...] = tuple(RIVER_VARIABLES)
    all_variables: tuple[str, ...] = tuple(RIVER_ALL_VARIABLES)
    node_names: tuple[str, ...] = tuple(RIVER_NODES)
    n_locs: int | None = None


def default_river_data_root() -> Path:
    candidates = [
        data_dir("river", "timeseries"),
        data_dir("basin data", "timeseries"),
        data_dir("basin data", "basin data", "timeseries"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def default_basins_info_path() -> Path:
    candidates = [
        data_dir("river", "basins_info.csv"),
        data_dir("basin data", "basins_info.csv"),
        data_dir("basin data", "basin data", "basins_info.csv"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def default_river_spacetime_config(
    *,
    score_type: str = "ff",
    context_col: str = "context",
    detect_contexts: bool = True,
    detect_regimes: bool = True,
    mechanism_clustering: str = "edge-strengths",
) -> SpaceTimeExperimentConfig:
    return SpaceTimeExperimentConfig(
        score_type=score_type,
        tau_max=3,
        context_col=context_col,
        changepoints="detect",
        d_min=30,
        max_iter=1,
        detect_contexts=detect_contexts,
        detect_regimes=detect_regimes,
        mechanism_clustering=mechanism_clustering,
        standardize="per_context",
        fill_method="ffill_bfill",
    )


def river_reference_graph(*, tau_max: int = 3):
    return graph_from_index_links(
        RIVER_RESULT_LINKS,
        variables=RIVER_NODES,
        tau_max=tau_max,
    )
