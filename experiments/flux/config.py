from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.common.data_types import SpaceTimeExperimentConfig
from experiments.common.paths import data_dir
from experiments.common.reference_graphs import graph_from_index_links

FLUX_RELEVANT_VARIABLES = [
    "SW_IN_F_MDS",
    "NEE_VUT_USTAR50",
    "TA_F_MDS",
    "VPD_F_MDS",
    "H_F_MDS",
    "LE_F_MDS",
]

FLUX_ALL_VARIABLES = [
    "SW_IN_F_MDS",
    "NEE_VUT_USTAR50",
    "TA_F_MDS",
    "VPD_F_MDS",
    "H_F_MDS",
    "LE_F_MDS",
    "GPP_NT_VUT_USTAR50",
    "P_F",
]

FLUX_NODES = ["R", "NEE", "T", "VPD", "H", "LE"]
FLUX_ALL_NODES = ["R", "NEE", "T", "VPD", "H", "LE", "GPP", "P"]

FLUX_MAIN_YEAR = 2006
FLUX_N_DAYS = 365
FLUX_MONTH_LENGTH = 30
FLUX_MAX_MISSING_PER_MONTH = 8

FLUX_INTERESTING_SITE_IDS = ["GF-Guy", "DE-Hai", "FR-Pue", "US-SRM", "FI-Hyy"]

# Old reference links from the paper/previous implementation.
FLUX_RESULT_LINKS = {
    0: [((0, 1), 1, None), ((5, 0), 1, None)],
    1: [
        ((0, 0), 1, None),
        ((1, 1), 1, None),
        ((2, 0), 1, None),
        ((5, 0), 1, None),
    ],
    2: [((2, 1), 1, None), ((3, 0), 1, None)],
    3: [((0, 0), 1, None), ((3, 1), 1, None)],
    4: [((0, 0), 1, None), ((4, 1), 1, None)],
    5: [((3, 0), 1, None), ((4, 0), 1, None), ((5, 1), 1, None)],
}


@dataclass(frozen=True)
class FluxLoadConfig:
    data_root: Path | None = None
    location_info_path: Path | None = None
    main_year: int = FLUX_MAIN_YEAR
    n_days: int = FLUX_N_DAYS
    max_missing_per_month: int = FLUX_MAX_MISSING_PER_MONTH
    month_length: int = FLUX_MONTH_LENGTH
    relevant_variables: tuple[str, ...] = tuple(FLUX_RELEVANT_VARIABLES)
    all_variables: tuple[str, ...] = tuple(FLUX_ALL_VARIABLES)
    node_names: tuple[str, ...] = tuple(FLUX_NODES)
    all_node_names: tuple[str, ...] = tuple(FLUX_ALL_NODES)
    n_locs: int | None = None


def default_flux_data_root() -> Path:
    candidates = [
        data_dir("flux"),
        data_dir("flux_krich_et_al"),
        data_dir("flux_krich_et_al2", "flux_krich_et_al"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def default_flux_location_info_path() -> Path:
    candidates = [
        data_dir("reproduce_info", "fluxnet_locations_krichetal.csv"),
        data_dir("flux", "fluxnet_locations_krichetal.csv"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def default_flux_spacetime_config(
    *,
    score_type: str = "ff",
    context_col: str = "context",
    detect_contexts: bool = True,
    detect_regimes: bool = True,
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
        standardize="per_context",
        fill_method="ffill_bfill",
    )


def flux_reference_graph(*, tau_max: int = 3):
    return graph_from_index_links(
        FLUX_RESULT_LINKS,
        variables=FLUX_NODES,
        tau_max=tau_max,
    )
