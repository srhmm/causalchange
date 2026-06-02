from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.data_types import PanelDataset
from experiments.flux.config import (
    FLUX_INTERESTING_SITE_IDS,
    FluxLoadConfig,
    default_flux_data_root,
    default_flux_location_info_path,
)


@dataclass(frozen=True)
class FluxData:
    """FluxNet data in the shapes needed by the experiments."""

    selected_year: PanelDataset
    selected_two_years: PanelDataset
    yearly: dict[int, dict[str, pd.DataFrame]]
    yearly_all_variables: dict[int, dict[str, pd.DataFrame]]
    site_ids: dict[int, str]
    selected_years: dict[int, tuple[str, str]]
    metadata: dict = field(default_factory=dict)

    def subset_sites(self, site_ids: Iterable[str]) -> FluxData:
        wanted = set(site_ids)
        context_ids = [idx for idx, site in self.site_ids.items() if site in wanted]

        return FluxData(
            selected_year=_subset_panel_dataset(self.selected_year, context_ids),
            selected_two_years=_subset_panel_dataset(self.selected_two_years, context_ids),
            yearly={idx: self.yearly[idx] for idx in context_ids},
            yearly_all_variables={idx: self.yearly_all_variables[idx] for idx in context_ids},
            site_ids={new_idx: self.site_ids[idx] for new_idx, idx in enumerate(context_ids)},
            selected_years={new_idx: self.selected_years[idx] for new_idx, idx in enumerate(context_ids)},
            metadata=dict(self.metadata),
        )

    def interesting_sites(self) -> FluxData:
        return self.subset_sites(FLUX_INTERESTING_SITE_IDS)


def load_flux_data(config: FluxLoadConfig | None = None) -> FluxData:
    """Load FluxNet data using the old experiment preprocessing choices.

    The loader preserves the old choices:
    - only daily FULLSET files,
    - years constrained by ``fluxnet_locations_krichetal.csv``,
    - selected year closest to 2006 plus the following year,
    - skip years/sites with too many sentinel values in a 30-day month,
    - replace -9999 sentinels by 0 before the common preprocessing stage.
    """
    config = config or FluxLoadConfig()
    data_root = Path(config.data_root or default_flux_data_root()).expanduser().resolve()
    location_info_path = Path(config.location_info_path or default_flux_location_info_path()).expanduser().resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Flux data root does not exist: {data_root}")
    if not location_info_path.exists():
        raise FileNotFoundError(f"Flux location info file does not exist: {location_info_path}")

    loc_info = pd.read_csv(location_info_path)

    selected_one_year_panel: dict[int, pd.DataFrame] = {}
    selected_two_year_panel: dict[int, pd.DataFrame] = {}
    yearly: dict[int, dict[str, pd.DataFrame]] = {}
    yearly_all_vars: dict[int, dict[str, pd.DataFrame]] = {}
    site_ids: dict[int, str] = {}
    selected_years: dict[int, tuple[str, str]] = {}

    file_count = 0

    for path in _daily_flux_files(data_root):
        site_id = _site_id_from_filename(path.name)

        if site_id is None:
            continue

        raw = pd.read_csv(path)
        if "TIMESTAMP" not in raw.columns:
            continue

        used_years = _used_years_for_site(loc_info, site_id)
        if not used_years:
            continue

        raw = _filter_timestamp_years(raw, used_years)
        available_years = _available_years(raw["TIMESTAMP"])

        if not available_years:
            continue

        year_1 = _closest_year(available_years, str(config.main_year))
        year_2_target = str(int(year_1) + 1)
        year_2 = _closest_year(available_years, year_2_target)

        combined = _filter_timestamp_years(raw, [year_1, year_2])

        if not _valid_flux_frame(
            combined,
            variables=config.relevant_variables,
            max_missing_per_month=config.max_missing_per_month,
            month_length=config.month_length,
        ):
            continue

        relevant_combined = _clean_flux_frame(
            combined,
            variables=config.relevant_variables,
            timestamp_col="TIMESTAMP",
        )
        # all_combined = _clean_flux_frame(
        #    combined,
        #    variables=config.all_variables,
        #    timestamp_col="TIMESTAMP",
        # )

        if relevant_combined.empty:
            continue

        context_id = file_count
        site_ids[context_id] = site_id
        selected_years[context_id] = (year_1, year_2)
        selected_two_year_panel[context_id] = relevant_combined.reset_index(drop=True)
        selected_one_year_panel[context_id] = relevant_combined.iloc[: config.n_days].reset_index(drop=True)

        yearly[context_id] = {}
        yearly_all_vars[context_id] = {}

        for year in used_years:
            year = str(year)
            raw_year = _filter_timestamp_years(raw, [year])

            if raw_year.empty:
                continue

            if not _valid_flux_frame(
                raw_year,
                variables=config.relevant_variables,
                max_missing_per_month=config.max_missing_per_month,
                month_length=config.month_length,
            ):
                continue

            yearly[context_id][year] = (
                _clean_flux_frame(
                    raw_year,
                    variables=config.relevant_variables,
                    timestamp_col="TIMESTAMP",
                )
                .iloc[: config.n_days]
                .reset_index(drop=True)
            )

            yearly_all_vars[context_id][year] = (
                _clean_flux_frame(
                    raw_year,
                    variables=config.all_variables,
                    timestamp_col="TIMESTAMP",
                )
                .iloc[: config.n_days]
                .reset_index(drop=True)
            )

        file_count += 1

        if config.n_locs is not None and file_count >= config.n_locs:
            break

    selected_year = PanelDataset(
        name="flux_selected_year",
        panel=selected_one_year_panel,
        variables=list(config.node_names),
        context_col="context",
        metadata={
            "site_ids": site_ids,
            "selected_years": selected_years,
            "source_root": str(data_root),
            "location_info_path": str(location_info_path),
            "raw_variables": list(config.relevant_variables),
        },
    )

    # Rename to compact paper node names only after all variable selection.
    selected_year = _rename_panel_dataset(
        selected_year, dict(zip(config.relevant_variables, config.node_names, strict=False))
    )

    selected_two_years = PanelDataset(
        name="flux_selected_two_years",
        panel=selected_two_year_panel,
        variables=list(config.node_names),
        context_col="context",
        metadata={
            "site_ids": site_ids,
            "selected_years": selected_years,
            "source_root": str(data_root),
            "location_info_path": str(location_info_path),
            "raw_variables": list(config.relevant_variables),
        },
    )
    selected_two_years = _rename_panel_dataset(
        selected_two_years, dict(zip(config.relevant_variables, config.node_names, strict=False))
    )

    yearly_renamed = {
        context_id: {
            year: frame.rename(columns=dict(zip(config.relevant_variables, config.node_names, strict=False))).loc[
                :, list(config.node_names)
            ]
            for year, frame in years.items()
        }
        for context_id, years in yearly.items()
    }

    return FluxData(
        selected_year=selected_year,
        selected_two_years=selected_two_years,
        yearly=yearly_renamed,
        yearly_all_variables=yearly_all_vars,
        site_ids=site_ids,
        selected_years=selected_years,
        metadata={
            "n_sites": len(site_ids),
            "data_root": str(data_root),
            "location_info_path": str(location_info_path),
        },
    )


def yearly_panel_dataset(
    flux_data: FluxData,
    *,
    context_id: int,
    year: str,
) -> PanelDataset:
    site_id = flux_data.site_ids[context_id]
    frame = flux_data.yearly[context_id][str(year)]

    return PanelDataset(
        name=f"flux_{site_id}_{year}",
        panel={0: frame.reset_index(drop=True)},
        variables=list(flux_data.selected_year.variables),
        context_col=flux_data.selected_year.context_col,
        metadata={
            "site_id": site_id,
            "context_id": context_id,
            "year": str(year),
        },
    )


def _daily_flux_files(root: Path) -> list[Path]:
    paths = []

    for path in root.rglob("*.csv"):
        parts = path.name.split("_")
        if "DD" in parts and "FULLSET" in parts:
            paths.append(path)

    return sorted(paths)


def _site_id_from_filename(name: str) -> str | None:
    parts = name.split("_")

    if "FLX" not in parts:
        return None

    idx = parts.index("FLX") + 1
    if idx >= len(parts):
        return None

    return parts[idx]


def _used_years_for_site(loc_info: pd.DataFrame, site_id: str) -> list[str]:
    rows = loc_info.loc[loc_info["FLUXNETID"] == site_id]

    if rows.empty:
        return []

    start = int(rows.iloc[0]["Startyear"])
    end = int(rows.iloc[0]["Endyear"])
    return [str(year) for year in range(start, end + 1)]


def _filter_timestamp_years(frame: pd.DataFrame, years: Iterable[str]) -> pd.DataFrame:
    years = tuple(str(year) for year in years)
    mask = frame["TIMESTAMP"].astype(str).map(lambda value: value.startswith(years))
    return frame.loc[mask].copy()


def _available_years(timestamp: pd.Series) -> list[str]:
    return sorted(timestamp.astype(str).str[:4].unique().tolist())


def _closest_year(years: Iterable[str], target: str) -> str:
    years = list(years)
    return years[int(np.argmin([abs(int(year) - int(target)) for year in years]))]


def _valid_flux_frame(
    frame: pd.DataFrame,
    *,
    variables: Iterable[str],
    max_missing_per_month: int,
    month_length: int,
) -> bool:
    variables = list(variables)
    n_months = 12

    for month_idx in range(n_months):
        start = month_idx * month_length
        stop = (month_idx + 1) * month_length
        block = frame.iloc[start:stop]

        if block.empty:
            continue

        max_missing = max(int((block[var] < -9000).sum()) for var in variables if var in block.columns)

        if max_missing > max_missing_per_month:
            return False

    return True


def _clean_flux_frame(
    frame: pd.DataFrame,
    *,
    variables: Iterable[str],
    timestamp_col: str,
) -> pd.DataFrame:
    variables = list(variables)
    missing = [var for var in variables if var not in frame.columns]

    if missing:
        raise ValueError(f"Flux frame is missing variables: {missing}")

    out = frame.drop(columns=[timestamp_col]).copy()
    out = out.replace(-9999, 0).replace(-9999.0, 0)
    return out.loc[:, variables].reset_index(drop=True)


def _rename_panel_dataset(dataset: PanelDataset, rename: dict[str, str]) -> PanelDataset:
    return PanelDataset(
        name=dataset.name,
        panel={
            context_id: frame.rename(columns=rename).loc[:, list(rename.values())].reset_index(drop=True)
            for context_id, frame in dataset.panel.items()
        },
        variables=list(rename.values()),
        context_col=dataset.context_col,
        metadata=dict(dataset.metadata),
    )


def _subset_panel_dataset(dataset: PanelDataset, context_ids: list[int]) -> PanelDataset:
    panel = {new_idx: dataset.panel[old_idx].reset_index(drop=True) for new_idx, old_idx in enumerate(context_ids)}
    site_ids = dataset.metadata.get("site_ids", {})
    selected_years = dataset.metadata.get("selected_years", {})

    metadata = dict(dataset.metadata)
    metadata["site_ids"] = {
        new_idx: site_ids[old_idx] for new_idx, old_idx in enumerate(context_ids) if old_idx in site_ids
    }
    metadata["selected_years"] = {
        new_idx: selected_years[old_idx] for new_idx, old_idx in enumerate(context_ids) if old_idx in selected_years
    }

    return PanelDataset(
        name=dataset.name,
        panel=panel,
        variables=dataset.variables,
        context_col=dataset.context_col,
        metadata=metadata,
    )
