from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.data_types import PanelDataset
from experiments.river.config import RiverLoadConfig, default_basins_info_path, default_river_data_root


@dataclass(frozen=True)
class RiverData:
    selected_year: PanelDataset
    yearly: dict[int, dict[str, pd.DataFrame]]
    yearly_all_variables: dict[int, dict[str, pd.DataFrame]]
    location_ids: dict[int, str]
    selected_years: dict[int, str]
    basins_info: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)

    def subset_first(self, n_locs: int) -> RiverData:
        context_ids = list(self.location_ids.keys())[:n_locs]

        return RiverData(
            selected_year=_subset_panel_dataset(self.selected_year, context_ids),
            yearly={idx: self.yearly[idx] for idx in context_ids},
            yearly_all_variables={idx: self.yearly_all_variables[idx] for idx in context_ids},
            location_ids={new_idx: self.location_ids[idx] for new_idx, idx in enumerate(context_ids)},
            selected_years={new_idx: self.selected_years[idx] for new_idx, idx in enumerate(context_ids)},
            basins_info=self.basins_info,
            metadata=dict(self.metadata),
        )


def load_river_data(config: RiverLoadConfig | None = None) -> RiverData:
    """Load river runoff data using the old experiment preprocessing choices."""
    config = config or RiverLoadConfig()
    data_root = Path(config.data_root or default_river_data_root()).expanduser().resolve()
    basins_info_path = Path(config.basins_info_path or default_basins_info_path()).expanduser().resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"River timeseries directory does not exist: {data_root}")

    basins_info = pd.read_csv(basins_info_path) if basins_info_path.exists() else None

    selected_panel: dict[int, pd.DataFrame] = {}
    yearly: dict[int, dict[str, pd.DataFrame]] = {}
    yearly_all_vars: dict[int, dict[str, pd.DataFrame]] = {}
    location_ids: dict[int, str] = {}
    selected_years: dict[int, str] = {}

    file_count = 0

    for path in sorted(data_root.rglob("*.csv")):
        location_id = path.stem
        raw = pd.read_csv(path)

        if "time" not in raw.columns:
            continue

        years = _available_years(raw["time"])
        if not years:
            continue

        selected_year = (
            str(config.main_year) if str(config.main_year) in years else _closest_year(years, str(config.main_year))
        )
        selected = _frame_for_year(raw, selected_year)

        if selected.empty:
            continue

        relevant = _clean_river_frame(
            selected,
            variables=config.relevant_variables,
            missing_threshold=config.missing_threshold,
        )
        # all_vars = _clean_river_frame(
        #    selected,
        #    variables=[var for var in config.all_variables if var in selected.columns],
        #    missing_threshold=None,
        # )

        if relevant is None:
            continue

        context_id = file_count
        location_ids[context_id] = location_id
        selected_years[context_id] = selected_year
        selected_panel[context_id] = relevant.iloc[: config.n_days].reset_index(drop=True)

        yearly[context_id] = {}
        yearly_all_vars[context_id] = {}

        for year in years:
            year_frame = _frame_for_year(raw, str(year))
            relevant_year = _clean_river_frame(
                year_frame,
                variables=config.relevant_variables,
                missing_threshold=config.missing_threshold,
            )
            all_year = _clean_river_frame(
                year_frame,
                variables=[var for var in config.all_variables if var in year_frame.columns],
                missing_threshold=None,
            )

            if relevant_year is None:
                continue

            yearly[context_id][str(year)] = relevant_year.iloc[: config.n_days].reset_index(drop=True)
            yearly_all_vars[context_id][str(year)] = all_year.iloc[: config.n_days].reset_index(drop=True)

        file_count += 1

        if config.n_locs is not None and file_count >= config.n_locs:
            break

    selected_year = PanelDataset(
        name="river_selected_year",
        panel=selected_panel,
        variables=list(config.node_names),
        context_col="context",
        metadata={
            "location_ids": location_ids,
            "selected_years": selected_years,
            "source_root": str(data_root),
            "raw_variables": list(config.relevant_variables),
        },
    )
    selected_year = _rename_panel_dataset(
        selected_year, dict(zip(config.relevant_variables, config.node_names, strict=False))
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

    return RiverData(
        selected_year=selected_year,
        yearly=yearly_renamed,
        yearly_all_variables=yearly_all_vars,
        location_ids=location_ids,
        selected_years=selected_years,
        basins_info=basins_info,
        metadata={
            "n_locations": len(location_ids),
            "data_root": str(data_root),
            "basins_info_path": str(basins_info_path) if basins_info_path.exists() else None,
        },
    )


def yearly_panel_dataset(
    river_data: RiverData,
    *,
    context_id: int,
    year: str,
) -> PanelDataset:
    location_id = river_data.location_ids[context_id]
    frame = river_data.yearly[context_id][str(year)]

    return PanelDataset(
        name=f"river_{location_id}_{year}",
        panel={0: frame.reset_index(drop=True)},
        variables=list(river_data.selected_year.variables),
        context_col=river_data.selected_year.context_col,
        metadata={
            "location_id": location_id,
            "context_id": context_id,
            "year": str(year),
        },
    )


def _available_years(time: pd.Series) -> list[str]:
    return sorted(time.astype(str).str[:4].unique().tolist())


def _closest_year(years: list[str], target: str) -> str:
    return years[int(np.argmin([abs(int(year) - int(target)) for year in years]))]


def _frame_for_year(frame: pd.DataFrame, year: str) -> pd.DataFrame:
    mask = frame["time"].astype(str).str.startswith(str(year))
    return frame.loc[mask].copy()


def _clean_river_frame(
    frame: pd.DataFrame,
    *,
    variables: list[str] | tuple[str, ...],
    missing_threshold: int | None,
) -> pd.DataFrame | None:
    variables = list(variables)
    missing_cols = [var for var in variables if var not in frame.columns]

    if missing_cols:
        return None

    out = frame.drop(columns=["time"]).loc[:, variables].copy()

    if missing_threshold is not None:
        missing = out.isna().sum(axis=0)
        if bool((missing > missing_threshold).any()):
            return None

    return out.fillna(0).reset_index(drop=True)


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
    location_ids = dataset.metadata.get("location_ids", {})
    selected_years = dataset.metadata.get("selected_years", {})

    metadata = dict(dataset.metadata)
    metadata["location_ids"] = {
        new_idx: location_ids[old_idx] for new_idx, old_idx in enumerate(context_ids) if old_idx in location_ids
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
