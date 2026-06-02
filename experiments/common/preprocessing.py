from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from experiments.common.data_types import PanelDataset

FillMethod = Literal["none", "ffill_bfill", "interpolate"]
StandardizeMode = Literal["none", "global", "per_context"]


def select_numeric_columns(
    frame: pd.DataFrame,
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Select numeric columns, optionally restricted to ``include``."""
    if include is not None:
        missing = [col for col in include if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        selected = frame.loc[:, list(include)].copy()
    else:
        selected = frame.select_dtypes(include=[np.number]).copy()

    if exclude:
        selected = selected.drop(columns=[col for col in exclude if col in selected.columns])

    return selected


def replace_sentinels(
    frame: pd.DataFrame,
    *,
    sentinels: Iterable[float | int] = (-9999, -9999.0, -999, -999.0),
) -> pd.DataFrame:
    """Replace common missing-value sentinels by NaN."""
    out = frame.copy()
    return out.replace(list(sentinels), np.nan)


def fill_missing(
    frame: pd.DataFrame,
    *,
    method: FillMethod = "ffill_bfill",
    limit_direction: Literal["forward", "backward", "both"] = "both",
) -> pd.DataFrame:
    if method == "none":
        return frame.copy()

    if method == "ffill_bfill":
        return frame.ffill().bfill()

    if method == "interpolate":
        return frame.interpolate(limit_direction=limit_direction).ffill().bfill()

    raise ValueError(f"Unknown fill method: {method!r}")


def drop_high_missing_columns(
    frame: pd.DataFrame,
    *,
    max_missing_fraction: float,
) -> pd.DataFrame:
    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError("max_missing_fraction must be in [0, 1].")

    missing_fraction = frame.isna().mean(axis=0)
    keep = missing_fraction[missing_fraction <= max_missing_fraction].index.tolist()
    return frame.loc[:, keep].copy()


def standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Z-standardize columns, leaving constant columns as zeros."""
    mean = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (frame - mean) / std


def standardize_panel(
    panel: Mapping[Hashable, pd.DataFrame],
    *,
    variables: Sequence[str],
    mode: StandardizeMode = "per_context",
) -> dict[Hashable, pd.DataFrame]:
    variables = list(variables)

    if mode == "none":
        return {context_id: frame.copy() for context_id, frame in panel.items()}

    if mode == "per_context":
        return {
            context_id: standardize_frame(frame.loc[:, variables]).reset_index(drop=True)
            for context_id, frame in panel.items()
        }

    if mode == "global":
        stacked = pd.concat(
            [frame.loc[:, variables] for frame in panel.values()],
            axis=0,
            ignore_index=True,
        )
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=0).replace(0.0, 1.0)

        return {
            context_id: ((frame.loc[:, variables] - mean) / std).reset_index(drop=True)
            for context_id, frame in panel.items()
        }

    raise ValueError(f"Unknown standardization mode: {mode!r}")


def align_panel_lengths(
    panel: Mapping[Hashable, pd.DataFrame],
    *,
    n_samples: int | None = None,
    trim: Literal["min", "raise"] = "min",
) -> dict[Hashable, pd.DataFrame]:
    """Ensure all contexts have the same length.

    SpaceTime currently assumes equal-length contexts for changepoint windows.
    """
    lengths = {context_id: len(frame) for context_id, frame in panel.items()}

    if n_samples is None:
        if trim == "min":
            n_samples = min(lengths.values())
        elif len(set(lengths.values())) == 1:
            n_samples = next(iter(lengths.values()))
        else:
            raise ValueError(f"Panel lengths differ: {lengths}")
    else:
        too_short = {context_id: length for context_id, length in lengths.items() if length < n_samples}
        if too_short:
            raise ValueError(f"Some contexts are shorter than n_samples={n_samples}: {too_short}")

    return {context_id: frame.iloc[: int(n_samples)].reset_index(drop=True) for context_id, frame in panel.items()}


def build_context_dataframe(
    panel: Mapping[Hashable, pd.DataFrame],
    *,
    context_col: str = "context",
    variables: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert ``{context_id: dataframe}`` to one dataframe with a context column."""
    blocks: list[pd.DataFrame] = []

    for context_id, frame in panel.items():
        current = frame.copy()

        if variables is not None:
            current = current.loc[:, list(variables)].copy()

        current[context_col] = context_id
        blocks.append(current)

    if not blocks:
        raise ValueError("Cannot build context dataframe from an empty panel.")

    return pd.concat(blocks, axis=0, ignore_index=True)


def preprocess_panel_dataset(
    dataset: PanelDataset,
    *,
    fill_method: FillMethod = "ffill_bfill",
    standardize: StandardizeMode = "per_context",
    max_missing_fraction: float | None = None,
    n_samples: int | None = None,
) -> PanelDataset:
    """Apply common cleaning steps to a PanelDataset."""
    dataset.validate()

    variables = list(dataset.variables)
    cleaned: dict[Hashable, pd.DataFrame] = {}

    for context_id, frame in dataset.panel.items():
        current = replace_sentinels(frame.loc[:, variables])
        if max_missing_fraction is not None:
            current = drop_high_missing_columns(current, max_missing_fraction=max_missing_fraction)
            missing_vars = [var for var in variables if var not in current.columns]
            if missing_vars:
                raise ValueError(
                    f"Context {context_id!r} dropped required variables due to missingness: {missing_vars}"
                )
        current = fill_missing(current, method=fill_method)
        cleaned[context_id] = current.loc[:, variables].reset_index(drop=True)

    cleaned = align_panel_lengths(cleaned, n_samples=n_samples)
    cleaned = standardize_panel(cleaned, variables=variables, mode=standardize)

    return PanelDataset(
        name=dataset.name,
        panel=cleaned,
        variables=variables,
        context_col=dataset.context_col,
        metadata=dict(dataset.metadata),
    )
