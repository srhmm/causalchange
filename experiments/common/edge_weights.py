from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any

import pandas as pd


def node_to_parts(node: Any) -> tuple[str, int]:
    """Return ``(variable, lag)`` from a tuple node or tuple-like string."""
    if isinstance(node, tuple) and len(node) == 2:
        return str(node[0]), int(node[1])

    if isinstance(node, str):
        try:
            parsed = ast.literal_eval(node)
            if isinstance(parsed, tuple) and len(parsed) == 2:
                return str(parsed[0]), int(parsed[1])
        except Exception:
            pass

        return node, 0

    return str(node), 0


def add_edge_columns(
    frame: pd.DataFrame,
    *,
    parent_col: str = "parent",
    effect_col: str = "effect",
) -> pd.DataFrame:
    """Add explicit parent/target variable, lag, and edge label columns."""
    out = frame.copy()

    parent_parts = out[parent_col].map(node_to_parts)
    effect_parts = out[effect_col].map(node_to_parts)

    out["parent_var"] = parent_parts.map(lambda x: x[0])
    out["parent_lag"] = parent_parts.map(lambda x: x[1])
    out["target_var"] = effect_parts.map(lambda x: x[0])
    out["target_lag"] = effect_parts.map(lambda x: x[1])

    out["edge"] = (
        out["parent_var"].astype(str)
        + ":lag"
        + out["parent_lag"].astype(str)
        + "->"
        + out["target_var"].astype(str)
        + ":lag"
        + out["target_lag"].astype(str)
    )

    out["summary_edge"] = out["parent_var"].astype(str) + "->" + out["target_var"].astype(str)

    return out


def contribution_matrix(
    edge_contributions: pd.DataFrame,
    *,
    value_col: str = "positive_gain",
    index_cols: Sequence[str] = ("dataset_id", "interval_id"),
    edge_col: str = "edge",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Pivot tidy edge contributions to a matrix for clustering."""
    if edge_contributions.empty:
        return pd.DataFrame()

    frame = edge_contributions.copy()

    if edge_col not in frame.columns:
        frame = add_edge_columns(frame)

    matrix = frame.pivot_table(
        index=list(index_cols),
        columns=edge_col,
        values=value_col,
        aggfunc="mean",
        fill_value=fill_value,
    )

    return matrix.sort_index(axis=0).sort_index(axis=1)


def mechanism_score_matrix(
    mechanism_scores: pd.DataFrame,
    *,
    value_col: str = "score",
    index_cols: Sequence[str] = ("dataset_id", "interval_id"),
    target_col: str = "target",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Pivot mechanism scores to one target-mechanism column per context/window."""
    if mechanism_scores.empty:
        return pd.DataFrame()

    frame = mechanism_scores.copy()
    frame["mechanism"] = frame[target_col].astype(str) + "|parents=" + frame["parents"].astype(str)

    matrix = frame.pivot_table(
        index=list(index_cols),
        columns="mechanism",
        values=value_col,
        aggfunc="mean",
        fill_value=fill_value,
    )

    return matrix.sort_index(axis=0).sort_index(axis=1)


def aggregate_contributions_by_summary_edge(
    edge_contributions: pd.DataFrame,
    *,
    value_col: str = "positive_gain",
    group_cols: Sequence[str] = ("dataset_id", "interval_id", "summary_edge"),
) -> pd.DataFrame:
    """Aggregate lag-specific edge contributions to variable-level summary edges."""
    if edge_contributions.empty:
        return pd.DataFrame(columns=[*group_cols, value_col])

    frame = add_edge_columns(edge_contributions)

    return frame.groupby(list(group_cols), dropna=False)[value_col].mean().reset_index()
