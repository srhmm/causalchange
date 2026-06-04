from __future__ import annotations

import pandas as pd

from causalchange.core.require import require_matplotlib


def plot_changepoints(
    X: pd.DataFrame,
    *,
    changepoints: list[int] | None = None,
    context_col: str | None = None,
    title: str | None = None,
):
    plt = require_matplotlib()

    changepoints = list(changepoints or [])

    if context_col is None or context_col not in X.columns:
        ax = X.plot(figsize=(10, 4), title=title)

        for cp in changepoints:
            ax.axvline(cp, linestyle="--", linewidth=1)

        ax.set_xlabel("time")
        return ax

    variables = [col for col in X.columns if col != context_col]
    contexts = list(X[context_col].unique())

    fig, axes = plt.subplots(
        len(variables),
        1,
        figsize=(10, 2.4 * len(variables)),
        sharex=True,
    )

    if len(variables) == 1:
        axes = [axes]

    for ax, variable in zip(axes, variables, strict=True):
        for context in contexts:
            values = X.loc[X[context_col] == context, variable].to_numpy()
            ax.plot(values, label=f"context={context}", alpha=0.85)

        for cp in changepoints:
            ax.axvline(cp, linestyle="--", linewidth=1)

        ax.set_ylabel(variable)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return axes


def plot_partitions(partitions, *, title: str | None = None):
    plt = require_matplotlib()

    rows = []

    for target, mapping in partitions.cell_clusters.items():
        for cell, label in mapping.items():
            interval = partitions.intervals_by_context[cell.dataset_id][cell.interval_id]

            rows.append(
                {
                    "target": target,
                    "dataset_id": str(cell.dataset_id),
                    "interval_id": cell.interval_id,
                    "interval_start": interval[0],
                    "interval_stop": interval[1],
                    "mechanism": label,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No partition information available.")

    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(df))))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    if title:
        ax.set_title(title)

    return ax
