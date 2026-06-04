from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from experiments.common.data_types import SpaceTimeExperimentRun
from experiments.common.edge_weights import add_edge_columns
from experiments.common.paths import ensure_dir


def _json_default(value: Any):
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def graph_to_edge_frame(graph: nx.DiGraph) -> pd.DataFrame:
    rows = []

    for parent, effect in graph.edges():
        parent_var, parent_lag = _node_parts(parent)
        target_var, target_lag = _node_parts(effect)

        rows.append(
            {
                "parent": parent,
                "effect": effect,
                "parent_var": parent_var,
                "parent_lag": parent_lag,
                "target_var": target_var,
                "target_lag": target_lag,
            }
        )

    return pd.DataFrame(rows)


def edge_frame_to_graph(frame: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()

    required = {"parent_var", "parent_lag", "target_var", "target_lag"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot reconstruct graph; missing columns: {sorted(missing)}")

    for row in frame.itertuples(index=False):
        parent = (str(row.parent_var), int(row.parent_lag))
        effect = (str(row.target_var), int(row.target_lag))
        graph.add_edge(parent, effect)

    return graph


def partitions_to_frames(partitions) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_rows = []
    regime_rows = []

    for target, mapping in partitions.contexts.items():
        for dataset_id, label in mapping.items():
            context_rows.append(
                {
                    "target": target,
                    "dataset_id": dataset_id,
                    "context_cluster": int(label),
                }
            )

    for target, mapping in partitions.regimes.items():
        for interval_id, label in mapping.items():
            regime_rows.append(
                {
                    "target": target,
                    "interval_id": int(interval_id),
                    "regime_cluster": int(label),
                }
            )

    return pd.DataFrame(context_rows), pd.DataFrame(regime_rows)


def save_posthoc_tables(run: SpaceTimeExperimentRun, out_dir: Path | str) -> dict[str, Path]:
    out_dir = ensure_dir(out_dir)
    paths: dict[str, Path] = {}

    for name, table in run.posthoc.nonempty().items():
        current = table.copy()

        if name.startswith("edge_contributions"):
            current = add_edge_columns(current)

        path = out_dir / f"{name}.csv"
        current.to_csv(path, index=False)
        paths[name] = path

    return paths


def save_spacetime_run(
    run: SpaceTimeExperimentRun,
    out_dir: Path | str,
    *,
    save_panel: bool = False,
) -> dict[str, Path]:
    out_dir = ensure_dir(out_dir)

    paths: dict[str, Path] = {}

    graph_path = out_dir / "graph_edges.csv"
    graph_to_edge_frame(run.graph).to_csv(graph_path, index=False)
    paths["graph_edges"] = graph_path

    changepoints_path = out_dir / "changepoints.json"
    write_json(
        changepoints_path,
        {
            "changepoints": run.changepoints,
            "changepoints_by_context": getattr(run.estimator, "changepoints_by_context_", None),
            "diagnostics": getattr(run.estimator.result_, "changepoint_diagnostics", None),
        },
    )
    paths["changepoints"] = changepoints_path

    context_partitions, regime_partitions = partitions_to_frames(run.partitions)
    context_path = out_dir / "context_partitions.csv"
    regime_path = out_dir / "regime_partitions.csv"
    context_partitions.to_csv(context_path, index=False)
    regime_partitions.to_csv(regime_path, index=False)
    paths["context_partitions"] = context_path
    paths["regime_partitions"] = regime_path

    config_path = out_dir / "config.json"
    write_json(
        config_path,
        {
            "dataset": {
                "name": run.dataset.name,
                "variables": list(run.dataset.variables),
                "context_col": run.dataset.context_col,
                "metadata": dict(run.dataset.metadata),
            },
            "config": run.config,
            "estimator_config": run.estimator.cfg,
        },
    )
    paths["config"] = config_path

    paths.update(save_posthoc_tables(run, out_dir))

    if save_panel:
        panel_dir = ensure_dir(out_dir / "panel")
        for context_id, frame in run.dataset.panel.items():
            safe_id = str(context_id).replace("/", "_").replace("\\", "_")
            path = panel_dir / f"context_{safe_id}.csv"
            frame.to_csv(path, index=False)
            paths[f"panel_{context_id}"] = path

    return paths


def _node_parts(node) -> tuple[str, int]:
    if isinstance(node, tuple) and len(node) == 2:
        return str(node[0]), int(node[1])
    return str(node), 0
