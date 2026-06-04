from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import pandas as pd

from causalchange.causal_change import CausalChange

ChangepointModeName = Literal["none", "detect", "fixed"]
ScoreName = Literal["lin", "gam", "spline", "krr", "gp", "ff"]
MechanismClusteringName = Literal["testing", "edge-strengths", "skip"]


@dataclass(frozen=True)
class PanelDataset:
    """A collection of aligned time-series datasets.

    ``panel`` maps a dataset/context id, such as site id or station id, to a
    dataframe with the same numeric variables in the same order.
    """

    name: str
    panel: Mapping[Hashable, pd.DataFrame]
    variables: Sequence[str]
    context_col: str = "context"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def context_ids(self) -> list[Hashable]:
        return list(self.panel.keys())

    def n_contexts(self) -> int:
        return len(self.panel)

    def validate(self) -> None:
        if not self.panel:
            raise ValueError("PanelDataset.panel must contain at least one dataset.")

        variables = list(self.variables)

        if not variables:
            raise ValueError("PanelDataset.variables must be non-empty.")

        for context_id, frame in self.panel.items():
            missing = [col for col in variables if col not in frame.columns]
            if missing:
                raise ValueError(f"Context {context_id!r} is missing columns: {missing}")

            if frame[variables].empty:
                raise ValueError(f"Context {context_id!r} has no rows after selecting variables.")


@dataclass(frozen=True)
class SpaceTimeExperimentConfig:
    """Configuration shared by Flux/River SpaceTime experiments."""

    score_type: ScoreName = "ff"
    tau_max: int = 1
    context_col: str = "context"

    changepoints: ChangepointModeName = "detect"
    fixed_changepoints: tuple[int, ...] = ()
    d_min: int = 30
    max_iter: int = 3
    pelt_penalty: float | str = "auto"

    detect_contexts: bool = True
    detect_regimes: bool = True
    mechanism_clustering: MechanismClusteringName = "edge-strengths"
    mechanism_test_alpha: float = 0.05

    standardize: Literal["none", "global", "per_context"] = "per_context"
    fill_method: Literal["none", "ffill_bfill", "interpolate"] = "ffill_bfill"

    output_name: str | None = None


@dataclass
class PosthocTables:
    mechanism_scores_global: pd.DataFrame | None = None
    mechanism_scores_windows: pd.DataFrame | None = None
    edge_contributions_global: pd.DataFrame | None = None
    edge_contributions_windows: pd.DataFrame | None = None

    def nonempty(self) -> dict[str, pd.DataFrame]:
        tables = {
            "mechanism_scores_global": self.mechanism_scores_global,
            "mechanism_scores_windows": self.mechanism_scores_windows,
            "edge_contributions_global": self.edge_contributions_global,
            "edge_contributions_windows": self.edge_contributions_windows,
        }
        return {name: table for name, table in tables.items() if table is not None and not table.empty}


@dataclass
class SpaceTimeExperimentRun:
    dataset: PanelDataset
    config: SpaceTimeExperimentConfig
    estimator: CausalChange
    graph: nx.DiGraph
    changepoints: list[int]
    partitions: Any
    posthoc: PosthocTables
    output_dir: Path | None = None
