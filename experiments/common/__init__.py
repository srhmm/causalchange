"""Shared utilities for real-world CausalChange/SpaceTime experiments.

Dataset-specific code should live in ``experiments.flux`` and
``experiments.river``. This package contains reusable path handling,
preprocessing, SpaceTime execution, post-hoc scoring, and output helpers.
"""

from experiments.common.data_types import (
    PanelDataset,
    PosthocTables,
    SpaceTimeExperimentConfig,
    SpaceTimeExperimentRun,
)

__all__ = [
    "PanelDataset",
    "PosthocTables",
    "SpaceTimeExperimentConfig",
    "SpaceTimeExperimentRun",
]
