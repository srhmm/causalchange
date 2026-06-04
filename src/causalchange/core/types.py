from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Literal

import numpy as np

# %% Supp input data types


class DataMode(Enum):
    SKIP = "skip"
    TABULAR = "tabular-single"
    TAB_CONTEXTS = "tabular-contexts"
    TIME = "time-single"
    TIME_CONTEXTS = "time-contexts"

    def is_context(self):
        """whether the data has contexts, i.e. is a set of datasets {d0, ... dm}, otherwise its a single d0"""
        return self.value in [DataMode.TAB_CONTEXTS.value, DataMode.TIME_CONTEXTS.value]

    def is_temporal(self):
        """whether the data is time series, otherwise it is tabular"""
        return self.value in [DataMode.TIME.value, DataMode.TIME_CONTEXTS.value]


# %% TIME, TIME_CONTEXTS


class ChangepointMode(Enum):
    SKIP = "skip"
    ORACLE = "fixed"
    DETECT = "detect"


class ChangepointScope(Enum):
    SKIP = "skip"
    GLOBAL = "global"
    PER_CONTEXT = "per-context"


class ChangepointMethod(Enum):
    SKIP = "skip"
    PELT = "pelt"


class MechanismClusteringScope(Enum):  # or -Partitioning
    SKIP = "skip"
    REGIMES = "regimes"
    CONTEXTS = "contexts"
    REGIMES_CONTEXTS = "regimes-contexts"

    def detects_contexts(self) -> bool:
        return self.value in [MechanismClusteringScope.CONTEXTS.value, MechanismClusteringScope.REGIMES_CONTEXTS.value]

    def detects_regimes(self) -> bool:
        return self.value in [MechanismClusteringScope.REGIMES.value, MechanismClusteringScope.REGIMES_CONTEXTS.value]


class MechanismClusteringMethod(Enum):  # or -Partitioning
    SKIP = "skip"
    TESTING = "statistical-testing"
    CLUSTERING = "mechanism-clustering"


# %% TAB_CONTEXTS, TAB_MIXED


# not strictly necessary
class TabularContextMode(Enum):
    SKIP = "skip"
    ORACLE = "fixed"
    DETECT = "detect"

    def compatible_data_modes(self) -> list[DataMode]:
        if self == TabularContextMode.SKIP:
            return [
                DataMode.TABULAR,
                DataMode.TIME,
                DataMode.TIME_CONTEXTS,
            ]

        if self == TabularContextMode.ORACLE:
            return [DataMode.TAB_CONTEXTS]

        if self == TabularContextMode.DETECT:
            return [DataMode.TABULAR]

        return []

    def is_compatible_with(self, data_mode):
        return data_mode in self.compatible_data_modes()


class TabularContextMethod(Enum):
    SKIP = "skip"
    CHAIN = "chain"
    LINC = "linc"

    def compatible_data_modes(self) -> list[DataMode]:
        if self == TabularContextMethod.SKIP:
            return [
                DataMode.TABULAR,
                DataMode.TIME,
                DataMode.TIME_CONTEXTS,
            ]

        if self in (TabularContextMethod.CHAIN, TabularContextMethod.LINC):
            return [DataMode.TAB_CONTEXTS]

        return []

    def is_compatible_with(self, data_mode):
        return data_mode in self.compatible_data_modes()


# %% Search


class GraphSearch(Enum):
    TOPIC = "topological"
    GLOBE = "edge-greedy"
    SKIP = "skip"

    def compatible_data_modes(self) -> list[DataMode]:
        if self == GraphSearch.TOPIC:
            return [
                DataMode.TABULAR,
                DataMode.TAB_CONTEXTS,
                # once implemented:
                # DataMode.TIME,
                # DataMode.TIME_CONTEXTS,
            ]

        if self == GraphSearch.GLOBE:
            return [
                # once implemented:
                # DataMode.TABULAR,
                # DataMode.TAB_CONTEXTS,
                DataMode.TIME,
                DataMode.TIME_CONTEXTS,
            ]

        return []

    def is_compatible_with(self, data_mode: DataMode) -> bool:
        return data_mode in self.compatible_data_modes()


class PostprocessingMode(Enum):
    SKIP = "skip"
    EDGE_STRENGTHS = "edge-strengths"


# %% Scoring, regression, stat testing
class LowerIsBetterScoreMixin:
    @property
    def higher_is_better(self) -> bool:
        return False

    def gain_threshold(self, n: int) -> float:
        return 0.5 * np.log(max(n, 2))

    def transition_gain(self, old_score: float, new_score: float) -> float:
        # Positive means improvement.
        return old_score - new_score

    def gain_is_better(self, a: float, b: float) -> bool:
        # Gains are normalized: larger is always better.
        return a > b

    def raw_score_is_better(self, a: float, b: float) -> bool:
        # Raw MDL / loss scores: smaller is better.
        return a < b

    def score_significant(self, gain: float, n: int) -> bool:
        return gain > self.gain_threshold(n)


# consider flattening these
class GPType(LowerIsBetterScoreMixin, Enum):
    EXACT = "gp"
    FOURIER = "ff"


class MixedSCMType(LowerIsBetterScoreMixin, Enum):
    LIN = "lin"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    N_SPLINE = "nspline"
    B_SPLINE = "bspline"

    SKIP = ""

    def __str__(self):
        return str(self.value)


class ScoreType(LowerIsBetterScoreMixin, Enum):
    LIN = "lin"
    GAM = "gam"
    SPLINE = "spline"
    KRR = "krr"
    GP = GPType
    MIX = MixedSCMType
    SKIP = "skip"


class StatisticalTestingMethod(Enum):
    SKIP = "skip"
    KERNEL = "kernel"
    NONE = "none"


@dataclass(frozen=True)
class ContextCombinationKwargs:
    method: Literal["components", "agglomerative"] = "components"
    gain_threshold: float = 0.0


def util_score_type_get_all():
    variants = []
    for st in ScoreType:
        if isinstance(st.value, EnumMeta):
            for sub in st.value:
                variants.append(sub)
        else:
            variants.append(st)
    return variants
