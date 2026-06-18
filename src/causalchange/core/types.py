from enum import Enum

import numpy as np

# %% Supp input data types


class DataMode(str, Enum):
    SKIP = "skip"
    TABULAR = "tabular-single"
    TAB_CONTEXTS = "tabular-contexts"
    TIME = "time"
    TIME_CONTEXTS = "time-contexts"

    def is_context(self):
        """whether the data has contexts, i.e. is a set of datasets {d0, ... dm}, otherwise its a single d0"""
        return self.value in [DataMode.TAB_CONTEXTS.value, DataMode.TIME_CONTEXTS.value]

    def is_temporal(self):
        """whether the data is time series, otherwise it is tabular"""
        return self.value in [DataMode.TIME.value, DataMode.TIME_CONTEXTS.value]


# %% TIME, TIME_CONTEXTS


class ChangepointMode(str, Enum):
    SKIP = "skip"
    ORACLE = "fixed"
    DETECT = "detect"


class ChangepointScope(str, Enum):
    SKIP = "skip"
    GLOBAL = "global"
    PER_CONTEXT = "per-context"


class ChangepointMethod(str, Enum):
    SKIP = "skip"
    PELT = "pelt"


class MechanismClusteringScope(str, Enum):  # or -Partitioning
    SKIP = "skip"
    REGIMES = "regimes"
    CONTEXTS = "contexts"
    REGIMES_CONTEXTS = "regimes-contexts"

    def detects_contexts(self) -> bool:
        return self.value in [MechanismClusteringScope.CONTEXTS.value, MechanismClusteringScope.REGIMES_CONTEXTS.value]

    def detects_regimes(self) -> bool:
        return self.value in [MechanismClusteringScope.REGIMES.value, MechanismClusteringScope.REGIMES_CONTEXTS.value]


class MechanismClusteringMethod(str, Enum):  # or -Partitioning
    SKIP = "skip"
    TESTING = "statistical-testing"
    CLUSTERING = "mechanism-clustering"


# %% Contexts, Mixed


# check necessary
class TabularContextMode(str, Enum):
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


class TabularContextMethod(str, Enum):
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


# sample missingness
class MissingMode(str, Enum):
    OBSERVED = "observed"
    MISSING = "missing"


# %% Search


class GraphSearch(str, Enum):
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


class PostprocessingMode(str, Enum):
    SKIP = "skip"
    EDGE_STRENGTHS = "edge-strengths"


# %% Scoring, regression, stat testing
class LowerIsBetterScoreMixin:
    @property
    def higher_is_better(self) -> bool:
        return False

    def gain_threshold(self, n: int) -> float:
        # One BIC/MDL parameter penalty in bits.
        return 0.5 * np.log(max(n, 2)) / np.log(2.0)

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


class MixedSCMType(LowerIsBetterScoreMixin, Enum):
    LIN = "lin"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    N_SPLINE = "nspline"
    B_SPLINE = "bspline"

    SKIP = ""

    def __str__(self):
        return str(self.value)


class ScoreType(LowerIsBetterScoreMixin, str, Enum):
    LIN = "lin"
    GAM = "gam"
    SPLINE = "spline"
    KRR = "krr"
    GP = "gp"
    FF = "ff"
    SKIP = "skip"


class StatisticalTestingMethod(str, Enum):
    SKIP = "skip"
    KERNEL = "kernel"
    NONE = "none"


class ContextCombinationKwargs(str, Enum):
    SKIP = "skip"
    AGGLOMERATIVE = "agglomerative"
    COMPONENTS = "components"


# todo consider per Algo? or config per tabular/temporal?
# @dataclass(frozen=True)
# class LINCKwargs:
#    method: Literal["components", "agglomerative"] = "components"
#    ...
