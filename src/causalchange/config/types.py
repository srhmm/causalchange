from enum import Enum, EnumMeta

import numpy as np


class DataMode(Enum):
    SKIP = "skip"
    IID = "single"
    CONTEXTS = "multi"
    TIME = "time"
    TIME_CONTEXTS = "time-contexts"
    MIXED = "mixed"

    def is_context(self):
        return self.value in [DataMode.CONTEXTS.value, DataMode.TIME_CONTEXTS.value]

    def is_temporal(self):
        return self.value in [DataMode.TIME.value, DataMode.TIME_CONTEXTS.value]


class GraphSearch(Enum):
    TOPIC = "topological"
    GLOBE = "edge-greedy"
    SKIP = "skip"

    def compatible_modes(self) -> list[DataMode]:
        if self == GraphSearch.TOPIC:
            return [
                DataMode.IID,
                DataMode.CONTEXTS,
                # once implemented:
                # DataMode.MIXED,
                # DataMode.TIME,
                # DataMode.TIME_CONTEXTS,
            ]

        if self == GraphSearch.GLOBE:
            return [
                # once implemented:
                # DataMode.IID,
                # DataMode.CONTEXTS,
                DataMode.TIME,
                DataMode.TIME_CONTEXTS,
            ]

        return []

    def is_compatible_with(self, data_mode: DataMode) -> bool:
        return data_mode in self.compatible_modes()

class ChangepointMode(Enum):
    NONE = "none"
    FIXED = "fixed"
    DETECT = "detect"


class ChangepointScope(Enum):
    GLOBAL = "global"
    PER_CONTEXT = "per-context"


class ChangepointMethod(Enum):
    PELT = "pelt"


class PartitioningMethod(Enum):
    KERNEL = "kernel"
    NONE = "none"


class ContextMode(Enum):
    SKIP = "skip"
    CHAIN = "chain"
    LINC = "linc"

    def compatible_modes(self) -> list[DataMode]:
        if self == ContextMode.SKIP:
            return [
                DataMode.IID,
                DataMode.MIXED,
                DataMode.TIME,
                DataMode.TIME_CONTEXTS,
            ]

        if self in (ContextMode.CHAIN, ContextMode.LINC):
            return [DataMode.CONTEXTS]

        return []

    def is_compatible_with(self, data_mode):
        return data_mode in self.compatible_modes()

class MDLScoreMixin:
    def higher_is_better(self) -> bool:
        return False

    def get_gain_threshold(self, n: int) -> float:
        return 0.5 * np.log(max(n, 2))



class GPType(MDLScoreMixin, Enum):
    EXACT = "gp"
    FOURIER = "ff"


class MixingType(MDLScoreMixin, Enum):
    MIX_LIN = "mixLin"
    MIX_QUAD = "mixQuad"
    MIX_CUB = "mixCub"
    MIX_NS = "mixNS"
    MIX_BS = "mixBS"

    SKIP = ""

    def __str__(self):
        return str(self.value)


class ScoreType(MDLScoreMixin, Enum):
    LIN = "lin"
    GAM = "gam"
    SPLINE = "spline"
    KRR = "krr"
    GP = GPType
    MIX = MixingType
    SKIP = "skip"


def util_score_type_get_all():
    variants = []
    for st in ScoreType:
        if isinstance(st.value, EnumMeta):
            for sub in st.value:
                variants.append(sub)
        else:
            variants.append(st)
    return variants
