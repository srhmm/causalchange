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

    def __eq__(self, other):
        return self.value == other.value


class GraphSearch(Enum):
    TOPIC = "topological"
    GLOBE = "edge-greedy"
    SKIP = "skip"

    def __eq__(self, other):
        return self.value == other.value

    def compatible_modes(self) -> list[DataMode]:
        return (
            [
                DataMode.IID,
                DataMode.CONTEXTS,
                DataMode.MIXED,
                DataMode.TIME,
                DataMode.TIME_CONTEXTS,
            ]
            if self.value == GraphSearch.TOPIC.value
            else (
                [DataMode.IID, DataMode.CONTEXTS]
                if self.value == GraphSearch.GLOBE.value
                else []
            )
        )

    def is_compatible_with(self, data_mode: DataMode) -> bool:
        return data_mode in self.compatible_modes()


class ContextAggregation(Enum):
    SKIP = "skip"
    CHAIN = "chain"
    LINC = "linc"

    def compatible_modes(self) -> list[DataMode]:
        return (
            [DataMode.IID, DataMode.MIXED, DataMode.TIME]
            if self.value == ContextAggregation.SKIP.value
            else (
                [DataMode.TIME_CONTEXTS, DataMode.CONTEXTS]
                if self.value
                in [ContextAggregation.CHAIN.value, ContextAggregation.LINC.value]
                else []
            )
        )

    def is_compatible_with(self, data_mode):
        return data_mode in self.compatible_modes()

    def __eq__(self, other):
        return self.value == other.value


class GPType(Enum):
    EXACT = "gp"
    FOURIER = "ff"

    def __eq__(self, other):
        return self.value == other.value


class MixingType(Enum):
    # mixtures of regressions
    MIX_LIN = "mixLin"
    MIX_QUAD = "mixQuad"
    MIX_CUB = "mixCub"
    MIX_NS = "mixNS"
    MIX_BS = "mixBS"

    SKIP = ""

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return str(self.value)


class ScoreType(Enum):
    LIN = "lin"
    GAM = "gam"
    SPLINE = "spline"
    KRR = "krr"
    GP = GPType
    # CI = CIType
    MIX = MixingType
    SKIP = "skip"

    def higher_is_better(self) -> bool:
        return False  # all mdl scores rn

    def get_gain_threshold(self, n: int) -> float:
        return 0.5 * np.log(n)

    def __eq__(self, other):
        return self.value == other.value


def util_score_type_get_all():
    variants = []
    for st in ScoreType:
        if isinstance(st.value, EnumMeta):
            for sub in st.value:
                variants.append(sub)
        else:
            variants.append(st)
    return variants
