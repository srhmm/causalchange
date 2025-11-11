from enum import Enum, EnumMeta

import numpy as np
from numpy._typing import NDArray

XArray = NDArray[np.number]
XType = XArray | dict[int, XArray]
#ScoreRes = dict[str, Any]


class GPType(Enum):
    EXACT = 'gp'
    FOURIER = 'ff'

    def __eq__(self, other):
        return self.value == other.value

    def is_scorebased(self): return True
    def is_constraintbased(self): return False

class CIType(Enum):
    KCI = 'kci'

    def __eq__(self, other):
        return self.value == other.value

    def is_scorebased(self): return False
    def is_constraintbased(self): return True

class ScoreType(Enum):
    LIN = 'lin'
    GAM = 'gam'
    SPLINE = 'spline'
    GP = GPType
    CI = CIType

    def get_model(self, **kwargs):
        raise ValueError(f"Only valid for ScoreType.GP, not {self.value}")

    def is_scorebased(self):
        return not self.value is CIType

    def is_constraintbased(self):
        return self.value is CIType

    def __eq__(self, other):
        return self.value == other.value

def score_type_get_all():
    variants = []
    for st in ScoreType:
        if isinstance(st.value, EnumMeta):
            for sub in st.value:
                variants.append(sub)
        else: variants.append(st)
            #variants.append((st, None))
    return variants




class DataMode(Enum):
    IID = 'iid'
    CONTEXTS = 'contexts'
    TIME = 'time'
    TIME_CONTEXTS = 'time-contexts'
    CONFOUNDED = 'confounded'
    MIXED = 'mixed'
    def is_dict_like(self):
        return self.value in [DataMode.CONTEXTS.value, DataMode.TIME_CONTEXTS.value]

    def __eq__(self, other):
        return self.value == other.value

class GraphSearch(Enum):
    TOPIC = 'topological'
    GLOBE = 'edge-greedy'

    def __eq__(self, other):
        return self.value == other.value

