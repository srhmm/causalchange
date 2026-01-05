from enum import Enum, EnumMeta

class DataMode(Enum):
    IID = 'single'
    CONTEXTS = 'multi'
    TIME = 'time'
    TIME_CONTEXTS = 'time-contexts'
    CONFOUNDED = 'confounded'
    MIXED = 'mixed'
    def is_dict_like(self):
        return self.value in [DataMode.CONTEXTS.value, DataMode.TIME_CONTEXTS.value]
    def is_temporal(self):
        return self.value in [DataMode.TIME.value, DataMode.TIME_CONTEXTS.value]
    def __eq__(self, other):
        return self.value == other.value

class GraphSearch(Enum):
    CHAIN = 'chain'
    TOPIC = 'topological'
    GLOBE = 'edge-greedy'

    def __eq__(self, other):
        return self.value == other.value

    def compatible_modes(self) -> list[DataMode]:
        if self is GraphSearch.TOPIC:
            return [DataMode.IID, DataMode.CONTEXTS, DataMode.MIXED, DataMode.TIME, DataMode.TIME_CONTEXTS]
        elif self is GraphSearch.GLOBE:
            return [DataMode.IID, DataMode.CONTEXTS]
        elif self is GraphSearch.CHAIN:
            return [DataMode.CONTEXTS]
        else:
            return []

    def is_compatible_with(self, data_mode: DataMode) -> bool:
        return data_mode in self.compatible_modes()


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
    KRR = 'krr'
    GP = GPType
    CI = CIType

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
    return variants



class MixingType(Enum):
    # mixtures of regressions
    MIX_LIN = 'mixLin'
    MIX_QUAD = 'mixQuad'
    MIX_CUB = 'mixCub'
    MIX_NS = 'mixNS'
    MIX_BS = 'mixBS'

    SKIP = ''

    def __eq__(self, other): return self.value == other.value
    def __str__(self): return str(self.value)
    def search_each_node(self): return not self.value.endswith('global')

    def is_unconditional_mixture(self): return self.value.startswith('clus')

