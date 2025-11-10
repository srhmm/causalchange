from enum import Enum


class CoCoTestType(Enum):
    ''' Test for confounding between pairs of variables
    '''
    SKIP = -1
    MI_ZTEST = 0
    AMI_ZTEST = 1
    VI_ZTEST = 2
    MI = 3
    AMI = 4
    VI = 5

    def __eq__(self, other):
        return self.value==other.value

class CoDAGType(Enum):
    ''' Method for discovering a causal DAG
    '''
    SKIP = 0 # oracle
    MSS = 1

    def __eq__(self, other):
        return self.value==other.value

class CoShiftTestType(Enum):
    ''' Test for mechanism shifts between pairs of contexts
    '''
    SKIP = 0
    VARIO = 1
    VARIO_GREEDY = 2
    LINC = 3
    PI_KCI = 4

    def __eq__(self, other):
        return self.value==other.value