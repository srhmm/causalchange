from enum import Enum


class MethodType(Enum):
    '''
    Methods considered in the evaluation

    '''
    ORACLE = 0 #partition knowledge (and dag not relevant)
    ORACLE_DAG = 1  # dag knowledge, thus easier than COCO
    COCO = 2
    FCI_JCI = 3  # mec knowledge, thus comparable to COCO
    FCI_JCI_FULL = 4  # no knowledge
    FCI_POOLED = 5
    FCI_POOLED_FULL = 6
    FCI_CONTEXT = 7
    FCI_CONTEXT_FULL = 8
    MSS = 9
    def is_coco(self):
        return(self.value < 3)
    def is_fci(self):
        return(self.value > 2 and self.value <9)


def all_methods():
        return [
            MethodType.ORACLE,
            MethodType.ORACLE_DAG,
            MethodType.COCO,
            MethodType.FCI_JCI,
            MethodType.FCI_POOLED,
            MethodType.FCI_CONTEXT
        ]

def all_coco_methods():
    return [
        MethodType.ORACLE,
        MethodType.ORACLE_DAG,
        MethodType.COCO
    ]

def all_fci_methods():
        return [
            MethodType.FCI_JCI, MethodType.FCI_JCI_FULL,
            MethodType.FCI_POOLED, MethodType.FCI_POOLED_FULL,
            MethodType.FCI_CONTEXT, MethodType.FCI_CONTEXT_FULL
        ]

