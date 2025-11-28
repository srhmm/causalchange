from enum import Enum


class IvType(Enum):
    COEF = 'coef'
    SHIFT = 'shift'
    HARD = 'hard'
    MIX = 'mix'
    def __eq__(self, other): return self.value == getattr(other, "value", other)

class FunType(Enum):
    LIN = 'lin'
    QUAD = 'quad'
    CUB = 'cub'
    EXP = 'exp'
    LOG = 'log'
    SIN = 'sin'
    MIX = 'mix'
    def __eq__(self, other): return self.value == getattr(other, "value", other)

class NoiseType(Enum):
    GAUSS = 'normal'
    EXP = 'exp'
    GUMBEL = 'gumbel'
    UNIF = 'unif'
    MIX = 'mix'
    def __eq__(self, other): return self.value == getattr(other, "value", other)
