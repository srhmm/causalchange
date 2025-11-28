# tests/utils_fake_edge_memoized.py
import numpy as np


class FakeEdgeMemoized:
    """
    Deterministic fake scorer for testing graph search logic.

    Design:
    - score_edge(child, parents):
        score = 100 * len(parents) + child
      -> more parents = worse (higher) score.
    - discrepancy(child, parents):
        base = float(child) + 1.0
        discrep = base - len(parents)
      -> more parents = lower discrepancy.
    """

    def __init__(self, X, data_mode, score_type, mixing_type, **kwargs):
        # X is (D, N): (n_samples, n_nodes)
        self.X = X
        self.data_mode = data_mode
        self.score_type = score_type
        self.mixing_type = mixing_type
        # kwargs ignored; they’re just for signature compatibility

    def score_edge(self, child, parents):
        score = 100.0 * len(parents) + float(child)
        res = {"fake_score": score, "parents": list(parents), "child": child}
        return score, res

    def discrepancy(self, child, parents):
        base = float(child) + 1.0
        val = base - float(len(parents))
        res = {"fake_discrepancy": val, "parents": list(parents), "child": child}
        return val, res
