

class FakeEdgeMemoized:
    def __init__(self, X, data_mode, score_type, mixing_type, **kwargs):
        self.X = X
        self.data_mode = data_mode
        self.score_type = score_type
        self.mixing_type = mixing_type

    def score_edge(self, child, parents):
        score = 100.0 * len(parents) + float(child)
        res = {"fake_score": score, "parents": list(parents), "child": child}
        return score, res

    def discrepancy(self, child, parents):
        base = float(child) + 1.0
        val = base - float(len(parents))
        res = {"fake_discrepancy": val, "parents": list(parents), "child": child}
        return val, res
