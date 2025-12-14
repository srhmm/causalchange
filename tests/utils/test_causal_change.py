# tests/utils_test_causal_change.py
from src.causalchange.causal_change import CausalChange
from tests.utils.fake_edge_memoized import FakeEdgeMemoized


class TestableCausalChange(CausalChange):

    def initialize(self):
        assert self.X is not None
        self.edges_state = FakeEdgeMemoized(
            self.X,
            self.data_mode,
            self.score_type,
            self.mixing_type,
            **self.get_scoring_params(),
        )
