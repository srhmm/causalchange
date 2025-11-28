# tests/utils_test_causal_change.py
from src.causalchange.causal_change import CausalChange
from tests.utils_fake_edge_memoized import FakeEdgeMemoized


class TestableCausalChange(CausalChange):
    """
    CausalChange subclass that uses FakeEdgeMemoized instead of the real EdgeMemoized.

    Use this in algorithmic tests where you want deterministic behavior and
    no dependency on the actual GP/MMD/RESIT scoring.
    """

    def initialize(self):
        # X is already set (D, N) by init_and_check_X
        assert self.X is not None
        self.edges_state = FakeEdgeMemoized(
            self.X,
            self.data_mode,
            self.score_type,
            self.mixing_type,
            **self.get_scoring_params(),
        )
