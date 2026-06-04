from __future__ import annotations

import pandas as pd

from causalchange.core.results import TabularResult
from causalchange.core.types import DataMode, PostprocessingMode
from causalchange.discovery.context_combination import SkipCombination
from causalchange.discovery.graph_tabular import GraphSearchTabularTopological
from causalchange.domain.context import SingleContextDomain
from causalchange.domain.tabular import TabularDomain
from causalchange.engines.tabular import TabularDiscoveryEngine


class SyntheticTabularScoring:
    """Deterministic local scores for x -> y.

    Lower scores are better. The only useful parent relation is x in Pa(y).
    """

    @property
    def higher_is_better(self) -> bool:
        return False

    def fit(self, X0: pd.DataFrame) -> None:
        self.X0 = X0

    def local_score(self, df: pd.DataFrame, effect: str, parents) -> float:
        parents = set(parents)
        if effect == "y" and "x" in parents:
            return 0.0
        return 10.0

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return old_score - new_score

    def gain_is_better(self, a: float, b: float) -> bool:
        return a > b

    def raw_score_is_better(self, a: float, b: float) -> bool:
        return a < b

    def score_significant(self, gain: float) -> bool:
        return gain > 1.0


def test_tabular_engine_synthetic_recovers_expected_edge_and_postprocessing():
    scoring = SyntheticTabularScoring()

    engine = TabularDiscoveryEngine(
        data_mode=DataMode.TABULAR,
        domain=TabularDomain(),
        context_preproc=SingleContextDomain(),
        scoring=scoring,
        context_comb=SkipCombination(),
        search=GraphSearchTabularTopological(scoring=scoring),
        postprocessing_mode=PostprocessingMode.EDGE_STRENGTHS,
    )

    X = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 2.0, 4.0, 6.0],
            "z": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = engine.fit(X).discover()

    assert isinstance(result, TabularResult)
    assert ("x", "y") in result.graph.edges
    assert ("y", "x") not in result.graph.edges
    assert result.edge_strengths[("x", "y")] == 10.0
    assert result.postprocessing.diagnostics["edge_strengths"]["n_edges"] >= 1
