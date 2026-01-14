from __future__ import annotations

from typing import Hashable
import pandas as pd


class SingleContextDomain:
    def make_contexts(self, X0: pd.DataFrame) -> dict[Hashable, pd.DataFrame]:
        return {0: X0}

    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
