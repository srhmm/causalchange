from __future__ import annotations

from typing import Hashable
import pandas as pd


class SingleContextProvider:

    def make_contexts(self, X0: pd.DataFrame) -> dict[Hashable, pd.DataFrame]:
        return {0: X0}
