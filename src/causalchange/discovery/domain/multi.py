from __future__ import annotations

from collections.abc import Hashable

import pandas as pd


class MultiContextDomain:
    """
    Splits X into contexts using a context column, and drops that column.
    Equivalent to  ContextScoreMixin._init_contexts.
    """

    def __init__(self, *, context_col: str = "context"):
        self.context_col = str(context_col)

    def make_contexts(self, X: pd.DataFrame) -> dict[Hashable, pd.DataFrame]:
        if self.context_col not in X.columns:
            raise ValueError(f"context_col '{self.context_col}' not found")

        out: dict[Hashable, pd.DataFrame] = {}
        for ctx, g in X.groupby(self.context_col, sort=False):
            out[ctx] = g.drop(columns=[self.context_col]).copy()
        return out

    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.context_col not in X.columns:
            raise ValueError(f"context_col '{self.context_col}' not found")
        return X.drop(columns=[self.context_col]).copy()
