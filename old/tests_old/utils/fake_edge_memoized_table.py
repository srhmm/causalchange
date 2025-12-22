
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Tuple, Any

@dataclass
class FakeEdgeMemoizedTable:
    X: Any
    data_mode: Any
    score_type: Any
    mixing_type: Any
    table: Dict[Tuple[int, FrozenSet[int]], float] | None = None
    base: float = 1000.0
    penalty: float = 10.0

    def __post_init__(self):
        if self.table is None:
            self.table = {}

    def _key(self, child: int, parents: Iterable[int]) -> Tuple[int, FrozenSet[int]]:
        return (int(child), frozenset(int(p) for p in parents))

    def score_edge(self, child, parents):
        key = self._key(child, parents)
        score = self.table.get(key, self.base + self.penalty * len(list(parents)) + float(child))
        res = {"fake_score": score, "child": int(child), "parents": list(parents)}
        return float(score), res

    def discrepancy(self, child, parents):
        val = float(child) + 1.0 - float(len(list(parents)))
        res = {"fake_discrepancy": val, "child": int(child), "parents": list(parents)}
        return float(val), res


