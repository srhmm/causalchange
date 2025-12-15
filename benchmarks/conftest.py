# benchmarks/conftest.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional
import json
import os
import numpy as np
import pytest


@dataclass
class BenchRow:
    bench: str
    data_mode: str
    graph_search: str
    score_type: str
    fun_type: str
    iv_type: str
    metric: str
    mean: float
    std: float
    n: int


class BenchRecorder:
    def __init__(self):
        self.rows: List[BenchRow] = []

    def add_series(
        self,
        bench: str,
        data_mode: str,
        graph_search: str,
        score_type: str,
        fun_type: str,
        iv_type: str,
        metric: str,
        values: list[float],
    ):
        arr = np.asarray(values, dtype=float)
        self.rows.append(
            BenchRow(
                bench=bench,
                data_mode=data_mode,
                graph_search=graph_search,
                score_type=score_type,
                fun_type=fun_type,
                iv_type=iv_type,
                metric=metric,
                mean=float(arr.mean()) if arr.size else float("nan"),
                std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                n=int(arr.size),
            )
        )


@pytest.fixture(scope="session")
def benchrec(pytestconfig) -> BenchRecorder:
    rec = getattr(pytestconfig, "_benchrec", None)
    if rec is None:
        rec = BenchRecorder()
        pytestconfig._benchrec = rec
    return rec


def pytest_configure(config):
    config.addinivalue_line("markers", "bench: benchmark-style tests (report metrics, no quality asserts)")


def pytest_sessionfinish(session, exitstatus):
    rec: Optional[BenchRecorder] = getattr(session.config, "_benchrec", None)
    if rec is None or not rec.rows:
        return

    print("\n\n================ BENCHMARK SUMMARY ================")
    rows = sorted(rec.rows, key=lambda r: (r.bench, r.data_mode, r.graph_search, r.score_type, r.metric))
    for r in rows:
        print(
            f"{r.bench:12s} | {r.data_mode:14s} | {r.graph_search:12s} | {r.score_type:12s} | "
            f"{r.metric:14s} = {r.mean:.3f} ± {r.std:.3f} (n={r.n})"
        )

    out_dir = os.path.join("benchmarks", "_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    print(f"Saved: {out_path}")
