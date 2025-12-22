from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional
import json
import os

import numpy as np
import pytest

from pgmpy.benchmarks.synthetic.generators import sample_nonlinear_additive


def pytest_configure(config):
    config.addinivalue_line("markers", "bench: benchmark-style tests (timings/metrics, light asserts only)")
    config.addinivalue_line("markers", "metrics: benchmarks that compute graph metrics like SHD")

@dataclass
class BenchRow:
    bench: str
    scoring_method: str
    metric: str
    mean: float
    std: float
    n: int


class BenchRecorder:
    def __init__(self):
        self.rows: List[BenchRow] = []

    def add_series(self, *, bench: str, scoring_method: str, metric: str, values: list[float]):
        arr = np.asarray(values, dtype=float)
        self.rows.append(
            BenchRow(
                bench=str(bench),
                scoring_method=str(scoring_method),
                metric=str(metric),
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
    config.addinivalue_line("markers", "bench: benchmark-style tests (timings/metrics, light asserts only)")


def pytest_sessionfinish(session, exitstatus):
    rec: Optional[BenchRecorder] = getattr(session.config, "_benchrec", None)
    if rec is None or not rec.rows:
        return

    print("\n\n================ BENCHMARK SUMMARY ================")
    rows = sorted(rec.rows, key=lambda r: (r.bench, r.scoring_method, r.metric))
    for r in rows:
        print(f"{r.bench:18s} | {r.scoring_method:6s} | {r.metric:12s} = {r.mean:.4f} ± {r.std:.4f} (n={r.n})")

    out_dir = os.path.join("../benchmarks", "_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    print(f"Saved: {out_path}")


@pytest.fixture(scope="session")
def fake_data():
    n_samples = int(os.environ.get("BENCH_NSAMPLES", "1000"))
    n_nodes = int(os.environ.get("BENCH_NNODES", "5"))
    edge_prob = float(os.environ.get("BENCH_EDGE_PROB", "0.4"))
    seed = int(os.environ.get("BENCH_SEED", "42"))

    df, true_g = sample_nonlinear_additive(
        n_samples=n_samples,
        n_nodes=n_nodes,
        edge_prob=edge_prob,
        seed=seed,
    )
    return df
