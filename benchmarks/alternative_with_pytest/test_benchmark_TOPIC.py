from __future__ import annotations

import os
import time

import pytest
import networkx as nx

from pgmpy.causal_discovery.score_based import TOPIC

from pgmpy.benchmarks.synthetic.generators import sample_linear_gaussian, sample_nonlinear_additive
from pgmpy.benchmarks.synthetic.metrics import compute_metrics


@pytest.mark.bench
@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
def test_bench_fit_smoke(fake_data, scoring_method, benchrec):
    est = TOPIC(scoring_method=scoring_method)

    n_repeats = int(os.environ.get("BENCH_REPEATS", "3"))
    timings = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        dag = est.fit(fake_data)
        t1 = time.perf_counter()
        assert dag is not None
        timings.append(t1 - t0)

    benchrec.add_series(bench="topic_fit", scoring_method=scoring_method, metric="time_s", values=timings)

@pytest.mark.bench
@pytest.mark.metrics
@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
@pytest.mark.parametrize("generator", ["linear", "nonlinear"])
def test_bench_fit_with_metrics(scoring_method, generator, benchrec):
    n_samples = 1000
    n_nodes = 5
    edge_prob = 0.4
    seed = 42
    n_repeats = 20

    shds, edge_f1s, times = [], [], []

    for r in range(n_repeats):
        if generator == "linear":
            df, true_g = sample_linear_gaussian(
                n_samples=n_samples, n_nodes=n_nodes, edge_prob=edge_prob, seed=seed + r
            )
        else:
            df, true_g = sample_nonlinear_additive(
                n_samples=n_samples, n_nodes=n_nodes, edge_prob=edge_prob, seed=seed + r,
                nonlinearity="tanh",
            )

        est = TOPIC(scoring_method=scoring_method)
        t0 = time.perf_counter()
        res = est.fit(df)
        t1 = time.perf_counter()

        dag = res

        est_nx = nx.DiGraph()
        est_nx.add_nodes_from([str(n) for n in dag.nodes()])
        est_nx.add_edges_from([(str(u), str(v)) for (u, v) in dag.edges()])

        m = compute_metrics(true_g, est_nx)
        shds.append(float(m.shd))
        edge_f1s.append(float(m.edge_f1))
        times.append(t1 - t0)

    bench_name = f"topic_fit_{generator}"
    benchrec.add_series(bench=bench_name, scoring_method=scoring_method, metric="shd", values=shds)
    benchrec.add_series(bench=bench_name, scoring_method=scoring_method, metric="edge_f1", values=edge_f1s)
