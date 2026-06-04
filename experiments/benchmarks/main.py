import json
import os
from dataclasses import asdict

import experiments.benchmarks.benchmark_grids
from experiments.benchmarks.run_methods import iter_valid_configs, run_on_config
from experiments.benchmarks.utils import (
    bench_name_from_cfg,
    config_group_key,
    file_name_from_cfg,
    mean_std,
    summarize_groups,
    to_json_safe,
)

BASE_SEED = 42
N_REPEATS = 10
BENCHMARK_GRID = (
    experiments.benchmarks.benchmark_grids.BENCHMARK_GRID_SPACETIME
)  # benchmarks.benchmark_grids.BENCHMARK_GRID_MULTI

if __name__ == "__main__":
    groups = {}
    n_runs = 0
    n_valid = 0

    last_cfg = None

    for cfg0 in iter_valid_configs(BENCHMARK_GRID):
        last_cfg = cfg0
        n_valid += 1
        local = {}

        for r in range(N_REPEATS):
            seed = BASE_SEED + r
            d = cfg0.model_dump()
            d["data"]["seed"] = seed
            cfg = cfg0.__class__.model_validate(d)

            metrics = run_on_config(cfg)
            n_runs += 1

            key = config_group_key(cfg)

            if key not in groups:
                example = cfg.model_dump()
                example["data"].pop("seed", None)
                bench = bench_name_from_cfg(cfg)

                groups[key] = {"config_example": example, "bench": bench, "metrics": {}}

            for metric_name, value in metrics.items():
                fv = float(value)
                groups[key]["metrics"].setdefault(metric_name, []).append(fv)
                local.setdefault(metric_name, []).append(fv)

        bench = bench_name_from_cfg(cfg0)
        algo_name = cfg0.algo.name
        data_setting = cfg0.data.setting
        data_nonlinearity = cfg0.data.nonlinearity

        header = f"[{bench}] algo={algo_name}  data={data_setting}/{data_nonlinearity} " f"n_n={cfg0.data.n_nodes}"
        print("\n" + header + str(cfg0))

        for metric_name in sorted(local.keys()):
            m, s = mean_std(local[metric_name])
            print(f"  {metric_name:10s} = {m:.4f} ± {s:.4f} (n={len(local[metric_name])})")

    rows = summarize_groups(groups)

    print("\n\n================ BENCHMARK SUMMARY ================")
    for rw in rows:
        print(f"{rw.bench:28s} | {rw.metric:10s} = {rw.mean:.4f} ± {rw.std:.4f} (n={rw.n})")

    out_dir = os.path.join("../benchmarks", "_results")
    os.makedirs(out_dir, exist_ok=True)

    if last_cfg is None:
        raise RuntimeError(
            "No valid benchmark configs were generated. "
            "Check BENCHMARK_GRID against BenchmarkConfig/DataConfig/AlgoConfig fields."
        )

    filenm = file_name_from_cfg(last_cfg)
    out_path = os.path.join(out_dir, f"{filenm}.json")

    with open(out_path, "w", encoding="utf-8") as fl:
        json.dump([to_json_safe(asdict(rw)) for rw in rows], fl, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Valid configs run: {n_valid}, total runs: {n_runs}")
