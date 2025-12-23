import json
import os
import math
from typing import Any
from dataclasses import asdict

from benchmarks.utils import config_group_key, bench_name_from_cfg, summarize_groups, mean_std
from benchmarks.run_methods import iter_valid_configs, run_on_config



if __name__ == "__main__":
    BASE_SEED = 42
    N_REPEATS = 10

    groups: dict[tuple[tuple[str, object], ...], dict[str, Any]] = {}

    n_runs = 0
    n_valid = 0

    from benchmarks.benchmark_grid import BENCHMARK_GRID

    for cfg0 in iter_valid_configs(BENCHMARK_GRID):
        n_valid += 1

        local: dict[str, list[float]] = {}

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
                scoring_method = getattr(cfg.algo, "scoring_method", "")
                bench = bench_name_from_cfg(cfg)
                bench = f"{bench} | {scoring_method}" if scoring_method else bench

                groups[key] = {
                    "config_example": example,
                    "bench": bench,
                    "metrics": {},
                }

            for metric_name, value in metrics.items():
                fv = float(value)
                groups[key]["metrics"].setdefault(metric_name, []).append(fv)
                local.setdefault(metric_name, []).append(fv)

        bench = bench_name_from_cfg(cfg0)
        algo_name = cfg0.algo.name
        scoring_method = getattr(cfg0.algo, "scoring_method", "")
        data_setting = cfg0.data.setting
        data_nonlinearity = cfg0.data.nonlinearity

        header = f"[{bench}] algo={algo_name} scoring={scoring_method} data={data_setting}/{data_nonlinearity} n_n={cfg0.data.n_nodes}"
        print("\n" + header + str(cfg0))

        for metric_name in sorted(local.keys()):
            m, s = mean_std(local[metric_name])
            print(f"  {metric_name:10s} = {m:.4f} ± {s:.4f} (n={len(local[metric_name])})")

    rows = summarize_groups(groups)

    print("\n\n================ BENCHMARK SUMMARY ================")
    for r in rows:
        print(f"{r.bench:28s} | {r.metric:10s} = {r.mean:.4f} ± {r.std:.4f} (n={r.n})")

    out_dir = os.path.join("../benchmarks", "_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latest.json")

    #with open(out_path, "w", encoding="utf-8") as f:
    #    json.dump([asdict(r) for r in rows], f, indent=2)

    #print(f"Saved: {out_path}")
    print(f"Valid configs run: {n_valid}, total runs: {n_runs}")
