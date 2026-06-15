import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from experiments.benchmarks.run_methods import iter_valid_configs, run_on_config
from experiments.benchmarks.utils import (
    bench_name_from_cfg,
    config_group_key,
    summarize_groups,
    to_json_safe, _short_config_label, _fmt,
)
from experiments.benchmarks.benchmark_grids import (
    GRID_TOPIC_SMALL, GRID_TOPIC_MEDIUM, GRID_LINC_NO_CHANGE, GRID_LINC_CONTEXT_CHANGES,
    GRID_SPACETIME_ORACLE_TIME,GRID_SPACETIME_ORACLE_CONTEXTS,GRID_SPACETIME_DETECT_SMALL)
BASE_SEED = 42
N_REPEATS = 10

OUT_DIR = Path(__file__).resolve().parent / "_results"

GRIDS = {
    # TOPIC
    "topic_small": GRID_TOPIC_SMALL,
    "topic_medium": GRID_TOPIC_MEDIUM,

    # LINC
    "linc_no_change": GRID_LINC_NO_CHANGE,
    "linc_context_changes": GRID_LINC_CONTEXT_CHANGES,

    # SpaceTime
    "spacetime_oracle_time": GRID_SPACETIME_ORACLE_TIME,
    "spacetime_oracle_contexts": GRID_SPACETIME_ORACLE_CONTEXTS,
    "spacetime_detect_small": GRID_SPACETIME_DETECT_SMALL,
}

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for BENCHMARK_NAME, BENCHMARK_GRID in GRIDS.items():

        print(f"\nRunning benchmark {BENCHMARK_NAME} ...")

        groups = {}
        n_runs = 0
        n_valid = 0

        valid_configs = list(iter_valid_configs(BENCHMARK_GRID))

        for cfg0 in tqdm(
                valid_configs,
                desc=BENCHMARK_NAME,
                unit="cfg",
                leave=True,
        ):
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

        rows = summarize_groups(groups)

        print(f"\n\n================ RESULTS {BENCHMARK_NAME} ================")

        grouped = defaultdict(dict)
        configs = {}

        for rw in rows:
            label = _short_config_label(rw.config)
            configs[label] = rw.config
            grouped[label][rw.metric] = rw


        header = (
            f"{'config':90s} "
            f"{'edge_f1':>10s} "
            f"{'skel_f1':>10s} "
            f"{'shd':>10s} "
            f"{'time_s':>10s}"
        )
        print(header)
        print("-" * len(header))

        for label in sorted(grouped):
            metrics = grouped[label]

            edge = metrics.get("summary_edge_f1") or metrics.get("edge_f1")
            skel = metrics.get("summary_skel_f1") or metrics.get("skel_f1")
            shd = metrics.get("summary_shd") or metrics.get("shd")
            time_s = metrics.get("time_s")

            print(
                f"{label[:90]:90s} "
                f"{_fmt(edge.mean if edge else None):>10s} "
                f"{_fmt(skel.mean if skel else None):>10s} "
                f"{_fmt(shd.mean if shd else None):>10s} "
                f"{_fmt(time_s.mean if time_s else None):>10s}"
            )

        if not rows: raise RuntimeError( "No valid benchmark configs" )

        out_path = OUT_DIR / f"{BENCHMARK_NAME}.json"

        with open(out_path, "w", encoding="utf-8") as fl:
            json.dump([to_json_safe(asdict(rw)) for rw in rows], fl, indent=2)

        print(f"\nSaved: {out_path}")
        print(f"Valid configs run: {n_valid}, total runs: {n_runs}")
