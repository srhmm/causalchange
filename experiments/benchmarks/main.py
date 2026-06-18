import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from experiments.benchmarks.grids import GRIDS
from experiments.benchmarks.run import iter_valid_configs, run_on_config
from experiments.benchmarks.utils import (
    _fmt,
    _get_config_label,
    bench_name_from_cfg,
    config_group_key,
    summarize_groups,
    to_json_safe,
)

if __name__ == "__main__":
    BASE_SEED = 42
    N_REPEATS = 10
    OUT_DIR = Path(__file__).resolve().parent / "_results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for BENCHMARK_NAME, BENCHMARK_GRID in GRIDS.items():
        print(f"\nRunning benchmark {BENCHMARK_NAME} ...")

        groups = {}
        n_runs = 0

        valid_configs = list(iter_valid_configs(BENCHMARK_GRID))
        n_valid = len(valid_configs)

        for cfg0 in tqdm(valid_configs, desc=BENCHMARK_NAME, unit="cfg", leave=True):
            for r in range(N_REPEATS):
                n_runs += 1
                seed = BASE_SEED + r
                d = cfg0.model_dump()
                d["data"]["seed"] = seed
                cfg = cfg0.__class__.model_validate(d)

                metrics = run_on_config(cfg)
                key = config_group_key(cfg)
                print("\n")
                print(metrics)
                print("\n")
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

        rows = summarize_groups(groups)
        if not rows:
            raise RuntimeError("No valid benchmark configs")

        print(f"\n\n================ RESULTS {BENCHMARK_NAME} ================")

        grouped = defaultdict(dict)
        seen_labels = {}
        for rw in rows:
            label = _get_config_label(rw.config)
            if label in seen_labels and seen_labels[label] != rw.config:
                raise RuntimeError(
                    "Compact benchmark label collision. "
                    f"Label {label!r} maps to multiple configs. "
                    "Add more fields to _short_config_label()."
                )

            seen_labels[label] = rw.config
            grouped[label][rw.metric] = rw

        label_width = max([90] + [len(label) for label in grouped])
        header = (
            f"{'config':{label_width}s} " f"{'edge_f1':>10s} " f"{'skel_f1':>10s} " f"{'nshd':>10s} " f"{'time_s':>10s}"
        )
        print(header)
        print("-" * len(header))

        for label in sorted(grouped):
            metrics = grouped[label]

            edge = metrics.get("summary_edge_f1") or metrics.get("edge_f1")
            skel = metrics.get("summary_skel_f1") or metrics.get("skel_f1")
            # shd = metrics.get("summary_shd") or metrics.get("shd")
            norm_shd = metrics.get("summary_norm_shd") or metrics.get("norm_shd")
            time_s = metrics.get("time_s")
            print(
                f"{label:{label_width}s} "
                f"{_fmt(edge.mean if edge else None):>10s} "
                f"{_fmt(skel.mean if skel else None):>10s} "
                # f"{_fmt(shd.mean if shd else None):>10s} "
                f"{_fmt(norm_shd.mean if norm_shd else None):>10s} "
                f"{_fmt(time_s.mean if time_s else None):>10s}"
            )

        out_path = OUT_DIR / f"{BENCHMARK_NAME}.json"

        with open(out_path, "w", encoding="utf-8") as fl:
            json.dump([to_json_safe(asdict(rw)) for rw in rows], fl, indent=2)

        print(f"\nSaved: {out_path}")
        print(f"Valid configs run: {n_valid}, total runs: {n_runs}")
