from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common.data_types import SpaceTimeExperimentConfig
from experiments.common.outputs import save_spacetime_run
from experiments.common.paths import ensure_dir, results_dir
from experiments.common.spacetime_runner import fit_spacetime
from experiments.river.config import RiverLoadConfig, river_reference_graph
from experiments.river.load import load_river_data, yearly_panel_dataset


def monthly_changepoints(n_days: int = 365, month_length: int = 30) -> list[int]:
    return [cp for cp in range(month_length, n_days, month_length)]


def run_river_edge_weights(
    *,
    data_root: Path | None = None,
    basins_info_path: Path | None = None,
    out_dir: Path | None = None,
    score_type: str = "ff",
    n_locs: int | None = None,
):
    river_data = load_river_data(
        RiverLoadConfig(
            data_root=data_root,
            basins_info_path=basins_info_path,
            n_locs=n_locs,
        )
    )

    out_dir = ensure_dir(out_dir or results_dir("river", "edge_weights", create=True))

    config = SpaceTimeExperimentConfig(
        score_type=score_type,
        tau_max=3,
        changepoints="fixed",
        fixed_changepoints=tuple(monthly_changepoints()),
        d_min=30,
        max_iter=1,
        detect_contexts=False,
        detect_regimes=False,
        standardize="per_context",
        fill_method="ffill_bfill",
    )
    graph = river_reference_graph(tau_max=config.tau_max)

    runs = {}

    for context_id, years in river_data.yearly.items():
        location_id = river_data.location_ids[context_id]

        for year in years:
            dataset = yearly_panel_dataset(river_data, context_id=context_id, year=year)
            run = fit_spacetime(
                dataset,
                config=config,
                graph_for_posthoc=graph,
                changepoints_for_posthoc=list(config.fixed_changepoints),
                compute_global_scores=True,
                compute_window_scores=True,
            )

            current_out = out_dir / f"{context_id}_{location_id}" / str(year)
            save_spacetime_run(run, current_out)
            runs[(context_id, year)] = run

    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--basins-info-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--score-type", default="ff")
    parser.add_argument("--n-locs", type=int, default=None)
    args = parser.parse_args()

    run_river_edge_weights(
        data_root=args.data_root,
        basins_info_path=args.basins_info_path,
        out_dir=args.out_dir,
        score_type=args.score_type,
        n_locs=args.n_locs,
    )


if __name__ == "__main__":
    main()
