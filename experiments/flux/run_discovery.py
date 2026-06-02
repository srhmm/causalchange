from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common.outputs import save_spacetime_run
from experiments.common.paths import results_dir
from experiments.common.spacetime_runner import fit_spacetime
from experiments.flux.config import FluxLoadConfig, default_flux_spacetime_config, flux_reference_graph
from experiments.flux.load import load_flux_data


def run_flux_discovery(
    *,
    data_root: Path | None = None,
    location_info_path: Path | None = None,
    out_dir: Path | None = None,
    score_type: str = "ff",
    n_locs: int | None = None,
    use_reference_graph_for_posthoc: bool = False,
):
    load_config = FluxLoadConfig(
        data_root=data_root,
        location_info_path=location_info_path,
        n_locs=n_locs,
    )
    flux_data = load_flux_data(load_config)

    config = default_flux_spacetime_config(score_type=score_type)
    graph = flux_reference_graph(tau_max=config.tau_max) if use_reference_graph_for_posthoc else None

    run = fit_spacetime(
        flux_data.selected_year,
        config=config,
        graph_for_posthoc=graph,
        compute_global_scores=True,
        compute_window_scores=True,
    )

    out_dir = out_dir or results_dir("flux", "discovery", create=True)
    save_spacetime_run(run, out_dir)

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--location-info-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--score-type", default="ff")
    parser.add_argument("--n-locs", type=int, default=None)
    parser.add_argument("--use-reference-graph-for-posthoc", action="store_true")
    args = parser.parse_args()

    run_flux_discovery(
        data_root=args.data_root,
        location_info_path=args.location_info_path,
        out_dir=args.out_dir,
        score_type=args.score_type,
        n_locs=args.n_locs,
        use_reference_graph_for_posthoc=args.use_reference_graph_for_posthoc,
    )


if __name__ == "__main__":
    main()
