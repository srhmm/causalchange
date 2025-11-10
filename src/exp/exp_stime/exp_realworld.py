from __future__ import annotations

import json
import warnings
from types import SimpleNamespace
from typing import Tuple, List

from sklearn.metrics import silhouette_score

import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import pandas as pd
from typing import Dict
import re


from src.exp.exp_stime.data_hypparams import PATH_RIVER_OUT, assumed_max_lag, assumed_min_dur, verbosity, \
    RIVER_RESULT_LINKS, RIVER_ATTRS, \
    PATH_PRE
from src.stime import DiscrepancyTestType
from src.stime import SpaceTime
from src.stime import MethodType, CpsInitializationStrategy

COL_SCHEME = {
    "pr-color1b": "#219ebc",   # blue
    "pr-color1a": "#84af2c",   # green
    "pr-color1d": "#eb794e",   # orange
    "pr-color1c": "#ca8a04",   # golden
    "pr-color1e": "#8ecae6",   # light blue
    "pr-color1f": "#e44a52",   # red
    "pr-color1g": "#86ba71",   # light green
    "pr-color1h": "#ffa600",   # sun
    "pr-color1i": "#005d80",   # teal
    "pr-color1j": "#786447",   # brown
    "pr-color1k": "#D8DE3F",   # lime
    "pr-color1l": "#eb794e",   # orange again
    "pr-color1m": "#84cc16",  # named color
    "pr-color1n": "#8e4988",   # purple
    "pr-color1o": "#61AAC0",   # turquoise
    "pr-color1p": "#f9cb6e",   # orangeyellow
}

CUSTOM_COLOR_LIST = list(COL_SCHEME.values())
for i in range(200): CUSTOM_COLOR_LIST.append("#f1f5f9")

#%% DAG discovery

def causal_discovery_loc(dinfo, out_dir):
    """ Causal discovery jointly over all locations and all months (here, for year 2006 (flux), 2010 (river) or closest year to it)
    @return: spacetime obj
    """
    #dinfo = get_river_data()
    #out_dir = PATH_RIVER_OUT
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    context_dic = {}
    for ci in dinfo['loc_one_year']:
        context_dic[ci] = np.array(dinfo['loc_one_year'][ci][:365])

    print(f'*** EVALUATING  {len(context_dic)} locations ***')
    method = MethodType.ST_GP
    spct = SpaceTime(
        max_lag=assumed_max_lag,
        min_dur=assumed_min_dur,
        scoring_function=method.get_scoring_function(),
        cps_init_strategy=CpsInitializationStrategy.BINS,
        method_type=method,
        logger=dinfo['log'],
        verbosity=verbosity,
        out=out_dir
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spct.run(context_dic)

    print(f'*** SpaceTime RESULT *** '
          f'\nDAG {spct.results.cur_dag_model}\nCPS {spct.results.cur_regimes}'
          f'\nRegimes {spct.results.cur_regimes}\nContexts {spct.results.cur_contexts}'
          f'\nRegimes/node {spct.results.cur_r_each_node}\nContexts/node {spct.results.cur_c_each_node}')
    return spct


#%% Changepoint detection
def cps_search(yr, dinfo, out_dir=PATH_PRE + PATH_RIVER_OUT):
    """CPS search and partitioning given our DAG, jointly over select locations and all months in select years
    @return:
    """
    print(yr)
    links = RIVER_RESULT_LINKS
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    context_dic = {}
    for ci in dinfo['loc_yearly']:
        if yr not in dinfo['loc_yearly'][ci]: continue
        idx = [i for i in range(365)]
        df = np.array(dinfo['loc_yearly'][ci][yr])[idx]
        context_dic[ci] = np.array(df)

    # Detect cutpoints under the true DAG
    truths = SimpleNamespace(true_links=links)
    method = MethodType.ST_GP_DAG
    spct = SpaceTime(
        truths = truths,
        max_lag=assumed_max_lag,
        min_dur=assumed_min_dur,
        scoring_function=method.get_scoring_function(),
        cps_init_strategy=CpsInitializationStrategy.BINS,
        discrepancy_test=DiscrepancyTestType.MDL,
        method_type=method,
        logger=dinfo['log'],
        verbosity=verbosity,
        out=out_dir
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spct.run(context_dic)

    print(f'*** CPS SEARCH RESULT *** \n\tFor {len(context_dic)} locations using windows {spct.result.cur_regimes} ')
    print(f'\tCPS: {spct.result.cur_regimes}\n\tRegimes: {spct.result.cur_partition}\n\tContexts: {spct.result.cur_c_partition}')
    print(f'\tRegimes/node: {spct.result.cur_r_partitions}\n\tContexts/node: {spct.result.cur_c_partitions}')
    yr_cps = dict(
        regimes = spct.result.cur_regimes, partition = spct.result.cur_partition,
        r_partitions = spct.result.cur_r_partitions)
    print(yr_cps)
    #for ci in ci_cps:
    #    reg_order = [p for _, _, p in ci_cps[ci].partition]
    #    diff_reg = np.unique([p for _, _, p in ci_cps[ci].partition])
    #    if len(diff_reg) <= 5:
    #        print(ci, dinfo["loc_ids"][ci],'\t', len(diff_reg), reg_order, '\t',ci_cps[ci].partition)

    return yr_cps


def cps_search_locs(dinfo, MAIN_YEAR, links=RIVER_RESULT_LINKS, out_dir=PATH_PRE + PATH_RIVER_OUT):
    """CPS search and partitioning given our DAG, jointly over select locations and all months in select years
    @return:
    """

    os.makedirs(out_dir+ f'/cps_{MAIN_YEAR}/', exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ci_cps = {}

    # Select locations
    for ci in dinfo['loc_one_year']:
        context_dic = {}
        idx = [i for i in range(365)]
        df = np.array(dinfo['loc_one_year'][ci])[idx]
        context_dic[0] = np.array(df)
        print(f"*** CPS SEARCH *** \n\tLoc {dinfo['loc_ids'][ci]}")

        # Detect cutpoints under the true DAG
        truths = SimpleNamespace(true_links=links)
        method = MethodType.ST_GP_DAG
        spct = SpaceTime(
            truths = truths,
            max_lag=assumed_max_lag,
            min_dur=assumed_min_dur,
            scoring_function=method.get_scoring_function(),
            cps_init_strategy=CpsInitializationStrategy.BINS,
            method_type=method,
            logger=dinfo['log'],
            verbosity=0,
            out=out_dir,
            eval=False
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spct.run(context_dic)
        diff_reg = len(np.unique([p for _, _, p in spct.result.cur_regimes]))
        print(f"\tResult:\n\tchangepts: {len(spct.result.cur_regimes)}, regimes: {diff_reg} ")
        print(f'\tCPS: {spct.result.cur_regimes}\n\tContexts: {spct.result.cur_contexts}')
        print(f'\tRegimes/node: {spct.result.cur_r_each_node}\n\tContexts/node: {spct.result.cur_c_each_node}')
        ci_cps[ci] = dict(
            regimes = spct.result.cur_regimes, regimes_node = spct.result.cur_r_each_node, data=context_dic)

        with open(Path(PATH_PRE + PATH_RIVER_OUT + '/cps_2010/',  f"{dinfo['loc_ids'][ci]}.json"), "w") as fl:
            json.dump(spct.result.cur_regimes, fl, indent=2)
        #READING
        #with open(Path(PATH_PRE + PATH_RIVER_OUT + f"{dinfo['loc_ids'][ci]}.json"), "r") as f:
        #    regimes = json.load(f)
        #    regimes = [tuple(x) for x in regimes]

    #with open(Path(PATH_PRE + PATH_RIVER_OUT + '/cps_2010/',  f"all_cps.json"), "w") as fl:
    #    json.dump(ci_cps, fl, indent=2)

    #for ci in ci_cps:
    #    reg_order = [p for _, _, p in ci_cps[ci]["regimes"]]
    #    diff_reg = np.unique([p for _, _, p in ci_cps[ci]["regimes"]])
    #    if len(diff_reg) <= 5: print(ci, dinfo["loc_ids"][ci],'\t', len(diff_reg), reg_order, '\t',ci_cps[ci]["regimes"])

    return ci_cps


#%% Causal strengths for edge pairs
def mdl_edge_strengths_loc_yr_month(dinfo, out_dir, LINKS, MAIN_YEAR=None, main_year_only=False):
    """ For the discovered DAG, compute edge weights for each location, year, and month'
    @return:
    """
    #dinfo = get_river_data()
    #write_to = os.path.join(PATH_PRE + PATH_RIVER_OUT, f'mdl_loc_mon_{RIVER_MAINYEAR}/') if main_year_only else os.path.join(PATH_PRE + PATH_RIVER_OUT, 'mdl_loc_mon_')
    write_to = os.path.join(out_dir, f'mdl_loc_mon_{MAIN_YEAR}/') if main_year_only else os.path.join(out_dir, 'mdl_loc_mon_all') #"_part"
    Path(write_to).mkdir(parents=True, exist_ok=True)

    links = LINKS
    method = MethodType.ST_GP

    spct = SpaceTime(
        max_lag=assumed_max_lag,
        min_dur=assumed_min_dur,
        scoring_function=method.get_scoring_function(),
        cps_init_strategy=CpsInitializationStrategy.BINS,
        method_type=method,
        logger=dinfo['log'],
        verbosity=verbosity,
        out=write_to, to_file=True, eval=False
    )

    nms = [f'{dinfo["loc_ids"][ci]}' for ci in dinfo["loc_ids"]]
    for ci in dinfo["loc_yearly"]:
        yrs = [MAIN_YEAR] if main_year_only else dinfo["loc_yearly"][ci]
        for yr in yrs:
            prefix = f"{ci}_{yr}"
            #  restart-safe, skip if <ci>_<yr>_0_*.csv exists
            already_done = list((Path(write_to)).glob(f"{prefix}_0_*.csv"))
            if already_done:
                print(f"✔  Skipping loc={nms[ci]} yr={yr}  "
                      f"({len(already_done)} files already present)")
                continue

            print(f"▶  Eval loc={nms[ci]} yr={yr}")


            context_dic = {0: np.array(dinfo["loc_one_year"][ci][:365])} if main_year_only else {0: np.array(dinfo["loc_yearly"][ci][yr][:365])}
            spct.out = os.path.join(write_to, f"{ci}_{yr}")
            spct.result.cur_links = links

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spct.util_score_given_links(context_dic, links)



#%% Cluster by MDL edge strength for one edge
def cluster_locations(
    weights: Dict[str, float],
    k_min: int = 1,
    k_max: int = 10,
    random_state: int = 0,
    use_silhouette: bool = True,
) -> Tuple[Dict[int, List[str]], KMeans, int]:

    loc_ids = list(weights.keys())
    X = np.array(list(weights.values()), dtype=float).reshape(-1, 1)
    n_points = len(X)

    if k_max is None:
        k_max = n_points  # Allow k=N

    best_k = k_min
    best_score = -np.inf
    best_model = None

    for k in range(k_min, min(k_max, n_points) + 1):
        model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = model.fit_predict(X)

        if use_silhouette and 2 <= k <= n_points - 1:
            try:
                score = silhouette_score(X, labels)
            except ValueError:
                score = -1
        else:
            # Use negative inertia to simulate elbow (smaller inertia = better)
            score = -model.inertia_

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    labels = best_model.labels_
    clusters: Dict[int, List[str]] = {}
    for loc, lab in zip(loc_ids, labels):
        clusters.setdefault(lab, []).append(loc)

    return clusters, best_model, best_k

#%% TSNE embedding of edge weights, then cluster by edge strength

import plotly.express as px

def save_interactive_tsne_plot(
    df: pd.DataFrame,
    out_path: str = "tsne_interactive.html",
    color_column: str = "color_value",
):
    """
    Creates an interactive scatter plot using Plotly.
    Requires columns: ['tsne1', 'tsne2', 'loc', 'year', 'month', color_column]
    """
    if color_column not in df.columns:
        raise ValueError(f"Column '{color_column}' not found in DataFrame")

    fig = px.scatter(
        df,
        x="tsne1",
        y="tsne2",
        color=color_column,
        hover_data=["loc", "year", "month", color_column],
        title="Interactive t-SNE Embedding of Link Strengths",
        width=900,
        height=700,
        color_continuous_scale="Viridis",  # can be changed to other scales
    )

    fig.update_layout(template="plotly_white")
    fig.write_html(out_path)
    print(f"[✓] Interactive t-SNE plot saved to: {out_path}")


def run_tsne_all_links(
    *,
    dinfo,
    folder: str | Path,
    nodes: list[str] | np.ndarray,
    allnodes: list[str] | np.ndarray,
    links: list[list[tuple[tuple[int, int], float, float]]],
    years: list[int] | range,
    main_year: int,
    months: list[int] | range = range(12),
    header: int | None = None,
    perplexity: int = 30,
    learning_rate: int = 200,
    n_iter: int = 1_000,
    random_state: int = 42,
    out_path: str | Path = "",
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    One sample = one (location, year, month) vector of *all* link strengths
    """
    import itertools
    from pathlib import Path
    from collections import defaultdict
    import numpy as np
    from sklearn.manifold import TSNE

    folder = Path(folder)
    os.makedirs(out_path, exist_ok=True)

    n_n = len(nodes)

    # causal edges
    edge_pairs = []
    for node_j in range(n_n):
        for (pa, lg), _, _ in links[node_j]:
            node_i = n_n * abs(lg) + pa
            edge_pairs.append((node_i, node_j))
    n_edges = len(edge_pairs)

    # causal edge ws
    sample_vectors = defaultdict(lambda: np.full(n_edges, np.nan, dtype="float32"))
    edge_weight_cache = {}
    for (edge_idx, (node_i, node_j)) in enumerate(edge_pairs):
        for year, month in itertools.product(years, months):
            weights = read_mdl_strengths_loc_mon(
                folder=folder,
                year=year,
                month=month,
                node_i=node_i,
                node_j=node_j,
                header=header,
            )
            for loc_id, w in weights.items():
                sample_vectors[(loc_id, int(year), int(month))][edge_idx] = w
                edge_weight_cache[(loc_id, year, month, node_i, node_j)] = w


    rows, X, color_values = [], [], []

    for key, vec in sample_vectors.items():
        if drop_incomplete and np.isnan(vec).any(): continue
        loc_id, year, month = key
        rows.append((loc_id, year, month))
        X.append(vec)

    X = np.asarray(X, dtype="float32")
    if X.size == 0:
        raise RuntimeError("No complete samples found — try drop_incomplete=False")

    #  t-SNE over edge ws
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=12,
        random_state=random_state,
    )
    Y = tsne.fit_transform(X)

    # clus over tsne
    labels  = cluster_tsne(Y)
    color_values_dict = {}

    for color_by in allnodes:
        color_values = []

        for key, vec in sample_vectors.items():
            if drop_incomplete and np.isnan(vec).any(): continue
            loc_id, year, month = key
            #if isinstance(color_by, tuple):  # Edge
            #    node_i, node_j = color_by
            #    w = edge_weight_cache.get((loc_id, year, month, node_i, node_j), np.nan)
            #    color_values.append(w)

            #elif isinstance(color_by, str):  # Node variable
            node_idx = allnodes.index(color_by)
            if year==main_year: df = dinfo["loc_one_year_all_vars"][int(loc_id)].iloc[:365]
            else: df = dinfo["loc_yearly_all_vars"][int(loc_id)][str(year)].iloc[:365]
            block_start = month * 30
            block_end = 365 if month == 11 else (month + 1) * 30

            avg = df.iloc[block_start:block_end, node_idx].mean()
            color_values.append(avg)

        color_values_dict[color_by] = color_values
                #except Exception:
                #    color_values.append(np.nan)
            #else:
            #    color_values.append(1)

        assert len(rows) == len(color_values) and len(rows) == X.shape[0]
        assert X.shape[1] == len(edge_pairs)

        # Plot by node
        out_name = f"tsne_{main_year}_col_{color_by}" if len(years) == 1 and years[
            0] == main_year else f"tsne_years_col_{color_by}"
        scatter_tnse(Y, rows, color_values, color_by, out_name, edge_pairs, X, out_path)

    #plot by cluster
    out_name = f"tsne_{main_year}_col_cluster" if len(years) == 1 and years[
        0] == main_year else f"tsne_years_col_cluster"
    scatter_tnse(Y, rows, labels, "cluster", out_name, edge_pairs, X, out_path)

    out_name = f"box_{main_year}" if len(years) == 1 and years[
        0] == main_year else f"box_years"
    boxes_tsne(
        labels=labels, color_values_dict=color_values_dict,
        out_dir=out_path + 'boxes/', out_nm=out_name, show=False
    )


import pandas as pd
import seaborn as sns
from pathlib import Path


def boxes_tsne(labels, color_values_dict, out_dir, out_nm, show=False):
    """
    labels: list or np.array of cluster labels (one per sample)
    color_values_dict: dict {node_name: list of values per sample}
    out_dir: where to save plots
    """
    df = pd.DataFrame({'cluster': labels})
    os.makedirs(out_dir, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for node, values in color_values_dict.items():
        df['value'] = values

        plt.figure(figsize=(6, 4))
        sns.boxplot(x='cluster', y='value', data=df)
        plt.title(f"{node} by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel(node)
        plt.tight_layout()

        filepath = Path(out_dir) / str(out_nm + f"_{node}.png")
        plt.savefig(filepath)
        if show:
            plt.show()
        else:
            plt.close()
        print(f"[✓] Saved: {filepath}")



def scatter_tnse(Y, rows, color_values, color_by, out_name, edge_pairs, X, out_path):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=color_values, s=20, alpha=0.7, cmap="viridis")
    plt.colorbar(scatter, label="Color value" if color_by else "")
    plt.title("t-SNE of All Link Strengths")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path + out_name + ".png")
    plt.close()
    print(f"[✓] t-SNE plot saved to {out_path + out_name + '.png'}")

    # Output dataframe
    df_out = pd.DataFrame(
        {
            "loc": [r[0] for r in rows],
            "year": [r[1] for r in rows],
            "month": [r[2] for r in rows],
            "tsne1": Y[:, 0],
            "tsne2": Y[:, 1],
            "color_value": color_values,
        }
    )
    edge_cols = {f"e_{i}->{j}": X[:, k] for k, (i, j) in enumerate(edge_pairs)}
    df_out = pd.concat([df_out, pd.DataFrame(edge_cols)], axis=1)
    save_interactive_tsne_plot(df_out, out_path + out_name + ".html")
    df_out.to_csv(out_path + out_name + f"_df.csv", index=False)

def cluster_tsne (Y: [float],
    k_min: int = 1,
    k_max: int = 10,
    random_state: int = 0,
    use_silhouette: bool = True):
    n_points = len(Y)

    if k_max is None:
        k_max = n_points  # Allow k=N
    best_k = k_min
    best_score = -np.inf
    best_model = None
    for k in range(k_min, min(k_max, n_points) + 1):
        model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = model.fit_predict(Y)
        if use_silhouette and 2 <= k <= n_points - 1:
            try:
                score = silhouette_score(Y, labels)
            except ValueError:
                score = -1
        else:
            # Use negative inertia to simulate elbow (smaller inertia = better)
            score = -model.inertia_
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
    labels = best_model.labels_
    return labels

#%% Cluster and show maps (river runoff)
def cluster_loc_yr(loc_id_file, info_file, loc_yr_mon_folder, out_folder, year, node_i, node_j, nm, figsize=(8,6), show_boxplots=False, plt_show=False):
    pre = PATH_PRE
    merged_df = read_loc_mappings(loc_id_file, info_file)

    avg_weights = read_mdl_strengths_loc_yr(loc_yr_mon_folder,
                                            year=year, node_i=node_i, node_j=node_j, header=None)

    clusters, model, best_k = cluster_locations(avg_weights)
    # print(f"Choosing k={best_k} contexts")
    cluster_map = {str(loc): cl for cl, locs in clusters.items() for loc in locs}
    merged_df["location_index"] = merged_df["location_index"].astype(str)
    merged_df["cluster"] = merged_df["location_index"].map(cluster_map)

    plot_cluster_attribute_boxplots(merged_df, RIVER_ATTRS,
                                    out_folder + nm + '/' + f'boxes/box_{year}/', show_boxplots)
    visualize_clusters_on_map(merged_df, clusters, model, avg_weights, out_folder  + nm + '/maps/',
                              f"clusters_map_{year}.html", figsize=figsize)
    visualize_clusters_geopandas(merged_df, clusters, out_folder + nm + '/maps/',
                                 f"clusters_map_{year}.png", year, model=model, figsize=figsize, plt_show=plt_show)
        # return  out_folder  + nm + '/maps/' +  f"clusters_map_{year}.html"

    # except ValueError: pass
    # #import webbrowser
    ##  webbrowser.open(PATH_RIVER_OUT + "clusters_map_2010.html")

def weightmaps_loc_yr(loc_id_file, info_file, loc_yr_mon_folder, out_folder, year, node_i, node_j, nm, figsize=(8,6), plt_show=False):
    pre = PATH_PRE
    merged_df = read_loc_mappings(loc_id_file, info_file)

    avg_weights = read_mdl_strengths_loc_yr(loc_yr_mon_folder,
                                            year=year, node_i=node_i, node_j=node_j, header=None)

    clusters, model, best_k = cluster_locations(avg_weights)
    # print(f"Choosing k={best_k} contexts")
    cluster_map = {str(loc): cl for cl, locs in clusters.items() for loc in locs}
    merged_df["location_index"] = merged_df["location_index"].astype(str)
    merged_df["cluster"] = merged_df["location_index"].map(cluster_map)

    #plot_cluster_attribute_boxplots(merged_df, RIVER_ATTRS,
    #                                out_folder + nm + '/' + f'boxes/box_{year}/', show_plots)
    visualize_clusters_on_map(merged_df, clusters, model, avg_weights, out_folder + nm + '/maps/',
                              f"weights_map_{year}.html", color_by_weight=True)
    visualize_clusters_geopandas(merged_df, clusters, file_path=out_folder + nm + '/maps/',
                                map_file= f"weights_map_{year}.png", year=year, model=model, weights=avg_weights, color_by_weight=True, figsize=figsize, plt_show=plt_show)

def cluster_loc_yr_mon(loc_id_file, info_file, loc_yr_mon_folder, out_dir, year, node_i, node_j, nm):
    pre = PATH_PRE
    merged_df = read_loc_mappings(loc_id_file, info_file)

    for mon in range(12):
        weights = read_mdl_strengths_loc_mon(loc_yr_mon_folder,  # Path(pre + PATH_RIVER_OUT) / "mdl_loc_mon_paper",
                                             year=year, month=mon, node_i=node_i, node_j=node_j, header=None)

        clusters, model, best_k = cluster_locations(weights)

        cluster_map = {str(loc): cl for cl, locs in clusters.items() for loc in locs}
        merged_df["location_index"] = merged_df["location_index"].astype(str)
        merged_df["cluster"] = merged_df["location_index"].map(cluster_map)

        visualize_clusters_on_map(merged_df, clusters, model, weights, out_dir + nm + '/maps/',
                                  f"clusters_map_{year}_{mon}.html")
        plot_cluster_attribute_boxplots(merged_df, RIVER_ATTRS,
                                        out_dir + nm + '/' + f'boxes/box_{year}_{mon}/')


def cluster_loc_avg(loc_id_file, info_file, loc_yr_mon_folder,out_dir, node_i, node_j, nm):
    pre = PATH_PRE
    merged_df = read_loc_mappings(loc_id_file, info_file)
    weights = read_mdl_strengths_loc_avg(loc_yr_mon_folder, node_i, node_j, header=None)

    clusters, model, best_k = cluster_locations(weights)
    cluster_map = {str(loc): cl for cl, locs in clusters.items() for loc in locs}
    merged_df["location_index"] = merged_df["location_index"].astype(str)
    merged_df["cluster"] = merged_df["location_index"].map(cluster_map)
    plot_cluster_attribute_boxplots(merged_df, RIVER_ATTRS,
                                    out_dir + nm + '/' + 'boxes/box_avg/')

    visualize_clusters_on_map(merged_df, clusters, model, weights,out_dir + nm + '/',
                              "maps/clusters_map_avg.html")


def read_loc_mappings(loc_id_file: Path, info_file: Path) -> pd.DataFrame:
    # First file: single row of loc_ids (from index 0 to 306)
    loc_ids = pd.read_csv(loc_id_file)["location_id"].tolist()
    loc_df = pd.DataFrame({"location_index": range(len(loc_ids)), "loc_id": loc_ids})

    meta_df = pd.read_csv(info_file)
    if "id" in meta_df.columns:
        meta_df = meta_df.rename(columns={"id": "loc_id"})

    merged = loc_df.merge(meta_df, on="loc_id", how="left")
    return merged


#%%  Boxplots (River runoff dataset)

def plot_cluster_attribute_boxplots(
        merged_df: pd.DataFrame,
        attributes: list[str],
        output_dir: str = "", show_plots=False
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = merged_df.dropna(subset=["cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)

    for attr in attributes:
        if attr not in df.columns:
            print(f"Skipping missing attribute: {attr}")
            continue

        plt.figure(figsize=(6, 5))
        sns.boxplot(data=df, x="cluster", y=attr, hue="cluster", palette="Set2", legend=False)

        plt.title(f"Distribution of '{attr}' by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel(attr)
        plt.tight_layout()

        output_path = Path(output_dir) / f"boxplot_{attr}.png"
        if show_plots: plt.show()
        else: plt.savefig(output_path)
        plt.close()



#%%  EUROPE MAPS (River runoff dataset)
import os
import pandas as pd
import folium


def visualize_clusters_on_map(
    merged_df: pd.DataFrame,
    clusters: dict,
    model,
    weights: dict,
    file_path,
    map_file,
    color_by_weight: bool = False,
    figsize = (8,6)
):
    fmap = folium.Map(
        location=[merged_df["lat"].mean(), merged_df["lon"].mean()],
        zoom_start=5
    )

    merged_df["location_index"] = merged_df["location_index"].astype(str)

    if color_by_weight:
        # Normalize weights to [0,1] and map to colormap
        values = np.array([weights.get(loc, np.nan) for loc in merged_df["location_index"]])
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colormap = cm.get_cmap("RdYlGn")

        for _, row in merged_df.iterrows():
            lat, lon = row["lat"], row["lon"]
            loc_index = row["location_index"]
            weight = weights.get(loc_index, np.nan)

            if np.isnan(weight):
                continue

            color = mcolors.to_hex(colormap(norm(weight)))
            popup = f"Location {loc_index}<br>Weight: {weight:.4f}"

            folium.CircleMarker(
                location=(lat, lon),
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=250),
            ).add_to(fmap)

    else:
        # Cluster color mapping by sorted centroids
        centroids = model.cluster_centers_.ravel()
        sorted_labels = np.argsort(centroids)
        ranked_cluster = {original: rank for rank, original in enumerate(sorted_labels)}

        color_palette = (
            ["blue", "green", "yellow", "orange", "red"]
            if len(centroids) <= 5
            else ["blue", "blue", "green", "green", "yellow", "yellow", "red", "red", "red", "red"]
        )
        assert len(set(ranked_cluster.values())) <= len(color_palette), "Too few colors!"

        cluster_map = {str(loc): cl for cl, locs in clusters.items() for loc in locs}
        merged_df["cluster"] = merged_df["location_index"].map(cluster_map)

        for _, row in merged_df.iterrows():
            lat, lon = row["lat"], row["lon"]
            loc_index = row["location_index"]
            cl = row["cluster"]

            if pd.isna(cl):
                continue

            cl = int(cl)
            rank = ranked_cluster[cl]
            color = color_palette[rank]
            weight = weights.get(loc_index, np.nan)

            popup = f"Location {loc_index}<br>Cluster {cl} (centroid rank {rank})<br>Weight: {weight:.4f}"

            folium.CircleMarker(
                location=(lat, lon),
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=250),
            ).add_to(fmap)

    os.makedirs(file_path, exist_ok=True)
    fmap.save(os.path.join(file_path, map_file))
    print(f"Map saved to {file_path + map_file}")

def visualize_clusters_on_map_old(
    merged_df: pd.DataFrame,
    clusters: dict,
    model,
    weights: dict,
    file_path, map_file,
    figsize = (8,6)
):
    # Cluster color mapping by sorted centroids
    centroids = model.cluster_centers_.ravel()
    sorted_labels = np.argsort(centroids)
    ranked_cluster = {original: rank for rank, original in enumerate(sorted_labels)}

    color_palette = ["blue", "green", "yellow", "orange", "red"] if len(centroids) <=5 else   ["blue","blue",  "green", "green", "yellow", "yellow", "red", "red", "red", "red"]
    assert len(set(ranked_cluster.values())) <= len(color_palette), "Too few colors!"

    cluster_map = {str(loc): cl for cl, locs in clusters.items() for loc in locs}
    merged_df["location_index"] = merged_df["location_index"].astype(str)
    merged_df["cluster"] = merged_df["location_index"].map(cluster_map)

    fmap = folium.Map(
        location=[merged_df["lat"].mean(), merged_df["lon"].mean()],
        zoom_start=5
    )

    for _, row in merged_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        loc_index = row["location_index"]
        cl = row["cluster"]

        if pd.isna(cl):
            continue

        cl = int(cl)
        rank = ranked_cluster[cl]
        color = color_palette[rank]

        weight = weights.get(loc_index, "N/A")
        popup = f"Location {loc_index}<br>Cluster {cl} (centroid rank {rank})<br>Weight: {weight:.4f}"

        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=250),
        ).add_to(fmap)
    os.makedirs(file_path, exist_ok=True)
    fmap.save(file_path+ map_file)

    print(f"Map saved to {file_path+map_file}")


from matplotlib.colors import ListedColormap

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualize_clusters_geopandas(
    merged_df: pd.DataFrame,
    clusters: dict,
    file_path: str,
    map_file: str,
    year,
    model=None,
    weights: dict = None,
    color_by_weight: bool = False,
    bbox_coords=(-36.386719, 29.228890, 60.292969, 74.543330),
    background_color='#cbd5e1',  # light slate (slate300),
    figsize = (8,6), plt_show=False
):
    from matplotlib.lines import Line2D

    merged_df["location_index"] = merged_df["location_index"].astype(str)

    # Assign cluster labels
    cluster_map = {str(loc): cl for cl, locs in clusters.items() for loc in locs}
    merged_df["cluster"] = merged_df["location_index"].map(cluster_map)

    # Filter and build geometry
    filtered_df = merged_df.dropna(subset=["cluster"]).copy()
    filtered_df["geometry"] = [Point(xy) for xy in zip(filtered_df["lon"], filtered_df["lat"])]
    gdf = gpd.GeoDataFrame(filtered_df, crs="EPSG:4326")

    colors_255 = [
        (34, 59, 88),
        (52, 90, 138),
        (133, 168, 204),
        (171, 188, 217),
        (207, 208, 228),
        (234, 230, 241),
        (253, 247, 251),
        (64, 111, 172),
        (92, 143, 189),
    ]

    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors_255]
    my_cmap = ListedColormap(colors, name='bluewhite')
    # Assign color
    if color_by_weight and weights:
        values = np.array([weights.get(loc, np.nan) for loc in gdf["location_index"]])
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap =my_cmap # cm.get_cmap("viridis")
        gdf["color"] = [mcolors.to_hex(cmap(norm(weights.get(loc, np.nan)))) for loc in gdf["location_index"]]
        show_legend = False
    else:
        # Cluster color mapping by sorted centroids
        centroids = model.cluster_centers_.ravel() if model is not None else [0 for _ in range(len(clusters))]
        sorted_cluster_ids = np.argsort(centroids)
        ranked_cluster = {cl: rank for rank, cl in enumerate(sorted_cluster_ids)}

        assert len(sorted_cluster_ids) <= len(CUSTOM_COLOR_LIST), "Too few colors in CUSTOM_COLOR_LIST!"
        cluster_colors = {cl: CUSTOM_COLOR_LIST[rank] for cl, rank in ranked_cluster.items()}
        gdf["cluster"] = gdf["cluster"].astype(int)
        gdf["color"] = gdf["cluster"].map(cluster_colors)
        show_legend = True

    # World map background
    long0, lat0, long1, lat1 = bbox_coords
    bbox = Polygon([[long0, lat0], [long1, lat0], [long1, lat1], [long0, lat1]])
    bbox_gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[bbox])

    import geodatasets
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    europe = world.overlay(bbox_gdf, how="intersection")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    europe.plot(ax=ax, color=background_color, zorder=0)
    gdf.plot(ax=ax, color=gdf["color"], markersize=10, legend=False, zorder=1)

    if show_legend:
        legend_elements = [
            Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=cluster_colors[cl],
                label=f"Context {cl} (µ={centroids[cl]:.3f})",
                markersize=8
            )
            for cl in sorted_cluster_ids
        ]
        ax.legend(
            handles=legend_elements,
            title="Cs", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False
        )

    ax.set_title(f"Year {year}")
    ax.set_axis_off()
    plt.tight_layout()

    Path(file_path).mkdir(parents=True, exist_ok=True)
    full_path = Path(file_path) / map_file
    plt.show() if plt_show else plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    if not plt_show: print(f"Geo map saved to {full_path.resolve()}")


def create_map_dashboard(
    map_folder: Path,
    output_file: str = "map_dashboard.html",
    columns: int = 2
):
    map_files = sorted(Path(map_folder).glob("*.html"))
    if not map_files:
        print(" No map HTML files found.")
        return

    html = [
        "<html><head><title>Map Dashboard</title></head><body>",
        "<style>iframe { width: 49%; height: 400px; border: none; margin: 0.5%; }</style>",
        "<div style='display:flex; flex-wrap:wrap;'>"
    ]

    for i, map_file in enumerate(map_files):
        html.append(f"<iframe src='{map_file.name}'></iframe>")

    html += ["</div></body></html>"]

    output_path = Path(map_folder) / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"Dashboard saved: {output_path.resolve()}")

#%% UTIL (reading edge strengths)

def read_mdl_strengths_loc_avg(
    folder: str | Path,
    node_i: int,
    node_j: int,
    header: int | None = None,
) -> Dict[str, float]:
    folder = Path(folder)
    weights_by_loc: Dict[str, list] = {}

    for file in folder.glob("*.csv"):
        match = re.match(r"(\d+)_\d+_0_\d+\.csv", file.name)
        if not match:
            continue

        loc_id = match.group(1)  # extract the first part (location ID)

        try:
            df = pd.read_csv(file, header=header)
            weight = float(df.iloc[node_i, node_j])
            weights_by_loc.setdefault(loc_id, []).append(weight)
        except Exception as e:
            print(f"Warning: skipping {file.name} → {e}")

    avg_weights = {
        loc: sum(ws) / len(ws) for loc, ws in weights_by_loc.items() if ws
    }
    return avg_weights


def read_mdl_strengths_loc_yr(
    folder: str | Path,
    *,
    year: str | int,
    node_i: int,
    node_j: int,
    header: int | None = None,
) -> Dict[str, float]:
    """For a given year and edge, get avg edge weight across all months for each loc, note: nodej is the target node, nodei the index of (parent, time lag)
    :return: dict {location_id: avg_weight}
    """
    folder = Path(folder)
    monthly_weights: Dict[str, List[float]] = {}

    for month in range(12):
        pattern = f"*_{year}_0_{month}.csv"
        for csv_path in folder.glob(pattern):
            loc_id = csv_path.stem.split("_")[0]
            try:
                df = pd.read_csv(csv_path, header=header)
                weight = float(df.iloc[node_i, node_j])
                monthly_weights.setdefault(loc_id, []).append(weight)
            except Exception as e:
                print(f"Warning: Skipped {csv_path.name} — {e}")
                continue
    return {
        loc: sum(ws) / len(ws) for loc, ws in monthly_weights.items() if len(ws) > 0
    }



def read_mdl_strengths_loc_mon(
    folder: str | Path,
    *,
    year: str | int,
    month: int,
    node_i: int,
    node_j: int,
    header: int | None = None,
) -> Dict[str, float]:
    """ as above for a given month
    """
    folder = Path(folder)
    pattern = f"*_{year}_0_{month}.csv"

    weights: Dict[str, float] = {}
    for csv_path in folder.glob(pattern):
        loc_id = csv_path.stem.split("_")[0]
        df = pd.read_csv(csv_path, header=header)
        weight = float(df.iloc[node_i, node_j])
        weights[loc_id] = weight
    return weights
""" 

def partitioning_permonth_selected_locs(): 

    # Hyperparameters: the discovered causal DAG
    links = {0: RIVER_RESULT_LINKS}
    truths = SimpleNamespace(true_links=links)
    method = MethodType.GP_DAG
    assumed_min_dur = 30

    out_dir = PATH_RIVER_OUT
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Select Locations
    n_locs = 50 #100  # subsample for efficiency
    dinfo = get_river_data(n_locs=n_locs)
    interesting_ci = [ci for ci in dinfo.loc_ids]
    interesting_cnm = [dinfo.loc_ids[ci] for ci in dinfo.loc_ids]
    results = {}

    for mi in range(12):
        context_dic = {}
        for ii, ci in enumerate(interesting_ci):
            idx = [i for i in range(mi * 30, (mi + 1) * 30)]
            #df = dinfo.loc_one_year[ci][:365]
            df = np.array(dinfo.loc_one_year[ci])[idx]
            context_dic[ii] = np.array(df)

        print(f'*** EVALUATING  {len(context_dic)} locations in month {dinfo.months[mi]}*** ')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spct = SpaceTime(method,
                             CpsInitializationStrategy.BINS,
                             truths=truths,
                             assumed_max_lag=assumed_max_lag,
                             assumed_min_dur=assumed_min_dur,
                             initial_bin_size=initial_bin_size,
                             logger=dinfo.log, verbosity=verbosity,
                             interleaving_iterations=interleaving_iterations,
                             out=out_dir)

            # Set Hyperparameters, monthly regime duration
            params_T, params_N = context_dic[0].shape
            params = {'N': params_N, 'T': params_T}
            options = get_options(DiscrepancyTestType.KCD, spct.method_type, params_N, params_T, spct.logger,
                                  spct.min_dur)
            n_bin_samples, n_bin_regions = assumed_min_dur, int(np.floor(params_T / assumed_min_dur))
            nb_chunks = int(np.floor(params_T / n_bin_samples))
            partition = partition_t(params_T, n_bin_regions, nb_chunks, n_bin_samples, True)
            windows_T = r_partition_to_windows_T(partition, spct.skip)

            # Regime and Context Partitioning
            r_partition, c_partition, r_partitions, c_partitions = spct.partition_under_regimes(context_dic, links,
                                                                                                windows_T)
            print(f'*** PARTITIONING RESULT *** \n\tFor locations {interesting_cnm} \nand month {dinfo.months[mi]} using windows {windows_T} ')
            print(f'\tRegimes: {r_partition}\n\tContexts: {c_partition}')
            print(f'\tRegimes/node: {r_partitions}\n\tContexts/node: {c_partitions}')

        results[f'Month_{mi}_{dinfo.months[mi]}'] = SimpleNamespace(
            month=mi, loc_names=interesting_cnm, locs=interesting_ci, context_dic=context_dic,
            r_partition=r_partition, c_partition=c_partition, r_partitions=r_partitions,
            c_partitions=c_partitions
        )
    basins_info = pd.read_csv(PATH_RIVER_BASINSINFO)
    print(f'*** SIMILAR LOCATIONS ***')


    for ky in results:
        print('\n', ky)
        for regime in np.unique(results[ky].c_partitions[2]):
            contexts_in_regime = [ci for ci, x in enumerate(results[ky].c_partitions[2]) if x == regime]
            if len(contexts_in_regime) > 2:

                print(f'\tRegime {int(regime)}:')  # [(ci, interesting_cnm[ci]) for ci in contexts_in_regime])

                for ci in contexts_in_regime:
                    row = basins_info.loc[basins_info['id'] == int(interesting_cnm[ci])]
                    print(f"\t\tLoc. {ci}: ({float(row['lon'])},{float(row['lat'])} )")

    print(results)
    return results



def partitioning_peryear_selected_locs(): 

    # Hyperparameters: the discovered causal DAG
    links = {0: RIVER_RESULT_LINKS}
    truths = SimpleNamespace(true_links=links)
    method = MethodType.GP_DAG
    assumed_min_dur = 365 # this means that we compare the timeseries of one year over all locations

    out_dir = PATH_RIVER_OUT
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Select Locations
    dinfo = get_river_data()

    interesting_ci = [ci for ci in dinfo.loc_ids]
    interesting_cnm = [dinfo.loc_ids[ci] for ci in dinfo.loc_ids]

    results = {}
    results_indicator = {}
    for ci, cj in itertools.combinations(interesting_ci, 2):
        context_dic = {}
        for ii, cii in enumerate([ci, cj]):
            idx = [i for i in range(365)]
            df = np.array(dinfo.loc_one_year[cii])[idx]
            context_dic[ii] = np.array(df)

        print(f'*** EVALUATING  {len(context_dic)} locations *** ')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spct = SpaceTime(method,
                             CpsInitializationStrategy.BINS,
                             truths=truths,
                             assumed_max_lag=assumed_max_lag,
                             assumed_min_dur=assumed_min_dur,
                             initial_bin_size=initial_bin_size,
                             logger=dinfo.log, verbosity=verbosity,
                             interleaving_iterations=interleaving_iterations,
                             out=out_dir)

            # Set Hyperparameters, monthly regime duration
            params_T, params_N = context_dic[0].shape
            params = {'N': params_N, 'T': params_T}
            options = get_options(DiscrepancyTestType.KCD, spct.method_type, params_N, params_T, spct.logger,
                                  spct.min_dur)
            n_bin_samples, n_bin_regions = assumed_min_dur, int(np.floor(params_T / assumed_min_dur))
            nb_chunks = int(np.floor(params_T / n_bin_samples))
            partition = partition_t(params_T, n_bin_regions, nb_chunks, n_bin_samples, True)
            windows_T = r_partition_to_windows_T(partition, spct.skip)

            # Regime and Context Partitioning
            r_partition, c_partition, r_partitions, c_partitions = spct.partition_under_regimes(context_dic, links,
                                                                                                windows_T)
            print(f'*** PARTITIONING RESULT ({ci}, {cj}) *** \n\tFor locations {interesting_cnm} \n  using windows {windows_T} ')
            print(f'\tRegimes: {r_partition}\n\tContexts: {c_partition}')
            print(f'\tRegimes/node: {r_partitions}\n\tContexts/node: {c_partitions}')

        results[f'{ci}_{cj}'] = SimpleNamespace(ci=ci, cj=cj, loci=interesting_ci[ci], locj = interesting_ci[cj],
            r_partition=r_partition, c_partition=c_partition, r_partitions=r_partitions,
            c_partitions=c_partitions
        )

    basins_info = pd.read_csv(PATH_RIVER_BASINSINFO)
    print(f'*** SIMILAR LOCATIONS ***')


    for ky in results:
        print('\n', ky)
        for regime in np.unique(results[ky].c_partitions[2]):
            contexts_in_regime = [ci for ci, x in enumerate(results[ky].c_partitions[2]) if x == regime]
            if len(contexts_in_regime) > 2:

                print(f'\tRegime {int(regime)}:')  # [(ci, interesting_cnm[ci]) for ci in contexts_in_regime])

                for ci in contexts_in_regime:
                    row = basins_info.loc[basins_info['id'] == int(interesting_cnm[ci])]
                    print(f"\t\tLoc. {ci}: ({float(row['lon'])},{float(row['lat'])} )")

    print(results)
    return results
 

def cps_search_selected_locs(): 

    links = RIVER_RESULT_LINKS
    dag = RIVER_RESULT_DAG
    out_dir = PATH_RIVER_OUT
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    dinfo = get_river_data()
    ci_cps = {}
    # Select locations
    for ci in dinfo.loc_one_year:
        context_dic = {}
        idx = [i for i in range(365)]
        df = np.array(dinfo.loc_one_year[ci])[idx]
        context_dic[0] = np.array(df)


        # Detect cutpoints under the true DAG
        truths = SimpleNamespace(true_links=links)
        method = MethodType.GP_DAG
        spct = SpaceTime(method,
                         CpsInitializationStrategy.BINS,
                         truths=truths,
                         assumed_max_lag=assumed_max_lag,
                         assumed_min_dur=assumed_min_dur,
                         initial_bin_size=initial_bin_size,
                         logger=dinfo.log, verbosity=verbosity,
                         interleaving_iterations=interleaving_iterations,
                         out=out_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spct.run(context_dic)
        print(f'*** CPS SEARCH RESULT *** \n\tFor {len(context_dic)} locations using windows {spct.cur_regimes} ')
        print(f'\tCPS: {spct.cur_regimes}\n\tRegimes: {spct.cur_partition}\n\tContexts: {spct.cur_c_partition}')
        print(f'\tRegimes/node: {spct.cur_r_partitions}\n\tContexts/node: {spct.cur_c_partitions}')
        ci_cps[ci] = SimpleNamespace(
            regimes = spct.cur_regimes, partition =spct.cur_partition,
            r_partitions = spct.cur_r_partitions   )
    print(ci_cps)
    for ci in ci_cps:
        reg_order = [p for _, _, p in ci_cps[ci].partition]
        diff_reg = np.unique([p for _, _, p in ci_cps[ci].partition])
        if len(diff_reg) <= 5:
            print(ci, dinfo.loc_ids[ci],'\t', len(diff_reg), reg_order, '\t',ci_cps[ci].partition)

    return ci_cps

def to_tex_cps():
    ci_cps =cps_search_selected_locs()
    dinfo = get_river_data()

    for ci in ci_cps:
        pass
    yr = '2010'

    for ci in ci_cps:
        dr = '../basin data/timeseries'
        nm = dinfo.loc_ids[ci]
        # ci = [cj for cj in dinfo.flux_ids if dinfo.flux_ids[cj] == nm][0]
        reg_order = [p for _, _, p in ci_cps[ci].partition]
        diff_reg = np.unique([p for _, _, p in ci_cps[ci].partition])
        min_dur = min([dr for _, dr, _ in ci_cps[ci].partition])
        if len(diff_reg) <= 5:  # and min_dur > 60:
            print(ci, dinfo.loc_ids[ci], '\t', len(diff_reg), reg_order, '\t', ci_cps[ci].partition)
            for root, dirs, files in os.walk(dr):
                for file in files:
                    idf = file.split('.')[0]
                    if str(idf) != str(nm):
                        continue
                    print('found', idf)
                    data_frame = pd.read_csv(os.path.join(root, file))
                    time_information = np.array(data_frame['time'])

                    relevant_year = '2010'
                    filter_row = [row for row in data_frame['time'] if str(row).startswith(relevant_year)]
                    data_frame = data_frame[data_frame['time'].isin(filter_row)]
                    data_frame.insert(0, "t", [i for i in range(len(data_frame))], True)
                    data_frame.to_csv(f'tex_river_data/{len(diff_reg)}_{nm}_{yr}_{ci_cps[ci].partition}')
                    for plotted_var in ['Qobs', 'tavg', 'prec']:
                        plt.plot(np.arange(0, len(data_frame), 1), data_frame[plotted_var], label=plotted_var
                                 )
                    for st, ln, reg in ci_cps[ci].partition:
                        plt.axvline(x=st, color='gray')
                        plt.text(st + 1, 0, reg, rotation=90)
                    plt.legend()
                    plt.savefig(f'tex_river_plots/{nm}_{ci_cps[ci].partition}.png')
                    plt.close()
        break

"""