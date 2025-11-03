import os
from pathlib import Path

import numpy as np
import matplotlib
import pandas as pd
from netgraph import Graph
from matplotlib.colors import ListedColormap

from src.exp.exp_stime.data_preproc import get_flux_data

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.exp.exp_stime.old.util_fluxnet_show import  _mean_var_loc_yr_month, _node_link_strengths_loc_yr_month, \
    load_mdl_loc_yr_month, load_cd_loc_month, _nms_loc_month, _node_strengths_loc_month, _link_strengths_loc_month, \
    _optics_loc_month, _link_strength_mean_loc_month, _mean_var_loc_month, _link_strengths_loc_yr_month, \
    correct_loc_yr_month, _allnode_link_strengths_loc_yr_month, \
    _allnode_link_strengths_loc_yr_month_small


def show_regime_partitions():
    # for the abnormal year experiment
    out_dir = 'plts/flux/'
    dinfo = get_flux_data()
    regime_df = pd.read_csv('reproduce_info/FLUX_regime_yearly.csv')
    regime_df_nodes = [pd.read_csv('reproduce_info/FLUX_regime_yearly_node_0.csv'),
                       pd.read_csv('reproduce_info/FLUX_regime_yearly_node_1.csv'),
                       pd.read_csv('reproduce_info/FLUX_regime_yearly_node_2.csv'),
                       pd.read_csv('reproduce_info/FLUX_regime_yearly_node_3.csv'),
                       pd.read_csv('reproduce_info/FLUX_regime_yearly_node_4.csv'),
                       pd.read_csv('reproduce_info/FLUX_regime_yearly_node_5.csv')]
    for ci in dinfo.flux_ids:
        Path(out_dir + f'{ci}/').mkdir(parents=True, exist_ok=True)
        for var in dinfo.all_variables:
            dr = 'flux_krich_et_al2/flux_krich_et_al/'
            for yr in dinfo.flux_contexts_years_allvars[ci]:
                for root, dirs, files in os.walk(dr):
                    for file in files:
                        # Select daily timeseries
                        if not (('MM' in file.split('_') and 'FULLSET' in file.split('_'))):
                            continue
                        idf = file.split('_')[file.split('_').index('FLX') + 1]
                        yrs = file.split('_')[file.split('_').index('MM') + 1]
                        st, end = yrs.split('-')
                        if not idf == dinfo.flux_ids[ci]:
                            continue
                        data_frame = pd.read_csv(os.path.join(root, file))
                        filter_row = [row for row in data_frame['TIMESTAMP'] if
                                      any([str(row).startswith(str(year)) for year in [yr]])]
                        data_frame = data_frame[data_frame['TIMESTAMP'].isin(filter_row)]
                        ks = [k for k in regime_df if k.split('_')[1] == idf and k.split('_')[2] == yr]
                        regime = int(regime_df[ks[0]])
                        print(ks)
                        plt.plot(np.arange(0, 12, 1), data_frame[var], label=yr,
                                 color="gray" if regime == 0 else "blue" if regime == 1 else "green" if regime == 2 else "purple" if regime == 3 else "yellow" if regime == 4 else "orange" if regime == 5 else "brown" if regime == 6 else "black")
            plt.legend()
            plt.savefig(out_dir + f'{ci}/{var}.png')
            plt.close()
        for ivar, var in enumerate(dinfo.relevant_variables):
            dr = 'flux_krich_et_al2/flux_krich_et_al/'
            for yr in dinfo.flux_contexts_years_allvars[ci]:
                for root, dirs, files in os.walk(dr):
                    for file in files:
                        # Select daily timeseries
                        if not (('MM' in file.split('_') and 'FULLSET' in file.split('_'))):
                            continue
                        idf = file.split('_')[file.split('_').index('FLX') + 1]
                        yrs = file.split('_')[file.split('_').index('MM') + 1]
                        st, end = yrs.split('-')
                        if not idf == dinfo.flux_ids[ci]:
                            continue
                        data_frame = pd.read_csv(os.path.join(root, file))
                        filter_row = [row for row in data_frame['TIMESTAMP'] if
                                      any([str(row).startswith(str(year)) for year in [yr]])]
                        data_frame = data_frame[data_frame['TIMESTAMP'].isin(filter_row)]
                        ks = [k for k in regime_df if k.split('_')[1] == idf and k.split('_')[2] == yr]
                        regime = int(regime_df_nodes[ivar][ks[0]])
                        print(ks)
                        plt.plot(np.arange(0, 12, 1), data_frame[var], label=yr,
                                 color="gray" if regime == 0 else "blue" if regime == 1 else "green" if regime == 2 else "purple" if regime == 3 else "yellow" if regime == 4 else "orange" if regime == 5 else "brown" if regime == 6 else "black")
            plt.legend()
            plt.savefig(out_dir + f'{ci}/nodep_{var}.png')
            plt.close()

    #show regime assignment
    yrs_per_loc = {}
    all_yrs = []
    for ky in regime_df.columns:
       # print(ky, int(regime_df[ky]))
        loc = ky.split('_')[1]
        yr = ky.split('_')[2]
        reg = int(regime_df[ky])
        if loc not in yrs_per_loc:
            yrs_per_loc[loc] = {}
        if reg not in yrs_per_loc[loc]:
            yrs_per_loc[loc][reg] = []
        yrs_per_loc[loc][reg].append(yr)
        if yr not in all_yrs:
            all_yrs.append(yr)
    thresh = 1
    thresh = 1
    for yr in all_yrs:
        print(yr)
        ct = 0
        dect = 0
        ave_len = 0
        exists_in_loc = 0
        for loc in yrs_per_loc:
            for reg in yrs_per_loc[loc]:
                if yr in yrs_per_loc[loc][reg]:
                    exists_in_loc += 1
                    if len(yrs_per_loc[loc][reg]) <= thresh:
                        ct += 1
                    else:
                        dect += 1
                    ave_len += len(yrs_per_loc[loc][reg])
        print(
            f'Year {yr}: {ct} / {exists_in_loc} occ={ct / exists_in_loc:.2f} \t {dect} / {exists_in_loc} = {dect / exists_in_loc:.2f}')


def show_tsne_loc_yr_month(all_pairs=False):
    plt.style.use('seaborn-v0_8-dark')
    cmap = matplotlib.colormaps["viridis"]
    mark_alpha = 0.5
    mark_sz = 15

    # out_dir = 'plts/loc_yr_month-pair/' if all_pairs else 'plts/loc_yr_month/'
    out_dir = 'plts/loc_yr_month/'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dinfo = get_flux_data()
    # weights_loc_yr_month = load_mdl_loc_yr_month_pairs(dinfo) if all_pairs else load_mdl_loc_yr_month(dinfo)
    weights_loc_yr_month = load_mdl_loc_yr_month(dinfo)

    # Exclude months that dont meet quality thresh
    corrected = correct_loc_yr_month(dinfo, weights_loc_yr_month)
    weights_loc_yr_month = corrected

    # a smaller version
    weights_small = {}
    for ci in weights_loc_yr_month:
        weights_small[ci] = {}
        for ri in range(12):
            weights_small[ci][ri] = {}
            ws = np.zeros((12, 6))
            for yr in weights_loc_yr_month[ci]:
                if str(ri) in weights_loc_yr_month[ci][yr]:
                    ws = ws + weights_loc_yr_month[ci][yr][str(ri)]
            weights_small[ci][ri] = ws / len(weights_loc_yr_month[ci])

    flat_loc_yr_month = [weights_loc_yr_month[ci][yr][ri].flatten() for ci in weights_loc_yr_month
                         for yr in weights_loc_yr_month[ci] for ri in
                         weights_loc_yr_month[ci][yr]]
    # flat_loc_yr_month = [weights_small[ci][ri].flatten() for ci in weights_small
    #                    for ri in weights_small[ci]]
    dims_loc_yr_month = TSNE(perplexity=30, random_state=653, learning_rate=200, early_exaggeration=12, n_iter=2000,
                             n_components=2, ).fit_transform(np.array(flat_loc_yr_month))

    dim1, dim2 = dims_loc_yr_month[:, 0], dims_loc_yr_month[:, 1]

    # plot link strengths for each edge
    _link_strengths_loc_yr_month(out_dir, dinfo, weights_loc_yr_month, dims_loc_yr_month)

    # Show by class

    igbp_geiger_info = pd.read_csv('reproduce_info/fluxnet_locations_krichetal.csv')
    igbp_categories = [i for i in igbp_geiger_info['IGBP']]
    cmap_colors = plt.cm.get_cmap('tab10', len(igbp_categories))  # Use a discrete colormap
    igbp_colormap = ListedColormap(cmap_colors(np.arange(len(igbp_categories))))

    # Reverse lookup for color mapping
    igbp_to_index = {igbp: i for i, igbp in enumerate(igbp_categories)}

    # Prepare data for plotting
    colrs = []
    for ci in weights_loc_yr_month:
        for yr in weights_loc_yr_month[ci]:
            for ri in weights_loc_yr_month[ci][yr]:
                value = weights_loc_yr_month[ci][yr][ri]
                igbp = igbp_categories[int(ci)]
                color_index = igbp_to_index[igbp]
                colrs.append(color_index)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(dim1, dim2, c=colrs, cmap=igbp_colormap, s=20, alpha=0.7)
    # Add legend for IGBP categories
    # legend_labels = [igbp for igbp in igbp_categories]
    # handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=cmap_colors(i), markersize=10)
    #           for i, label in enumerate(legend_labels)]
    # plt.legend(handles=handles, title='IGBP Classifications', loc='best')
    plt.colorbar(scatter, ticks=np.arange(len(igbp_categories)), label='IGBP')
    plt.savefig(out_dir + 'scatter_by_igbp.png')

    # plot link strengths per node
    Path(out_dir + "by_node/").mkdir(exist_ok=True)
    for node in range(len(dinfo.nodes)):
        clus = _node_link_strengths_loc_yr_month(node, dinfo, weights_loc_yr_month)
        plt.scatter(dim1, dim2, c=clus, s=mark_sz, alpha=mark_alpha, cmap=cmap)
        v = np.linspace(-.1, 2.0, 15, endpoint=True)
        x = plt.colorbar(ticks=v)
        plt.savefig(out_dir + "by_node/" + f"node_{node}_{dinfo.nodes[node]}")
        plt.close()

    # plot link strengths per node
    Path(out_dir + "by_all_nodes/").mkdir(exist_ok=True)

    clus = _allnode_link_strengths_loc_yr_month(weights_loc_yr_month)
    plt.scatter(dim1, dim2, c=clus, s=mark_sz, alpha=mark_alpha, cmap=cmap)
    v = np.linspace(-.1, 2.0, 15, endpoint=True)
    x = plt.colorbar(ticks=v)
    plt.savefig(out_dir + "by_all_nodes/plt")
    plt.close()
    dff = np.vstack([dim1, dim2, clus])
    pd.DataFrame(dff.T).to_csv('fluxnet_tsne_strength.csv', header=None, index=False)

    clus = _allnode_link_strengths_loc_yr_month_small(weights_small)
    plt.scatter(dim1, dim2, c=clus, s=mark_sz, alpha=mark_alpha, cmap=cmap)
    v = np.linspace(-.1, 2.0, 15, endpoint=True)
    x = plt.colorbar(ticks=v)
    plt.savefig(out_dir + "by_all_nodes/plt_small")
    plt.close()
    dff = np.vstack([dim1, dim2, clus])
    pd.DataFrame(dff.T).to_csv('fluxnet_tsne_strength_yearavg.csv', sep='\t', header=None, index=False)

    # Show whether system variables are reflected in the embbedding
    Path(out_dir + "by_sys_var/").mkdir(exist_ok=True)
    for var in dinfo.all_variables:
        cols = _mean_var_loc_yr_month(var, dinfo, weights_loc_yr_month)
        plt.scatter(dim1, dim2, c=cols, s=mark_sz, alpha=mark_alpha, cmap=cmap)
        plt.colorbar()
        plt.savefig(out_dir + f"by_sys_var/{var}")
        # v = np.linspace(-.1, 2.0, 15, endpoint=True)
        plt.close()

    # show by month
    cyc = [-1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]  # use cyclic colors, otherwise month 12 and month 1 appear different
    cols = [cyc[int(ri)] for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for ri in
            weights_loc_yr_month[ci][yr]]
    plt.scatter(dim1, dim2, c=cols, s=mark_sz, alpha=mark_alpha, cmap=cmap)
    plt.savefig(out_dir + f"by_month")
    plt.close()

    # show each month
    Path(out_dir + "by_month/").mkdir(exist_ok=True)
    for ri, month in enumerate(dinfo.months):
        cols = [1 if int(rj) == ri else 0 for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for rj in
                weights_loc_yr_month[ci][yr]]
        plt.scatter(dim1, dim2, c=cols, s=mark_sz, alpha=mark_alpha, cmap=cmap)
        plt.savefig(out_dir + f"by_month/{ri}_{month}")
        plt.close()

    # show each year
    if False:
        Path(out_dir + "by_year/").mkdir(exist_ok=True)
        yrs = np.unique([yr for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci]])
        for yi, yr in enumerate(yrs):
            cols = [1 if year == yr else 0 for ci in weights_loc_yr_month for year in weights_loc_yr_month[ci] for rj in
                    weights_loc_yr_month[ci][year]]
            plt.scatter(dim1, dim2, c=cols, s=mark_sz, alpha=mark_alpha, cmap=cmap)
            plt.savefig(out_dir + f"by_year/{yi}_{yr}")
            plt.close()

    # Show by location
    if False:
        Path(out_dir + "by_loc/").mkdir(exist_ok=True)
        ix = np.arange(
            len([0 for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for ri in
                 weights_loc_yr_month[ci][yr]]))
        cix = np.array([ci for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for ri in
                        weights_loc_yr_month[ci][yr]])
        for loc in weights_loc_yr_month:
            sel = np.where(cix == loc)
            cols = np.array([int(ri) for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for ri in
                             weights_loc_yr_month[ci][yr]])
            plt.scatter(dim1, dim2,  # c=cols[sel],, cmap=cmap
                        s=20, alpha=0.1)

            plt.scatter(dim1[sel], dim2[sel], s=mark_sz, alpha=1)
            plt.savefig(out_dir + f"by_loc/{loc}_{dinfo.flux_ids[int(loc)]}")
            plt.close()

    # Show trajectory over one year
    if False:
        Path(out_dir + "trajectories/").mkdir(exist_ok=True)
        for loc in weights_loc_yr_month:
            ryr = '2006' if '2006' in weights_loc_yr_month[loc] else '2010' if '2010' in weights_loc_yr_month[loc] else \
                [k for k in weights_loc_yr_month[loc]][0]
            print(loc, ryr)  # todo could do average but number of years and range is different
            yix = np.array(
                [ci if yr == ryr else -1 for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for ri in
                 weights_loc_yr_month[ci][yr]])
            sel = np.where(yix == loc)
            plt.scatter(dim1, dim2,  # c=cols[sel],, cmap=cmap
                        s=20, alpha=0.1)

            plt.scatter(dim1[sel], dim2[sel], s=mark_sz, alpha=1)
            plt.savefig(out_dir + f"trajectories/{loc}_{dinfo.flux_ids[int(loc)]}_{ryr}")
            plt.close()

    #   CLUSTERS
    Path(out_dir + "clus2/").mkdir(exist_ok=True)
    from sklearn.cluster import OPTICS
    clus = OPTICS().fit_predict(dims_loc_yr_month)
    from sklearn.cluster import KMeans
    km = KMeans(4).fit(dims_loc_yr_month)
    clus = km.labels_
    centers = km.cluster_centers_
    from itertools import cycle
    import matplotlib.cm as cm

    weights = np.array(
        [weights_loc_yr_month[ci][yr][ri] for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for ri
         in
         weights_loc_yr_month[ci][yr]])
    weights = np.array(
        [weights_small[ci][ri] for ci in weights_small for ri
         in
         weights_small[ci]])

    colors = cycle(cm.tab10.colors)

    # plt.figure()
    fig, ax = plt.subplots()
    for i in np.unique(clus):
        color = next(colors)
        idx = clus == i
        ax.scatter(dim1[idx], dim2[idx], color=color, s=10, label=i,  # label=labels[i],
                   alpha=0.25)
        ax.scatter(centers[i, 0], centers[i, 1], edgecolors="k", linewidth=2, color=color, s=200, alpha=1)
    legend1 = ax.legend(loc="upper left", title="")
    ax.add_artist(legend1)
    # plt.scatter(dim1, dim2, c=clus, s=mark_sz, alpha=mark_alpha, cmap=cmap)
    plt.savefig(out_dir + f"clus2/kmeans2")
    plt.close()
    # regions
    clus = [
        0 if i < 0.25 * j and j < 0.75 * i else 1 if i > 0.25 * j and j > 0.75 * i else 2 if i > 0.25 * j and j < 0.75 * i else 3
        for (i, j) in zip(dim1, dim2)]


    #   GRAPH p CLUSTER
    plt.close()
    for ic in np.unique(clus):
        color = next(colors)
        idx = clus == ic
        ws = sum(weights[idx]) / len(weights[idx])
        graph_data = [(i, j, np.round(ws[i][j], 2)) for i in range(6) for j in range(6) if
                      ws[i][j] != 0]
        edge_labels = {(i, j): np.round(ws[i][j], 2) for i in range(6) for j in range(6) if
                       ws[i][j] != 0}
        cmap = 'RdGy'
        Graph(graph_data, node_layout='circular', edge_labels=edge_labels,
              node_labels={i: dinfo.nodes[i] for i in range(6)}, edge_cmap=cmap,
              edge_width=2., arrows=True)
        plt.savefig(out_dir + f"clus2/graph_{ic}")
        plt.close()

    # GRAPH P CLUSTER THRESHOLDED
    for ic in np.unique(clus):
        color = next(colors)
        idx = clus == ic
        plt.scatter(dim1[idx], dim2[idx], color=color, s=10,  # label=labels[i],
                    alpha=0.25)
        plt.savefig(out_dir + f"clus/clus_thrsh_{ic}")
        plt.close()

        ws = sum(weights[idx]) / len(weights[idx])
        graph_data = [(i, j, max(np.round(ws[i][j], 2), 0)) for i in range(6) for j in range(6) if
                      max(ws[i][j], 0) > 0.2]
        edge_labels = {(i, j): max(np.round(ws[i][j], 2), 0) for i in range(6) for j in range(6) if
                       max(ws[i][j], 0) > 0.2}
        cmap = 'RdGy'
        node_nms = {i: dinfo.nodes[i] for i in range(6) if
                    i in [g[0] for g in graph_data] or i in [g[1] for g in graph_data]}
        Graph(graph_data, node_layout='circular', edge_labels=edge_labels,
              node_labels=node_nms, edge_cmap=cmap,
              edge_width=2., arrows=True)
        plt.savefig(out_dir + f"clus/graph_thrsh_{ic}")
        plt.close()


def show_tsne_loc_month():
    ''' Embedding all nodes together, one sample is a location and month in a fixed year'''
    out_dir = 'plts/loc_month/'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    dinfo = get_flux_data()
    weights_loc_month = load_cd_loc_month()
    nms = _nms_loc_month(dinfo, weights_loc_month)

    flat_loc_month = np.array(
        [weights_loc_month[ci][ri].flatten() for ci in weights_loc_month for ri in weights_loc_month[ci]])

    dims_loc_month = TSNE(perplexity=30, random_state=653, learning_rate=200, early_exaggeration=12, n_iter=2000,
                          n_components=2, ).fit_transform(np.array(flat_loc_month))

    dim1, dim2 = dims_loc_month[:, 0], dims_loc_month[:, 1]

    # Show by month
    cols = [int(ri) for ci in weights_loc_month for ri in weights_loc_month[ci]]
    plt.scatter(dim1, dim2, c=cols)
    plt.savefig(out_dir + f"by_month")

    nms = _nms_loc_month(dinfo, weights_loc_month)
    # Annotation - only useful for interactive plot
    for i, txt in enumerate(nms):
        plt.annotate(txt, (dim1[i], dim2[i]))

    # group by link strength for each node
    _node_strengths_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month)

    # group by link strength for each edge
    _link_strengths_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month)

    # cluster based on tsne distances using optics
    clus = _optics_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month)

    # color each point according to its cluster's mean edge strength
    _link_strength_mean_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month, clus)

    # color each point according to its cluster's mean edge strength per node
    for i in range(len(dinfo.nodes)):
        _link_strength_mean_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month, clus, i)

    # Special month dec
    cols = [1 if ri in [11] else 0 for ci in weights_loc_month for ri in weights_loc_month[ci]]
    plt.scatter(dim1, dim2, c=cols)

    # plot trajectory of specific location
    ix = [i for i in dinfo.flux_ids if dinfo.flux_ids[i] == 'US-SRM'][0]
    cols = [1 if ri in [ix] else 0 for ci in weights_loc_month for ri in weights_loc_month[ci]]
    plt.scatter(dim1, dim2, c=cols)

    # show precipitation and radiation
    var = 'P_F'  # SW_IN_F_MDS
    _mean_var_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month, var)


def show_tsne_loc_yr_month_node(node_j):
    plt.style.use('seaborn-v0_8-dark')
    cmap = matplotlib.colormaps["viridis"]
    mark_alpha = 0.5
    mark_sz = 15
    dinfo = get_flux_data()
    out_dir = f'plts/loc_yr_month-{node_j}-{dinfo.nodes[node_j]}/'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    weights_loc_yr_month = load_mdl_loc_yr_month()

    flat_loc_yr_month = np.array([
        weights_loc_yr_month[ci][yr][ri][:, node_j].flatten() for ci in weights_loc_yr_month for yr in
        weights_loc_yr_month[ci] for ri in weights_loc_yr_month[ci][yr]])

    dims_loc_yr_month = TSNE(perplexity=30, random_state=653, learning_rate=200, early_exaggeration=12, n_iter=2000,
                             n_components=2, ).fit_transform(np.array(flat_loc_yr_month))
    dim1, dim2 = dims_loc_yr_month[:, 0], dims_loc_yr_month[:, 1]


