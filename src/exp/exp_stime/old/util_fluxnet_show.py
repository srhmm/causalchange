import logging
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing


### LOADING PRECOMP RESULTS
def load_mdl_loc_yr_month_pairs(dinfo):
    # todo return _load_mdl_loc_yr_month('flux_results/mdl_loc_year_month_pairs/')
    # missing /
    weights = {}
    for root, dirs, files in os.walk('flux_results/'):
        for file in files:
            if not 'csv' in file.split('.'):
                continue
            if not 'pairs' in file:
                continue
            ci, yr, ci2, ri = file.split('pairs')[1].split('.')[0].split('_')

            if not yr in dinfo.flux_contexts_years[int(ci)]:
                print('skip yr', yr)
                continue
            df = np.array(
                pd.read_csv(os.path.join(root, file), header=None, sep=','))
            if not ci in weights:
                weights[ci] = {}
            if yr not in weights[ci]:
                weights[ci][yr] = {}
            weights[ci][yr][ri] = df
    print(f'Reading {len(weights)} locations,'
          f' {min([len(weights[ci].keys()) for ci in weights])}-'
          f'{max([len(weights[ci].keys()) for ci in weights])} years, '
          f'{min([len(weights[ci][yr].keys()) for ci in weights for yr in weights[ci]])}-'
          f'{max([len(weights[ci][yr].keys()) for ci in weights for yr in weights[ci]])} regimes each')
    weights_loc_yr_month = weights

    return weights_loc_yr_month


def load_mdl_loc_yr_month(dinfo):
    return _load_mdl_loc_yr_month(dinfo, 'flux_results/mdl_loc_year_month_links/')


def _load_mdl_loc_yr_month(dinfo, dr='flux_results/mdl_loc_year_month_links/'):
    weights = {}

    for root, dirs, files in os.walk(dr):
        for file in files:
            if not 'csv' in file.split('.'):
                continue
            ci, yr, ci2, ri = file.split('.')[0].split('_')
            # if not yr in dinfo.flux_contexts_years[int(ci)]:
            #    print('skip yr', yr)
            #    continue
            df = np.array(
                pd.read_csv(os.path.join(root, file), header=None, sep=','))
            if not ci in weights:
                weights[ci] = {}
            if yr not in weights[ci]:
                weights[ci][yr] = {}
            weights[ci][yr][ri] = df
    print(f'Reading {len(weights)} locations,'
          f' {min([len(weights[ci].keys()) for ci in weights])}-'
          f'{max([len(weights[ci].keys()) for ci in weights])} years, '
          f'{min([len(weights[ci][yr].keys()) for ci in weights for yr in weights[ci]])}-'
          f'{max([len(weights[ci][yr].keys()) for ci in weights for yr in weights[ci]])} regimes each')
    weights_loc_yr_month = weights
    return weights_loc_yr_month


# todo filter out missing data cases
def load_cd_loc_month(load_dir='exp_results/95'):
    weights = defaultdict(dict)
    for root, dirs, files in os.walk(load_dir):
        for file in files:
            if not 'csv' in file.split('.'):
                continue
            ci, ri = file.split('.')[0].split('_')[1], file.split('.')[0].split('_')[2]
            weights[int(ci)][int(ri)] = np.array(
                pd.read_csv(os.path.join(root, file), header=None, sep=','))
    print(f'Reading {len(weights.keys())} locations with at least'
          f' {min([len(weights[ci].keys()) for ci in weights])} regimes each')
    weights_loc_month = weights
    return weights_loc_month


def _load_mdl_loc_merged(load_dir='exp_results/95_mdl'):
    weights = {ci: [] for ci in range(95)}

    for root, dirs, files in os.walk(load_dir):
        for file in files:
            if not 'csv' in file.split('.'):
                continue
            # node = int(file.split('.')[0].split('_')[1])
            # parents = [int(cand.replace("(", "").replace("[", "")) for cand in
            #           file.split('_')[3].split(".")[0].split(",") if '(' in cand]
            # print(node, parents)
            df = np.array(
                pd.read_csv(os.path.join(root, file), header=None, sep=',')).flatten()
            for ci in range(95):
                weights[ci].append(df[ci])

    print(f'Reading {len(weights.keys())} locations w {min([weights[ci].shape[0] for ci in weights])} samples')
    weights_loc = weights
    return weights_loc


### LOC_YR_MONTH
def _mean_var_loc_yr_month(var, dinfo, weights_loc_yr_month):
    mean_vals = {}
    for ci in weights_loc_yr_month:
        mean_vals[int(ci)] = {}
        for yr in weights_loc_yr_month[ci]:
            mean_vals[int(ci)][yr] = {}
            for ri in weights_loc_yr_month[ci][yr]:
                # print(f"{dinfo.flux_ids[int(ci)]} year {yr} month {ri} indices {30 * int(ri)}-{(int(ri) + 1) * 30}")
                sub = dinfo.flux_contexts_years_allvars[int(ci)][yr][var][30 * int(ri):(int(ri) + 1) * 30]
                invalid = min(sub) < -9000
                mean_vals[int(ci)][yr][ri] = mean(sub)
                if invalid:
                    print('invalid')
                    mean_vals[int(ci)][yr][ri] = 0
    clus = [mean_vals[int(ci)][yr][ri] for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for ri in
            weights_loc_yr_month[ci][yr]]
    return clus


### LOC_YR_MONTH
def correct_loc_yr_month(dinfo, weights_loc_yr_month):
    corrected = {}
    for ci in weights_loc_yr_month:
        for yr in weights_loc_yr_month[ci]:
            for ri in weights_loc_yr_month[ci][yr]:
                # print(f"{dinfo.flux_ids[int(ci)]} year {yr} month {ri} indices {30 * int(ri)}-{(int(ri) + 1) * 30}")
                invalid = False
                for var in dinfo.all_variables:
                    sub = dinfo.flux_contexts_years_allvars[int(ci)][yr][var][30 * int(ri):(int(ri) + 1) * 30]
                    invalid = invalid or min(sub) < -9000  # todo and quality flag
                if invalid:
                    print('invalid')
                    continue
                else:
                    if ci not in corrected:
                        corrected[ci] = {}
                    if yr not in corrected[ci]:
                        corrected[ci][yr] = {}
                    corrected[ci][yr][ri] = weights_loc_yr_month[ci][yr][ri]

    return corrected


def _node_link_strengths_loc_yr_month(node, dinfo, weights_loc_yr_month):
    weights = np.array(
        [weights_loc_yr_month[ci][yr][ri] for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for ri in
         weights_loc_yr_month[ci][yr]])
    strengths = np.array([sum(in_weight[:, node]) for in_weight in weights])
    cols = [int(np.floor(st)) for st in strengths]

    # nms = _nms_loc_yr_month(dinfo, weights_loc_yr_month)
    # for i, txt in enumerate(nms):
    #    plt.annotate(txt, (dim1[i], dim2[i]))
    # srt = np.argsort(strengths)
    # for i in srt:
    #    print(nms[i])
    return cols


def _allnode_link_strengths_loc_yr_month(weights_loc_yr_month):
    weights = np.array(
        [weights_loc_yr_month[ci][yr][ri] for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for ri in
         weights_loc_yr_month[ci][yr]])
    strengths = np.array([sum(sum(in_weight)) for in_weight in weights])
    cols = [int(np.floor(st)) for st in strengths]
    min_max_scaler = preprocessing.MinMaxScaler()
    ccols = min_max_scaler.fit_transform(np.array(cols).reshape(-1, 1))
    ccols2 = min_max_scaler.fit_transform(np.array([-c for c in cols]).reshape(-1, 1))

    return cols, ccols, ccols2


def _allnode_link_strengths_loc_yr_month_small(weights_loc_yr_month):
    weights = np.array(
        [weights_loc_yr_month[ci][ri] for ci in weights_loc_yr_month for ri in
         weights_loc_yr_month[ci]])
    strengths = np.array([sum(sum(in_weight)) for in_weight in weights])
    cols = [int(np.floor(st)) for st in strengths]
    min_max_scaler = preprocessing.MinMaxScaler()
    ccols = min_max_scaler.fit_transform(np.array(cols).reshape(-1, 1))
    cclus2 = min_max_scaler.fit_transform(np.array([-c for c in cols]).reshape(-1, 1))

    return ccols


### LOC_MONTH
def _node_strengths_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month):
    strengths = np.array(
        [weights_loc_month[ci][ri] for ci in weights_loc_month for ri in weights_loc_month[ci]])
    fig, axs = plt.subplots(1, 6)
    for i, (node, row) in enumerate(zip(dinfo.nodes, range(len(axs)))):
        node_strengths = np.array([sum(in_strength[:, i]) for in_strength in strengths])
        cols = [int(np.floor(st)) for st in node_strengths]
        dim1, dim2 = dims_loc_month[:, 0], dims_loc_month[:, 1]
        axs[row].scatter(dim1, dim2, c=cols)
        # srt = np.argsort(node_strengths)
        # for i in srt:
        #    print(nms[i])
    plt.savefig(out_dir + f"stren_node_{node}_{dinfo.nodes[node]}")


def _link_strengths_loc_yr_month(out_dir, dinfo, weights_loc_yr_month, dims_loc_yr_month):
    cmap = matplotlib.colormaps["viridis"]
    mark_alpha = 0.5
    mark_sz = 15

    dim1, dim2 = dims_loc_yr_month[:, 0], dims_loc_yr_month[:, 1]
    strengths = np.array(
        [weights_loc_yr_month[ci][yr][ri] for ci in weights_loc_yr_month for yr in weights_loc_yr_month[ci] for ri in
         weights_loc_yr_month[ci][yr]])

    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
    nextrow = 0
    nextcol = 0
    for i, (node1) in enumerate(dinfo.nodes):
        for j, (node2) in enumerate(dinfo.nodes):

            node_strengths = np.array([in_weight[i, j] for in_weight in strengths])
            if all([st == 0 for st in node_strengths]):
                continue
            cols = [int(np.floor(st)) for st in node_strengths]
            sc = axs[nextrow][nextcol].scatter(dim1, dim2, c=cols, alpha=mark_alpha, s=8, cmap=cmap)
            axs[nextrow][nextcol].set_title(f"{node1}->{node2}")
            nextcol += 1
            if nextcol == 4:
                nextrow += 1
                nextcol = 0

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(sc, cax=cbar_ax)
    plt.savefig(out_dir + f"edge_strengths")
    plt.close()


def _link_strengths_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month):
    dim1, dim2 = dims_loc_month[:, 0], dims_loc_month[:, 1]
    strengths = np.array(
        [weights_loc_month[ci][ri] for ci in weights_loc_month for ri in weights_loc_month[ci]])
    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
    nextrow = 0
    nextcol = 0
    for i, (node1) in enumerate(dinfo.nodes):
        for j, (node2) in enumerate(dinfo.nodes):

            node_strengths = np.array([in_weight[i, j] for in_weight in strengths])
            if all([st == 0 for st in node_strengths]):
                continue
            cols = [int(np.floor(st)) for st in node_strengths]
            sc = axs[nextrow][nextcol].scatter(dim1, dim2, c=cols)
            axs[nextrow][nextcol].set_title(f"{node1}->{node2}")
            nextcol += 1
            if nextcol == 4:
                nextrow += 1
                nextcol = 0

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)
    plt.savefig(out_dir + f"stren_edges")


def _optics_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month):
    from sklearn.cluster import OPTICS
    clus = OPTICS().fit_predict(dims_loc_month)

    dim1, dim2 = dims_loc_month[:, 0], dims_loc_month[:, 1]
    plt.scatter(dim1, dim2, c=clus)
    nms = _nms_loc_month(dinfo, weights_loc_month)
    for c in np.unique(clus):
        print(c)
        points = [i for i in range(len(clus)) if clus[i] == c]
        print([nms[p] for p in points])
    plt.savefig(out_dir + f"clus_optics")
    return clus


def _link_strength_mean_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month, clus):
    labels = np.zeros(clus.shape)
    for c in np.unique(clus):
        points = [i for i in range(len(clus)) if clus[i] == c]
        weights = np.array(
            [weights_loc_month[ci][ri] for ci in weights_loc_month for ri in
             weights_loc_month[ci]])

        adjs_clus = [weights[i] for i in range(len(clus)) if clus[i] == c]
        mn = sum(adjs_clus) / len(adjs_clus)

        graph_data = [(i, j, np.round(mn[i][j], 2)) for i in range(6) for j in range(6) if
                      mn[i][j] != 0]
        for i in range(len(clus)):
            if clus[i] == c:
                labels[i] = np.round(sum([mn[i][j] for i in range(6) for j in range(6) if
                                          mn[i][j] != 0]), 2)

    dim1, dim2 = dims_loc_month[:, 0], dims_loc_month[:, 1]
    plt.scatter(dim1, dim2, c=labels)
    plt.savefig(out_dir + f"clus_optics_strength")


def _link_strength_mean_node_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month, clus, node_j):
    labels = np.zeros(clus.shape)
    for c in np.unique(clus):
        weights = np.array(
            [weights_loc_month[ci][ri] for ci in weights_loc_month for ri in
             weights_loc_month[ci]])

        adjs_clus = [weights[i] for i in range(len(clus)) if clus[i] == c]
        mn = sum(adjs_clus) / len(adjs_clus)

        for i in range(len(clus)):
            if clus[i] == c:
                labels[i] = np.round(sum([mn[node_i][node_j] for node_i in range(6) if
                                          mn[node_i][node_j] != 0]), 2)

        dim1, dim2 = dims_loc_month[:, 0], dims_loc_month[:, 1]
        plt.scatter(dim1, dim2, c=labels)
    plt.savefig(out_dir + f"clus_optics_node_{node_j}_{dinfo.nodes[node_j]}")


def _mean_var_loc_month(out_dir, dinfo, weights_loc_month, dims_loc_month, var):
    mean_vals = {}
    for ci in range(95):
        mean_vals[ci] = {}
        for ri in range(12):
            sub = dinfo.flux_contexts_allvars[ci][var][30 * ri:30 * (ri + 1)]
            assert (len(sub) == 30)
            mean_vals[ci][ri] = mean(sub)
    clus = [mean_vals[ci][ri] for ci in weights_loc_month for ri in
            weights_loc_month[ci]]
    dim1, dim2 = dims_loc_month[:, 0], dims_loc_month[:, 1]

    plt.scatter(dim1, dim2, c=clus)

    plt.savefig(out_dir + f"clus_optics_node_{node_j}_{dinfo.nodes[node_j]}")


### UTIL
def _nms_loc_yr_month(dinfo, weights_loc_yr_month):
    locations = dinfo.flux_ids
    return [f'{locations[int(ci)]}-{yr[2:]}-{dinfo.months[int(ri)]}' for ci in weights_loc_yr_month for yr in
            weights_loc_yr_month[ci] for ri in
            weights_loc_yr_month[ci][yr]]


def _nms_loc_month(dinfo, weights_loc_month):
    locations = dinfo.flux_ids
    return [f'{locations[ci]}-{dinfo.months[ri]}' for ci in weights_loc_month for ri in
            weights_loc_month[ci]]
