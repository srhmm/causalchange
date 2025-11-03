import os

from src.exp.exp_stime.data_hypparams import RIVER_MAINYEAR, PATH_RIVER_BASINSINFO, MONTH_NMS
from src.exp.exp_stime.old.st_results import contexts_river_01_jan, contexts_river_02_feb, contexts_river_03_mar, contexts_river_05_my, \
    contexts_river_09_sep, contexts_river_06_june, contexts_river_10_oct, contexts_river_07_july, contexts_river_11_nov, \
    contexts_river_04_ap, contexts_river_08_au, contexts_river_12_dec, RIVER_LOC_IDS
from src.exp.exp_stime.old.util_fluxnet_show import _mean_var_loc_yr_month
from data_preproc import get_flux_data

from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib
import pandas as pd
import geopandas as gpd
import geodatasets
from shapely import box

from src.exp.exp_stime.exp_realworld import get_river_data

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_mdl_loc_yr_month(dtype, main_year=None):
    dr = 'river_runoff_results/mdl_loc_year_month_links/'
    if main_year is None:
        dr = 'river_runoff_results/mdl_loc_year_month_links_all/'
    weights = {}

    for root, dirs, files in os.walk(dr):
        for file in files:
            if not 'csv' in file.split('.'):
                continue
            ci, yr, ci2, ri = file.split('.')[0].split('_')
            if main_year is not None and yr != main_year:
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


def show_clusters():
    dinfo = get_river_data()
    basins_all = pd.read_csv(PATH_RIVER_BASINSINFO)
    clusters = {}
    for month_nm, month_result in zip(
            MONTH_NMS,
            [contexts_river_01_jan, contexts_river_02_feb, contexts_river_03_mar, contexts_river_04_ap,
             contexts_river_05_my, contexts_river_06_june, contexts_river_07_july, contexts_river_08_au
                , contexts_river_09_sep, contexts_river_10_oct, contexts_river_11_nov, contexts_river_12_dec]):
        cluster_regimes = {}
        print(f'Month {month_nm}')
        for regime in np.unique(month_result):
            contexts_in_regime = [ci for ci, x in enumerate(month_result) if x == regime]
            ids_in_regime = [int(RIVER_LOC_IDS[ci]) for ci, x in enumerate(month_result) if x == regime]
            if len(contexts_in_regime) > 2:
                print(f'\t\tRegime {int(regime)}:',
                      contexts_in_regime)  # [(ci, interesting_cnm[ci]) for ci in contexts_in_regime])
                filter = [row for row in basins_all['id'] if
                          row in ids_in_regime]
                rows = basins_all[basins_all['id'].isin(filter)]
                # for ci in contexts_in_regime:
                #    row = basins_info.loc[basins_info['id'] == int(interesting_cnm[ci])]
                # print(
                #    f"\t\tLoc. {ci}, {interesting_cnm[ci]}: ({float(row['lon']):.2f},{float(row['lat']):.2f}), area {float(row['area']):.2f}, altitude s {float(row['altitude_station']):.2f}, altitude b {float(row['altitude_basin']):.2f}, slope {float(row['slope']):.2f}")
                print(
                    f"\t\t\tArea ({min(rows['area'])}-{max(rows['area'])}) \t\t{float(mean(rows['area'])):.2f}\t {stdev(rows['area']):.2f},{stdev(basins_info['area']):.2f}, \n\t\taltitude s ({min(rows['altitude_station'])}-{max(rows['altitude_station'])}) \t\t {float(mean(rows['altitude_station'])):.2f},\t {stdev(rows['altitude_station']):.2f},{stdev(basins_info['altitude_station']):.2f},  \n\t\taltitude b ({min(rows['altitude_basin'])}-{max(rows['altitude_basin'])}) \t\t {float(mean(rows['altitude_basin'])):.2f},\t {stdev(rows['altitude_basin']):.2f},{stdev(basins_info['altitude_basin']):.2f},  \n\t\tslope({min(rows['slope'])}-{max(rows['slope'])}) \t\t  {float(mean(rows['slope'])):.2f}\t {stdev(rows['slope']):.2f},{stdev(basins_info['slope']):.2f}, ")

                cluster_regimes[regime] = rows
        clusters[month_nm] = cluster_regimes
        print(f'\tMonth {month_nm} n.clusters: {len(clusters)}')

    weights = load_mdl_loc_yr_month(RIVER_MAINYEAR)

    basins_all = pd.read_csv(PATH_RIVER_BASINSINFO)
    for month_nm, month_result in zip(
            MONTH_NMS,
            [contexts_river_01_jan, contexts_river_02_feb, contexts_river_03_mar, contexts_river_04_ap,
             contexts_river_05_my, contexts_river_06_june, contexts_river_07_july, contexts_river_08_au
                , contexts_river_09_sep, contexts_river_10_oct, contexts_river_11_nov, contexts_river_12_dec]):
        cluster_regimes = {}
        print(f'Month {month_nm}')
        for regime in np.unique(month_result):
            contexts_in_regime = [ci for ci, x in enumerate(month_result) if x == regime]
            ids_in_regime = [int(RIVER_LOC_IDS[ci]) for ci, x in enumerate(month_result) if x == regime]
            if len(contexts_in_regime) > 2:
                print(f'\t\tRegime {int(regime)}:',
                      contexts_in_regime)  # [(ci, interesting_cnm[ci]) for ci in contexts_in_regime])
                filter = [row for row in basins_all['id'] if
                          row in ids_in_regime]
                rows = basins_all[basins_all['id'].isin(filter)]
                for ci in contexts_in_regime:
                    ky = [cj for cj in dinfo.loc_ids if int(dinfo.loc_ids[cj]) == int(RIVER_LOC_IDS[ci])][0]
                    w = weights[str(ky)]

                #    row = basins_info.loc[basins_info['id'] == int(interesting_cnm[ci])]
                #print(
                #    f"\t\t\tLoc. {ci}, {RIVER_LOC_IDS[ci]}, {ky}: {w['2010']['0'][4, 2]}") # 0 -> month id
                # print(
                #    f"\t\t\tArea ({min(rows['area'])}-{max(rows['area'])}) \t\t{float(mean(rows['area'])):.2f}\t {stdev(rows['area']):.2f},{stdev(basins_all['area']):.2f}, \n\t\taltitude s ({min(rows['altitude_station'])}-{max(rows['altitude_station'])}) \t\t {float(mean(rows['altitude_station'])):.2f},\t {stdev(rows['altitude_station']):.2f},{stdev(basins_all['altitude_station']):.2f},  \n\t\taltitude b ({min(rows['altitude_basin'])}-{max(rows['altitude_basin'])}) \t\t {float(mean(rows['altitude_basin'])):.2f},\t {stdev(rows['altitude_basin']):.2f},{stdev(basins_all['altitude_basin']):.2f},  \n\t\tslope({min(rows['slope'])}-{max(rows['slope'])}) \t\t  {float(mean(rows['slope'])):.2f}\t {stdev(rows['slope']):.2f},{stdev(basins_all['slope']):.2f}, ")
                cluster_regimes[regime] = rows


    for mnth in MONTH_NMS:
        dff = clusters[mnth]
        colors = plt.cm.get_cmap('tab10', len(dff))
        for i, reg in enumerate(dff):
            df = dff[reg]
            plt.scatter(df['lon'], df['lat'], color=colors(i), label=f'C. {int(reg)}', alpha=0.7)
        # Add labels and title
        plt.xlabel('Lon')
        plt.ylabel('Lat')
        plt.title(f'Month {mnth}')
        plt.legend()
        plt.savefig(f"map_contexts_{mnth}")

        plt.close()

    attrs = ['altitude_basin', 'area', 'slope']
    for mnth in MONTH_NMS:
        for attr in attrs:
            dff = clusters[mnth]
            colors = plt.cm.get_cmap('tab10', len(dff))
            pos = 0
            for i, reg in enumerate(dff):
                if len(df) >= 5:
                    df = dff[reg]
                    plt.boxplot(df[attr], positions=[i]  # color=colors(i),  label=f'C. {int(reg)}', alpha=0.7
                                )
                    pos += 1
            # Add labels and title
            plt.xlabel('Lon')
            plt.ylabel('Lat')
            plt.title(f'Month {mnth}')
            plt.legend()
            plt.savefig(f"box_{mnth}_{attr}")

            plt.close()
            break


def show_map(weights_loc_yr_month, basins_info):
    node = 2
    out_dir = 'plts/river/'
    plt.style.use('seaborn-v0_8-dark')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
                                                               ["#fbbf24", "#16a34a", "#0369a1", "#0c4a6e", "#831843"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#fbbf24", "#16a34a", "#0369a1", "#0c4a6e"])
    node = 2

    out_dir = 'plts/river/'
    plt.style.use('seaborn-v0_8-dark')
    slate50, slate100, slate200, slate300, slate400, slate500 = '#f8fafc', '#f1f5f9', '#e2e8f0', '#cbd5e1', '#94a3b8', '#64748b',

    slate50, slate400 = '#f0f9ff', '#075985'

    Path(out_dir + "maps/").mkdir(exist_ok=True)

    def _avg_link_strengths_loc_small(node, weights_loc_yr_month):
        weights = np.array(
            [mean([sum(weights_loc_yr_month[ci][yr][ri][:, node]) for ri
                   in weights_loc_yr_month[ci][yr]]) for ci in weights_loc_yr_month for yr in [main_year]])

        return weights

    clus = _avg_link_strengths_loc_small(node, weights_loc_yr_month)

    # World map

    # filter_row = [row for row in basins_info['id'] if str(row) in dinfo.loc_ids.values()]
    # basins_info = basins_info[basins_info['id'].isin(filter_row)]
    # assert len(set(zip(basins_info['lon'], basins_info['lat'])))==len(dinfo.loc_ids)

    # geometry = [Point(xy) for xy in zip(basins_info['lon'], basins_info['lat'])]
    # gdf = GeoDataFrame(basins_info, geometry=geometry)
    ##world = gpd.read_file(geodatasets.data.naturalearth.land['url'])
    ##gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', c=[int(di ) for di in dim3], markersize=5);
    ##plt.savefig(out_dir+'world_map')

    from shapely.geometry import Polygon

    # Make polygon from bbox coordinates https://stackoverflow.com/a/68741143/18253502
    def make_bbox(long0, lat0, long1, lat1):
        return Polygon([[long0, lat0],
                        [long1, lat0],
                        [long1, lat1],
                        [long0, lat1]])

    # Coords covering Europe made with http://bboxfinder.com
    bbox = make_bbox(-36.386719, 29.228890, 60.292969, 74.543330)

    bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[bbox])

    europe = gpd.read_file(geodatasets.data.naturalearth.land['url'])
    # Use bbox as clipping border for Europe
    europe = europe.overlay(bbox_gdf, how="intersection")
    # axis
    fig, ax = plt.subplots(figsize=(8, 6))
    europe.plot(ax=ax, color=slate300)  # '#cbd5e1')
    for seli in range(1, int(np.ceil(max(clus))), 2):
        sel = [row for row in range(len(clus)) if clus[row] < seli]
        filter = [row for i, row in enumerate(basins_info['id']) if i in sel]
        print(seli, len(sel), len(filter))
        basins_info[basins_info['id'].isin(filter)].plot(x="lon", y="lat", kind="scatter",
                                                         c=clus[sel],
                                                         marker='o', s=5,  # c=weights,
                                                         # alpha=mark_alpha,
                                                         cmap=cmap, ax=ax)
    # background box for color
    xMin, yMin, xMax, yMax = europe.total_bounds
    bx = gpd.GeoDataFrame(geometry=[box(xMin, yMin, xMax, yMax)], crs=europe.crs)
    bx.plot(ax=ax, color="white", zorder=0)

    plt.savefig(out_dir + f'maps/world_map_highres_vio.svg', format='svg', dpi=3000)
    plt.close()

node = 2

out_dir = 'plts/river/'
plt.style.use('seaborn-v0_8-dark')
slate50, slate100, slate200, slate300, slate400, slate500 = '#f8fafc', '#f1f5f9', '#e2e8f0', '#cbd5e1', '#94a3b8', '#64748b',

slate50, slate400 = '#f0f9ff', '#075985'

Path(out_dir + "maps/").mkdir(exist_ok=True)


def _avg_link_strengths_loc_small(node, weights_loc_yr_month):
    weights = np.array(
        [mean([sum(weights_loc_yr_month[ci][yr][ri][:, node]) for ri
               in weights_loc_yr_month[ci][yr]]) for ci in weights_loc_yr_month for yr in [main_year]])

    return weights


    clus = _avg_link_strengths_loc_small(node, weights_loc_yr_month)

    # World map

    # filter_row = [row for row in basins_info['id'] if str(row) in dinfo.loc_ids.values()]
    # basins_info = basins_info[basins_info['id'].isin(filter_row)]
    # assert len(set(zip(basins_info['lon'], basins_info['lat'])))==len(dinfo.loc_ids)

    # geometry = [Point(xy) for xy in zip(basins_info['lon'], basins_info['lat'])]
    # gdf = GeoDataFrame(basins_info, geometry=geometry)
    ##world = gpd.read_file(geodatasets.data.naturalearth.land['url'])
    ##gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', c=[int(di ) for di in dim3], markersize=5);
    ##plt.savefig(out_dir+'world_map')

    from shapely.geometry import Polygon


    # Make polygon from bbox coordinates https://stackoverflow.com/a/68741143/18253502
    def make_bbox(long0, lat0, long1, lat1):
        return Polygon([[long0, lat0],
                        [long1, lat0],
                        [long1, lat1],
                        [long0, lat1]])


    # Coords covering Europe made with http://bboxfinder.com
    bbox = make_bbox(-36.386719, 29.228890, 60.292969, 74.543330)

    bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[bbox])

    europe = gpd.read_file(geodatasets.data.naturalearth.land['url'])
    # Use bbox as clipping border for Europe
    europe = europe.overlay(bbox_gdf, how="intersection")
    # axis
    fig, ax = plt.subplots(figsize=(8, 6))
    europe.plot(ax=ax, color=slate300)  # '#cbd5e1')
    #for seli in range(1, int(np.ceil(max(clus))), 2):
    #    sel = [row for row in range(len(clus)) if clus[row] < seli]
    #    filter = [row for i, row in enumerate(basins_info['id']) if i in sel]
    #    print(seli, len(sel), len(filter))
    #    basins_info[basins_info['id'].isin(filter)].
    basins_info.plot(x="lon", y="lat", kind="scatter",
                         c=clus,
                         marker='o', s=5,  # c=weights,
                         # alpha=mark_alpha,
                        zorder =1,
                         cmap=cmap, ax=ax)
    # background box for color
    xMin, yMin, xMax, yMax = europe.total_bounds
    bx = gpd.GeoDataFrame(geometry=[box(xMin, yMin, xMax, yMax)], crs=europe.crs)
    bx.plot(ax=ax, color="white", zorder=0)

    plt.savefig(out_dir + f'maps/world_map_highres_vio.svg', format='svg', dpi=3000)
    plt.close()


def show_river( ):
    out_dir = 'plts/river/'
    plt.style.use('seaborn-v0_8-dark')
    cmap = matplotlib.colormaps["viridis"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
                                                               ["#fbbf24", "#16a34a", "#0369a1", "#0c4a6e", "#831843"])
    mark_alpha = 0.5
    mark_sz = 15
    main_year = '2010'
    slate0, slate1, slate2 = '#94a3b8', '#64748b', '#cbd5e1'

    dinfo = get_river_data()
    loc_ids = dinfo.loc_ids
    # del dinfo

    weights = load_mdl_loc_yr_month( RIVER_MAINYEAR)
    weights_loc_yr_month = weights

    import pandas as pd
    basins_all = pd.read_csv(PATH_RIVER_BASINSINFO)
    filter = [row for row in basins_all['id'] if
              str(row) in dinfo.loc_ids.values()]
    basins_ours = basins_all[basins_all['id'].isin(filter)]
    assert len(set(zip(basins_ours['lon'], basins_ours['lat']))) == len(weights)

    #basins_ours.to_csv( 'basin data/basin data/basins_info_ours.csv')
    #basins_ours.to_csv(  '../basin data/basins_info_ours.csv')

    flat_loc_yr_month_small = [weights_loc_yr_month[ci][yr][ri].flatten() for ci in weights_loc_yr_month for yr in
                               weights_loc_yr_month[ci] if yr == main_year for ri in
                               weights_loc_yr_month[ci][yr]]
    # flat_loc_yr_month_sm = [fl[np.where(fl != 0)] for fl in flat_loc_yr_month_small]
    flat_loc_yr_month_sm = [[fl[0], fl[13], fl[14]] for fl in flat_loc_yr_month_small]

    dim1, dim2, dim3 = [fl[0] for fl in flat_loc_yr_month_sm], [fl[1] for fl in flat_loc_yr_month_sm], [fl[2] for fl in
                                                                                                        flat_loc_yr_month_sm]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ax = plt.axes(projection='3d')
    ax.scatter3D(dim1, dim2, dim3, c=dim3, cmap='Greens')
    plt.savefig(out_dir + 'plt')
    plt.close()

    def _node_link_strengths_loc_yr_month_small(node, weights_loc_yr_month):
        weights = np.array(
            [weights_loc_yr_month[ci][yr][ri] for ci in weights_loc_yr_month for yr in [main_year] for ri
             in weights_loc_yr_month[ci][yr]])
        strengths = np.array([sum(in_weight[:, node]) for in_weight in weights])
        cols = [int(np.floor(st)) for st in strengths]
        return cols

    def _monthly_link_strengths_loc_small(node, weights_loc_yr_month, month):
        weights = np.array(
            [mean([sum(weights_loc_yr_month[ci][yr][ri][:, node]) for ri
                   in weights_loc_yr_month[ci][yr] if str(ri) == str(month)]) for ci in weights_loc_yr_month for yr in
             [main_year]])
        return weights

    nodes = ['T', 'P', 'Qobs']
    # plot link strengths per node
    Path(out_dir + "by_node/").mkdir(exist_ok=True)
    for node in range(len(nodes)):
        clus = _node_link_strengths_loc_yr_month_small(node, weights_loc_yr_month)
        ax = plt.axes(projection='3d')
        ax.scatter3D(dim1, dim2, dim3, dim3, c=clus, s=mark_sz, alpha=mark_alpha, cmap=cmap)
        # v = np.linspace(-.1, 2.0, 15, endpoint=True)
        # x = plt.colorbar(ticks=v)
        plt.savefig(out_dir + "by_node/" + f"node_{node}_{nodes[node]}")
        plt.close()

    # show by month
    cycle = [-1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]  # use cyclic colors, otherwise month 12 and month 1 appear different
    cols = [cycle[int(ri)] for ci in weights_loc_yr_month for yr in ['2010'] for ri in
            weights_loc_yr_month[ci][yr]]
    ax = plt.axes(projection='3d')
    ax.scatter3D(dim1, dim2, dim3, dim3, c=cols, s=mark_sz, alpha=mark_alpha, cmap=cmap)
    plt.savefig(out_dir + f"by_month")
    plt.close()

    node = 2
    for ri in range(12):
        Path(out_dir + "maps/").mkdir(exist_ok=True, parents=True)
        clus = _monthly_link_strengths_loc_small(node, weights_loc_yr_month, ri)

        # World map
        import geopandas as gpd
        import geodatasets
        # filter_row = [row for row in basins_info['id'] if str(row) in dinfo.loc_ids.values()]
        # basins_info = basins_info[basins_info['id'].isin(filter_row)]
        # assert len(set(zip(basins_info['lon'], basins_info['lat'])))==len(dinfo.loc_ids)

        # geometry = [Point(xy) for xy in zip(basins_info['lon'], basins_info['lat'])]
        # gdf = GeoDataFrame(basins_info, geometry=geometry)
        ##world = gpd.read_file(geodatasets.data.naturalearth.land['url'])
        ##gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', c=[int(di ) for di in dim3], markersize=5);
        ##plt.savefig(out_dir+'world_map')

        # World plot
        from shapely.geometry import Polygon
        # Make polygon from bbox coordinates https://stackoverflow.com/a/68741143/18253502
        def make_bbox(long0, lat0, long1, lat1):
            return Polygon([[long0, lat0],
                            [long1, lat0],
                            [long1, lat1],
                            [long0, lat1]])

        # Coords covering Europe made with http://bboxfinder.com
        bbox = make_bbox(-36.386719, 29.228890, 60.292969, 74.543330)

        bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[bbox])

        europe = gpd.read_file(geodatasets.data.naturalearth.land['url'])
        # Use bbox as clipping border for Europe
        europe = europe.overlay(bbox_gdf, how="intersection")
        # initialize an axis
        fig, ax = plt.subplots(figsize=(8, 6))
        europe.plot(ax=ax, color=slate1)  # '#cbd5e1')
        basins_ours.plot(x="lon", y="lat", kind="scatter",
                         c=clus,
                         marker='o', s=5,  # c=weights,
                         # alpha=mark_alpha,
                         cmap=cmap, ax=ax)
        plt.savefig(out_dir + f'maps/world_map_month_{ri}')
        plt.close()

    ### LOC_YR_MONTH
    def _mean_var_loc_yr_month_small(var, dinfo, weights_loc_yr_month):
        mean_vals = {}
        for ci in weights_loc_yr_month:
            mean_vals[int(ci)] = {}
            for yr in ['2010']:
                mean_vals[int(ci)][yr] = {}
                for ri in weights_loc_yr_month[ci][yr]:
                    # print(f"{dinfo.flux_ids[int(ci)]} year {yr} month {ri} indices {30 * int(ri)}-{(int(ri) + 1) * 30}")
                    sub = dinfo.loc_one_year_all_vars[int(ci)][var][30 * int(ri):(int(ri) + 1) * 30]
                    # todo sub = dinfo.loc_yearly_allvars[int(ci)][yr][var][30 * int(ri):(int(ri) + 1) * 30]
                    mean_vals[int(ci)][yr][ri] = mean(sub)
                    # if invalid:
                    # mean_vals[int(ci)][yr][ri] = 0
        clus = [mean_vals[int(ci)][yr][ri] for ci in weights_loc_yr_month for yr in ['2010']  # weights_loc_yr_month[ci]
                for ri in weights_loc_yr_month[ci][yr]]
        return clus

    # Show whether system variables are reflected in the embbedding
    Path(out_dir + "by_sys_var/").mkdir(exist_ok=True)
    for var in dinfo.loc_one_year_all_vars.columns:  # dinfo.all_variables:
        cols = _mean_var_loc_yr_month(var, dinfo, _mean_var_loc_yr_month_small)
        plt.scatter(dim1, dim2, c=cols, s=mark_sz, alpha=mark_alpha, cmap=cmap)
        plt.colorbar()
        plt.savefig(out_dir + f"by_sys_var/{var}")
        # v = np.linspace(-.1, 2.0, 15, endpoint=True)
        plt.close()

