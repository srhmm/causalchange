import re
from pathlib import Path

import numpy as np

from src.exp.exp_stime.data_hypparams import RIVER_RESULT_LINKS, RIVER_NMS, PATH_RIVER_OUT, PATH_RIVER_BASINSINFO, \
    PATH_PRE, RIVER_MAINYEAR, PATH_RIVER_LOC_INFO
from src.exp.exp_stime.data_preproc import get_river_data
from src.exp.exp_stime.exp_realworld import cluster_loc_yr, weightmaps_loc_yr

if __name__ == "__main__":

    dinfo = get_river_data()

    links = RIVER_RESULT_LINKS
    n_n = len(RIVER_NMS)
    nodes = RIVER_NMS

    for node in range(n_n):
        for (pa, lg), _, _ in links[node]: print(f"{ n_n * np.abs(lg) + pa}, {node}: { nodes[node]}-> { nodes[pa]}")
    edges = [#(0, 0,  "Edge_T_T"), (4, 1,  "Edge_P_P"),
             (4, 2,  "Edge_Q_P")]

    main_yr = RIVER_MAINYEAR
    out_dir = PATH_PRE + PATH_RIVER_OUT
    loc_id_file = PATH_PRE + PATH_RIVER_LOC_INFO
    info_file = PATH_PRE + PATH_RIVER_BASINSINFO
    loc_yr_mon_folder = Path(PATH_PRE + PATH_RIVER_OUT) / "mdl_loc_mon_2010" #"mdl_loc_mon_2010"


    year_pattern = re.compile(r"^\d+_(\d{4})_0_\d+\.csv$")
    yrs = np.unique([
        int(m.group(1))
        for f in loc_yr_mon_folder.glob("*.csv")
        if (m := year_pattern.match(f.name))
    ])

    for (node_i, node_j, nm) in edges:
        cluster_loc_yr(loc_id_file, info_file, loc_yr_mon_folder, out_dir, main_yr, node_i, node_j, nm)
        weightmaps_loc_yr(loc_id_file, info_file, loc_yr_mon_folder, out_dir, main_yr, node_i, node_j, nm)
    #df = run_tsne_all_links(dinfo=dinfo, folder=loc_yr_mon_folder, nodes=nodes, links=links, years=yrs, main_year=main_yr, out_path=out_dir+"tsne_all_links.png")
    #save_interactive_tsne_plot(df, out_dir + f"interactive_tsne_plot.html")
    #df.to_csv(out_dir+f"tsne_output.csv", index=False)

    #for node in RIVER_NMS:
    #    df = run_tsne_all_links(dinfo=dinfo, folder=loc_yr_mon_folder, nodes=nodes, allnodes=nodes, links=links, years=[main_yr],
    # main_year = main_yr,    #                            out_path=out_dir+f"tsne_all_links_{main_yr}_colorby_{node}.png", color_by=node)
    #    save_interactive_tsne_plot(df, out_dir + f"interactive_tsne_plot_{main_yr}_colorby_{node}.html")
    #    df.to_csv(out_dir+f"tsne_output_{main_yr}_colorby_{node}.csv", index=False)

    #for (e1, e2, nm) in edges:
    #    df = run_tsne_all_links(dinfo=dinfo, folder=loc_yr_mon_folder, nodes=nodes, links=links, years=[main_yr],
    #                        out_path=out_dir+f"tsne_all_links_{main_yr}.png", color_by=(e1, e2))

    #df = run_tsne_all_links(folder=loc_yr_mon_folder, nodes=nodes, links=links, years=yrs, out_path=out_dir+"tsne_all_links.png")
    #save_interactive_tsne_plot(df, out_dir + f"interactive_tsne_plot.html")
    #df.to_csv(out_dir+f"tsne_output.csv", index=False)



    ## Discover the causal DAG over locations and months

    #spct = causal_discovery_loc(dinfo, out_dir)

    ## Causal edge strengths (per loc and timeframe) under the causal DAG
    #mdl_edge_strengths_loc_yr_month(dinfo, out_dir, RIVER_RESULT_LINKS, RIVER_MAINYEAR, False)

    ## Discover cps
    #ci_cps = cps_search_locs(dinfo, out_dir)

    ##yr_cps = {}
    ##for year in sorted(yrs): yr_cps[year] = cps_search(year)
    ##print(yr_cps)

    # Clusters, boxes, maps
    #for (node_i, node_j, nm) in nodes:
    #    cluster_loc_yr(loc_id_file, info_file, loc_yr_mon_folder,out_dir, '2010', node_i, node_j, nm)
    #    #cluster_loc_avg(loc_id_file, info_file, loc_yr_mon_folder, out_dir, node_i, node_j, nm)
    ##    #for year in sorted(yrs):
    ##    #    print(year)
    ##    #    cluster_loc_yr(loc_id_file, info_file, loc_yr_mon_folder,out_dir, year, node_i, node_j, nm)

    #    #create_map_dashboard( out_dir + nm + '/maps/' )


