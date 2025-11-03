import re
from pathlib import Path

import numpy as np

from src.exp.exp_stime.data_hypparams import PATH_PRE, FLUX_RESULT_LINKS, FLUX_NODES, FLUX_MAINYEAR, \
    FLUX_LOC_INFO_OURS_PATH, \
    FLUX_LOC_INFO_KRICH_PATH, FLUX_OUT_PATH, FLUX_ALLNODES
from src.exp.exp_stime.data_preproc import  get_flux_data
from src.exp.exp_stime.exp_realworld import run_tsne_all_links

if __name__ == "__main__":

    dinfo = get_flux_data()


    links = FLUX_RESULT_LINKS
    n_n = len(FLUX_NODES)
    nodes = FLUX_NODES
    for node in range(n_n):
        for (pa, lg), _, _ in links[node]: print(f"{ n_n * np.abs(lg) + pa}, {node}: { nodes[node]}-> { nodes[pa]}")

    main_yr = FLUX_MAINYEAR
    loc_id_file = FLUX_LOC_INFO_OURS_PATH
    info_file =  FLUX_LOC_INFO_KRICH_PATH
    loc_yr_mon_folder = Path(PATH_PRE + FLUX_OUT_PATH) / "mdl_loc_mon_all" #todo all

    ## Discover the causal DAG over locations and months
    out_dir = PATH_PRE + FLUX_OUT_PATH
    ##spct = causal_discovery_loc(dinfo, out_dir)

    ## Causal edge strengths (per loc and timeframe) under the causal DAG
    #mdl_edge_strengths_loc_yr_month(dinfo, out_dir, FLUX_RESULT_LINKS, FLUX_MAINYEAR, False)


    run_tsne_all_links(dinfo=dinfo, folder=loc_yr_mon_folder, nodes=nodes, allnodes=FLUX_ALLNODES, links=links, years=[main_yr], main_year=main_yr,
                            out_path=out_dir+'tsne/')
    #save_interactive_tsne_plot(df, out_dir + f"interactive_tsne_plot_{main_yr}_colorby_{node}.html")
    year_pattern = re.compile(r"^\d+_(\d{4})_0_\d+\.csv$")

    yrs = np.unique([
        int(m.group(1))
        for f in loc_yr_mon_folder.glob("*.csv")
        if (m := year_pattern.match(f.name))
    ])
    run_tsne_all_links(dinfo=dinfo, folder=loc_yr_mon_folder, nodes=nodes, allnodes=FLUX_ALLNODES, links=links, years=yrs, main_year=main_yr,
                            out_path=out_dir+'tsne/')
    #save_interactive_tsne_plot(df, out_dir + f"interactive_tsne_plot_years_colorby_{node}.html")


""" 
    
    df = run_tsne_all_links(folder=loc_yr_mon_folder, nodes=nodes, links=links, years=yrs, out_path=out_dir+"tsne_all_links.png")
    save_interactive_tsne_plot(df, out_dir + f"interactive_tsne_plot.html")
    df.to_csv(out_dir+f"tsne_output.csv", index=False)

"""