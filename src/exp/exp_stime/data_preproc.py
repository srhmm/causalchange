import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


from typing import Dict, Any, Optional, List

import logging

from src.exp.exp_stime.data_hypparams import  FLUX_PATH, FLUX_MAINYEAR, FLUX_RELEVANT_VARIABLES, \
    FLUX_ALL_VARIABLES, \
    FLUX_NODES, FLUX_ALLNODES, PATH_RIVER_OUT, PATH_PRE, PATH_RIVER_LOC_INFO, PATH_FLUX_OUT, FLUX_LOC_INFO_KRICH_PATH, \
    FLUX_LOC_INFO_OURS_PATH
from src.exp.exp_stime.data_hypparams import PATH_RIVER_DATA, RIVER_VARS, RIVER_MAINYEAR, RIVER_MISSING_THRESH,  \
    RIVER_NMS, RIVER_ALL_VARS, MONTH_NMS


""" Loading the datasets """


def get_river_data(
    dr: str = PATH_PRE + PATH_RIVER_DATA,
    *,
    n_locs: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Load river‑runoff CSV files and organise them by location and year.

    Parameters
    ----------
    dr : str
        Directory containing CSV files (one file per location).
    n_locs : int, optional
        If given, stop after loading this many locations.
    logger : logging.Logger, optional
        Provide an existing logger to reuse; otherwise a basic logger is created.

    Returns
    -------
    dict
        Same structure as the original implementation, but built faster.
    """

    Path(PATH_RIVER_OUT).mkdir(parents=True, exist_ok=True)
    if logger is None:
        logging.basicConfig()
        logger = logging.getLogger("SPCTME-river-runoff")
        logger.setLevel(logging.INFO)
        out_dir = PATH_RIVER_OUT
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(f"{out_dir}/run_river_runoff.log")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)


    # Containers -------------------------------------------------------
    loc_ids: Dict[int, str] = {}
    loc_one_year, loc_one_year_all_vars = {}, {}
    loc_yearly, loc_yearly_all_vars = {}, {}
    file_ct = -1

    for root, _, files in os.walk(dr):
        for file in files:
            filepath = os.path.join(root, file)
            df = pd.read_csv(filepath)

            # Add a 'year' column (vectorised)
            df["year"] = df["time"].astype(str).str[:4]
            relevant_years = df["year"].unique()

            # Choose the "main" year (closest to RIVER_MAINYEAR)
            if RIVER_MAINYEAR in relevant_years:
                main_year = RIVER_MAINYEAR
            else:
                main_year = min(
                    relevant_years,
                    key=lambda y: abs(int(y) - int(RIVER_MAINYEAR)),
                )

            # Helper: extract a year‑specific frame ----------------------
            def frame_for(year: str, vars_subset: bool = True) -> pd.DataFrame:
                cols_to_drop = ["time", "year"]
                frame = df[df["year"] == year].drop(columns=cols_to_drop)
                if vars_subset:
                    frame = frame[RIVER_VARS]
                return frame

            # Main‑year frames ------------------------------------------
            year_frame = frame_for(main_year, vars_subset=True)
            all_frame = frame_for(main_year, vars_subset=False)

            if year_frame.empty:
                logger.warning("Skipping %s (empty after filtering)", file)
                continue

            # Basic NA check & fill once --------------------------------
            if (year_frame.isna().sum() > RIVER_MISSING_THRESH).any():
                logger.warning("Skipping %s (too many missing)", file)
                continue

            year_frame = year_frame.fillna(0)
            all_frame = all_frame.fillna(0)

            # Register location -----------------------------------------
            file_ct += 1
            loc_ids[file_ct] = file.split(".")[0]
            loc_one_year[file_ct] = year_frame
            loc_one_year_all_vars[file_ct] = all_frame
            loc_yearly[file_ct], loc_yearly_all_vars[file_ct] = {}, {}

            logger.info("%d: %s  (main year %s)", file_ct, loc_ids[file_ct], main_year)

            # Stop early if requested -----------------------------------
            if n_locs is not None and file_ct >= n_locs - 1:
                break

            for yr in relevant_years:
                y_frame = frame_for(yr, vars_subset=True).fillna(0)
                y_all = frame_for(yr, vars_subset=False).fillna(0)

                if (y_frame.isna().sum() > RIVER_MISSING_THRESH).any():
                    logger.debug("Skipping year %s in %s (missing data)", yr, file)
                    continue

                loc_yearly[file_ct][yr] = y_frame
                loc_yearly_all_vars[file_ct][yr] = y_all

        if n_locs is not None and file_ct >= n_locs - 1:
            break

    pd.DataFrame(loc_ids.values(), columns=["location_id"]).to_csv(
        PATH_PRE + PATH_RIVER_LOC_INFO, #f"{PATH_RIVER_OUT}/runoff_locations_ours.csv",
        index=False
    )

    return dict(
        nodes=RIVER_NMS,
        relevant_variables=RIVER_VARS,
        all_variables=RIVER_ALL_VARS,
        months=MONTH_NMS,
        file_ct=file_ct,
        loc_one_year=loc_one_year,
        loc_one_year_all_vars=loc_one_year_all_vars,
        loc_ids=loc_ids,
        loc_yearly=loc_yearly,
        loc_yearly_all_vars=loc_yearly_all_vars,
        log=logger,
    )




def get_flux_data(
    dr: str = PATH_PRE + FLUX_PATH,
    *,
    loc_info_path: str = PATH_PRE + FLUX_LOC_INFO_KRICH_PATH,
    n_locs: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    use_twin_year: bool = False
) -> Dict[str, Any]:
    """

    :param dr: data
    :param loc_info_path: csv w site start/end year, by Krich et al.
    :param n_locs: at most this many locs for quick tests
    :param logger: logging.Logger, optional
    :param use_twin_year: two year frame for "main year"
    :return: dict
    """
    # --------------------------------------------------  logging
    out_dir = Path(PATH_PRE + PATH_FLUX_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logging.basicConfig()
        logger = logging.getLogger("SPCTME-flux")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(out_dir / "run_flux.log")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    # --------------------------------------------------  containers
    flux_ids: Dict[int, str] = {}
    loc_main: Dict[int, pd.DataFrame] = {}          # closest to FLUX_MAINYEAR
    loc_main_all: Dict[int, pd.DataFrame] = {}
    loc_years: Dict[int, Dict[str, pd.DataFrame]] = {}
    loc_years_all: Dict[int, Dict[str, pd.DataFrame]] = {}

    meta_df = pd.read_csv(loc_info_path)
    file_ct = -1

    # --------------------------------------------------  iterate files
    for root, _, files in os.walk(dr):
        for fname in files:
            parts: List[str] = fname.split("_")
            if not ("DD" in parts and "FULLSET" in parts):
                continue

            site_id = parts[parts.index("FLX") + 1]
            file_path = Path(root) / fname

            # --- read once
            df = pd.read_csv(file_path)
            df["year"] = df["TIMESTAMP"].astype(str).str[:4]

            # --- site‑specific year window from meta file
            meta_row = meta_df.loc[meta_df["FLUXNETID"] == site_id]
            if meta_row.empty:
                logger.warning("Meta data missing for site %s – skipped", site_id)
                continue

            start_year = int(meta_row["Startyear"].iloc[0])
            end_year   = int(meta_row["Endyear"].iloc[0])

            df = df[df["year"].astype(int).between(start_year, end_year)]
            if df.empty:
                continue

            # --- choose 2‑year window around FLUX_MAINYEAR (or closest)
            centroid_year = (df["year"].astype(int)
                                   .unique()
                                   [np.argmin(np.abs(df["year"].astype(int).unique()
                                                     - FLUX_MAINYEAR))])

            twin_year = centroid_year + 1 if (centroid_year + 1) in df["year"].astype(int).unique() \
                else centroid_year - 1
            if use_twin_year:
                main_mask = df["year"].isin([str(centroid_year), str(twin_year)])
            else:  main_mask = df["year"].isin([str(centroid_year)])
            main_df   = df.loc[main_mask].copy()

            # --- missing‑value threshold check (vectorised)
            invalid_month = False
            month_idx = main_df.groupby(main_df.index // 30)  # rough 30‑day months
            for _, sub in month_idx:
                na_max = (sub[FLUX_RELEVANT_VARIABLES] < -9000).sum().max()
                if na_max > 8:
                    invalid_month = True
                    break
            if invalid_month:
                logger.debug("Skip %s – too many missing in 2‑year window", site_id)
                continue

            # --- replace sentinel values
            main_df.replace({-9999: 0, -9999.0: 0}, inplace=True)

            # --- slice vars
            main_sel   = main_df[FLUX_RELEVANT_VARIABLES].drop(columns="TIMESTAMP", errors="ignore")
            main_all   = main_df[FLUX_ALL_VARIABLES].drop(columns="TIMESTAMP", errors="ignore")

            # register location
            file_ct += 1
            loc_idx = file_ct
            flux_ids[loc_idx] = site_id
            loc_main[loc_idx]      = main_sel
            loc_main_all[loc_idx]  = main_all
            loc_years[loc_idx]     = {}
            loc_years_all[loc_idx] = {}

            logger.info("%d: %s (%d–%d), main %d/%d",
                        loc_idx, site_id, start_year, end_year,
                        centroid_year, twin_year)

            # stop early if requested
            if n_locs is not None and file_ct >= n_locs - 1:
                break

            # ------------------------------------------------  loop over each year once (reuse memory)
            for yr in range(start_year, end_year + 1):
                y_mask = df["year"] == str(yr)
                y_df   = df.loc[y_mask].copy()
                if y_df.empty:
                    continue

                # quick NA check
                na_count = (y_df[FLUX_RELEVANT_VARIABLES] < -9000).sum().max()
                if na_count > 8:
                    continue

                y_df.replace({-9999: 0, -9999.0: 0}, inplace=True)
                loc_years[loc_idx][str(yr)] = y_df[FLUX_RELEVANT_VARIABLES].drop(columns="TIMESTAMP",
                                                                                  errors="ignore")
                loc_years_all[loc_idx][str(yr)] = y_df[FLUX_ALL_VARIABLES].drop(columns="TIMESTAMP",
                                                                                errors="ignore")

        if n_locs is not None and file_ct >= n_locs - 1:
            break
    pd.DataFrame(flux_ids.values(), columns=["location_id"]).to_csv(
        PATH_PRE + FLUX_LOC_INFO_OURS_PATH,
        index=False
    )
    # --------------------------------------------------  assemble output
    return dict(
        nodes              = FLUX_NODES,
        allnodes           = FLUX_ALLNODES,
        relevant_variables = FLUX_RELEVANT_VARIABLES,
        all_variables      = FLUX_ALL_VARIABLES,
        months             = MONTH_NMS,
        file_ct            = file_ct,
        loc_ids            = flux_ids,                # {loc_idx: site_id}
        loc_one_year       = loc_main,                # 2‑year window closest to FLUX_MAINYEAR
        loc_one_year_all_vars= loc_main_all,
        loc_yearly         = loc_years,               # {loc_idx: {year: df}}
        loc_yearly_all_vars= loc_years_all,
        log                = logger,
    )



def get_flux_data_mess():
    file_ct = -1
    timefr = []

    flux_ids = {}  # location identifiers

    loc_sel_year = {}  # each location, selected year(s) closest to FLUX_MAINYEAR, relevant vars
    loc_sel_year_allvars = {}  # each location, selected year(s) closest to FLUX_MAINYEAR, all vars

    loc_years = {}  # each location, each year, relevant vars
    loc_years_allvars = {}  # each location, each year, all vars
    loc_st_end = pd.read_csv(FLUX_LOC_INFO_PATH)  # locations and years considered in Krich et. al.

    for root, dirs, files in os.walk(FLUX_PATH):
        for file in files:
            # Select daily timeseries
            if not (('DD' in file.split('_') and 'FULLSET' in file.split('_'))):
                continue

            idf = file.split('_')[file.split('_').index('FLX') + 1]
            yrs = file.split('_')[file.split('_').index('DD') + 1]
            st, end = yrs.split('-')

            data_frame = pd.read_csv(os.path.join(root, file))
            time_information = np.array(data_frame['TIMESTAMP'])
            relevant_years = np.unique([str(t)[0:4] for t in time_information])

            st_year = loc_st_end.loc[lambda l: l['FLUXNETID'] == idf]['Startyear']
            end_year = loc_st_end.loc[lambda l: l['FLUXNETID'] == idf]['Endyear']
            used_years = range(int(st_year), int(end_year + 1))

            filter_row = [row for row in data_frame['TIMESTAMP'] if
                          any([str(row).startswith(str(year)) for year in used_years])]
            data_frame = data_frame[data_frame['TIMESTAMP'].isin(filter_row)]

            relevant_year_1 = str(FLUX_MAINYEAR) if str(FLUX_MAINYEAR) in relevant_years else relevant_years[
                np.argmin([np.abs(int(year) - 2006) for year in relevant_years])]
            relevant_year_2 = str(int(relevant_year_1) + 1) if str(int(relevant_year_1) + 1) in relevant_years else \
                relevant_years[
                    np.argmin([np.abs(int(year) - (int(relevant_year_1) + 1)) for year in relevant_years])]
            filter_row = [row for row in data_frame['TIMESTAMP'] if
                          str(row).startswith(relevant_year_1) or str(row).startswith(relevant_year_2)]

            year_frame = data_frame[data_frame['TIMESTAMP'].isin(filter_row)]
            invalid = False
            is_na = 0
            for ri in range(12):
                is_na = max([len(year_frame[var][30 * int(ri):(int(ri) + 1) * 30][
                                     year_frame[var][30 * int(ri):(int(ri) + 1) * 30] < -9000]) for var in
                             FLUX_RELEVANT_VARIABLES])
                invalid = invalid or is_na > 8
            if invalid:
                continue
            else:
                print('\tmax. missing data per month:', is_na)

            year_frame = year_frame.replace(-9999, 0)
            year_frame = year_frame.replace(-9999.0, 0)
            print('\t min val: ', min(year_frame.min()))

            year_frame = year_frame.drop('TIMESTAMP', axis=1)
            year_frame2 = year_frame[FLUX_ALL_VARIABLES]
            year_frame = year_frame[FLUX_RELEVANT_VARIABLES]

            if not (year_frame.shape[0] != 0 and year_frame.shape[1] != 0):
                continue

            print(f"{file_ct + 1}: {idf}, {st}-{end}, {relevant_year_1}, {relevant_year_2}")

            file_ct = file_ct + 1

            loc_sel_year[file_ct] = year_frame
            loc_sel_year_allvars[file_ct] = year_frame2
            loc_years[file_ct] = {}
            loc_years_allvars[file_ct] = {}

            write_to = os.path.join('flux_res/',
                                    idf + '/' + str(relevant_year_1) + '_' + str(relevant_year_2) + '/causal/')
            Path(write_to).mkdir(parents=True, exist_ok=True)

            flux_ids[file_ct] = idf
            for yr in used_years:
                relevant_year = str(yr)

                data_frame = pd.read_csv(os.path.join(root, file))
                filter_row = [row for row in data_frame['TIMESTAMP'] if str(row).startswith(relevant_year)]

                year_frame = data_frame[data_frame['TIMESTAMP'].isin(filter_row)]

                invalid = False
                is_na = 0
                for ri in range(12):
                    is_na = max([len(year_frame[var][30 * int(ri):(int(ri) + 1) * 30][
                                         year_frame[var][30 * int(ri):(int(ri) + 1) * 30] < -9000]) for var in
                                 FLUX_RELEVANT_VARIABLES])

                    invalid = invalid or is_na > 8
                if invalid:
                    print('\tinvalid year:', yr, ':', is_na)
                    continue
                else:
                    print('\tmax. missing data per month in year:', yr, ':', is_na)
                year_frame = year_frame.replace(-9999, 0)
                year_frame = year_frame.replace(-9999.0, 0)
                print('\t min val: ', min(year_frame.min()))

                year_frame = year_frame.drop('TIMESTAMP', axis=1)
                year_frame = year_frame[FLUX_RELEVANT_VARIABLES]
                loc_years[file_ct][relevant_year] = year_frame

                year_frame2 = data_frame[data_frame['TIMESTAMP'].isin(filter_row)]
                year_frame2 = year_frame2.drop('TIMESTAMP', axis=1)
                year_frame2.replace(-9999, 0)
                year_frame2.replace(-9999.0, 0)
                loc_years_allvars[file_ct][relevant_year] = year_frame2
    logging.basicConfig()
    log = logging.getLogger("SPCTME-flux")
    log.setLevel("INFO")
    out_dir = 'out-flux/'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"{out_dir}run.log")
    fh.setLevel(logging.INFO)
    log.addHandler(fh)

    filter_row = [row for row in loc_st_end['FLUXNETID'] if str(row) in flux_ids.values()]
    loc_st_end_ours = loc_st_end[loc_st_end['FLUXNETID'].isin(filter_row)]
    # loc_st_end_ours.to_csv('reproduce_info/fluxnet_locations_ours.csv', index=False)

    return SimpleNamespace(nodes=FLUX_NODES, allnodes=FLUX_ALLNODES,
                           relevant_variables=FLUX_RELEVANT_VARIABLES,
                           all_variables=FLUX_ALL_VARIABLES,
                           months=MONTH_NMS,
                           file_ct=file_ct,
                           timefr=timefr,
                           flux_contexts=loc_sel_year,
                           flux_ids=flux_ids,
                           flux_contexts_allvars=loc_sel_year_allvars,
                           flux_contexts_years_allvars=loc_years_allvars,
                           flux_contexts_years=loc_years,
                           log=log)
