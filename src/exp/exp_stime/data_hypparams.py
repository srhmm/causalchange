from pathlib import Path

import numpy as np
#PATH_PRE = "YOURPREFIX\causalchange\src/"
PATH_PRE =  str(Path(__file__).resolve().parent.parent.parent) +'/' #  paths should be fixed .....


""" Data parameters for fluxnet and river runoff experiments"""

# Paths
FLUX_PATH = '/dsets/dsets_stime/flux_krich_et_al/'
FLUX_LOC_INFO_KRICH_PATH = '/dsets/dsets_stime/fluxnet_locations_krichetal.csv'
FLUX_LOC_INFO_OURS_PATH = 'exp/exp_stime/res/res_stime/flux/fluxnet_locations_ours.csv'
FLUX_OUT_PATH = 'exp/exp_stime/res/res_stime/flux/'
PATH_FLUX_OUT = 'exp/exp_stime/res/res_stime/flux/'

# Relevant vars
FLUX_RELEVANT_VARIABLES = [
    'SW_IN_F_MDS',
    'NEE_VUT_USTAR50',
    'TA_F_MDS',
    'VPD_F_MDS',
    'H_F_MDS',
    'LE_F_MDS',
]
FLUX_ALL_VARIABLES = [
    'SW_IN_F_MDS',  # RG
    'NEE_VUT_USTAR50',
    'TA_F_MDS',
    'VPD_F_MDS',
    'H_F_MDS',
    'LE_F_MDS',
    'GPP_NT_VUT_USTAR50',  # 'SWC_F_MDS_1',
    'P_F'
]
FLUX_NODES = ['R', 'NEE', 'T', 'VPD', 'H', 'LE']
FLUX_ALLNODES = ['R', 'NEE', 'T', 'VPD', 'H', 'LE', 'GPP', 'P']
FLUX_MAINYEAR = 2006  # selected by large overlap among the locations


# Hyperparameters for our method
assumed_max_lag = 3
assumed_min_dur = 30
initial_bin_size = 30  # monthly initial splits might be reasonable
verbosity = 3
interleaving_iterations = 1  # get monthly results first


# Results for reference
FLUX_RESULT_LINKS = {
    0: [((0, 1), 1, None), ((5, 0), 1, None)],
    1: [((0, 0), 1, None),
        ((1, 1), 1, None),
        ((2, 0), 1, None),
        ((5, 0), 1, None)],
    2: [((2, 1), 1, None), ((3, 0), 1, None)],
    3: [((0, 0), 1, None), ((3, 1), 1, None)],
    4: [((0, 0), 1, None), ((4, 1), 1, None)],
    5: [((3, 0), 1, None), ((4, 0), 1, None), ((5, 1), 1, None)]}

FLUX_RESULT_REGIME_PARTITION = [
    (0, 52, 0.0),
    (52, 51, 1.0),
    (103, 62, 2.0),
    (165, 64, 3.0),
    (229, 52, 4.0),
    (281, 32, 5.0),
    (313, 52, 6.0)]


# Paths

PATH_RIVER_DATA = '/dsets/dsets_stime/basin data/timeseries'
PATH_RIVER_OUT = 'exp/exp_stime/res/res_stime/river/'
PATH_RIVER_BASINSINFO = '/dsets/dsets_stime/basin data/basins_info.csv'
PATH_RIVER_LOC_INFO = PATH_RIVER_OUT + 'runoff_locations_ours.csv'

# Relevant variables
RIVER_VARS = [
    'tavg',
    'prec',
    'Qobs',
]
RIVER_ALL_VARS = ['tavg', 'tmin', 'tmax', 'prec', 'rad', 'Qobs', 'QsimGlobal',
                  'snowpackGlobal', 'SWC005Global', 'SMGlobal', 'ETGlobal']
RIVER_NMS = ['T', 'P', 'Q']

MONTH_NMS = ['Jan', 'Fb', 'Mr', 'Ap', 'My', 'Jun', 'Jl', 'Au', 'Sp', 'Oc', 'Nv', 'Dc']

RIVER_ATTRS = [ 'slope', 'altitude_basin', 'altitude_station', 'area', 'impervious', 'forest', 'r_volume_yr']
# Hyperparameters for our method
assumed_max_lag = 3
assumed_min_dur = 30
initial_bin_size = 30  # monthly splits
verbosity = 3
interleaving_iterations = 1  # get monthly results
RIVER_MAINYEAR = '2010'
RIVER_MISSING_THRESH = 20


# Results for reference
RIVER_RESULT_LINKS = {
    0: [((0, 0), 1, None)],
    1: [((1, 1), 1, None)],
    2: [((1, 1), 1, None)
        ]}
RIVER_RESULT_DAG = np.array(
    [[0., 0., 0.],
     [0., 0., 1.],
     [0., 0., 0.]])

