import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data_preproc import get_flux_data
from src.exp.exp_stime.data_hypparams import FLUX_OUT_PATH, assumed_max_lag, assumed_min_dur, initial_bin_size, verbosity, \
    interleaving_iterations, FLUX_RESULT_LINKS
from src.stime.cps_search import mdl_error_cp_search
from src.stime.discrepancy_test import DiscrepancyTestType
from src.stime.methods import MethodType, CpsInitializationStrategy

from src.stime.spacetime import SpaceTime, get_options
from src.stime.utils.util_regimes import partition_t, r_partition_to_windows_T

# DAG discovery
def causal_discovery_loc():
    ''' Causal discovery jointly over all locations and all months (here, for 2006 or closest year to it)'''

    dinfo = get_flux_data()
    context_dic = {}
    for ci in dinfo.flux_contexts:
        context_dic[ci] = np.array(dinfo.flux_contexts[ci][:365])
    out_dir = FLUX_OUT_PATH
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    method = MethodType.GP
    spct = SpaceTime(method,
                     CpsInitializationStrategy.BINS,
                     truths=None,
                     assumed_max_lag=assumed_max_lag,
                     assumed_min_dur=assumed_min_dur,
                     initial_bin_size=initial_bin_size,
                     logger=dinfo.log, verbosity=verbosity,
                     interleaving_iterations=interleaving_iterations,
                     out=out_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spct.run(context_dic)

    print(f'*** SpaceTime RESULT *** '
          f'\nDAG {spct.cur_dag_search_result}\nCPS {spct.cur_regimes}'
          f'\nRegimes {spct.cur_partition}\nContexts {spct.cur_c_partition}'
          f'\nRegimes/node {spct.cur_r_partitions}\nContexts/node {spct.cur_c_partitions}')
    return spct


# MDL edge strengths
def mdl_scoring_loc_yr_month():
    """ Compute MDL edge weights for each location, year, and month
    @return:
    """

    dinfo = get_flux_data()

    for ci in dinfo.flux_contexts_years:
        for yr in dinfo.flux_contexts_years[ci]:
            nms = [f'{dinfo.flux_ids[ci]}' for ci in dinfo.flux_ids]
            print('EVALUATING', nms[ci], yr)

            context_dic = {0: np.array(dinfo.flux_contexts_years[ci][yr][:365])}
            loc = 'mdl_loc_year_month_links/'
            write_to = os.path.join('res_flux_mdl_weights/', loc)
            Path(write_to).mkdir(parents=True, exist_ok=True)
            write_to = write_to + f"{ci}_{yr}"

            method = MethodType.GP
            spcetme = SpaceTime(
                method,
                CpsInitializationStrategy.BINS,
                truths=None,
                assumed_max_lag=assumed_max_lag,
                assumed_min_dur=assumed_min_dur,
                initial_bin_size=initial_bin_size,
                logger=dinfo.log, verbosity=verbosity,
                interleaving_iterations=interleaving_iterations,
                out=write_to)

            links = FLUX_RESULT_LINKS
            spcetme.score_under_links(context_dic, links)


# todo
def context_partitioning_loc_year(out_dir='context_loc_each_year'):
    assumed_min_dur = 30  # start from the months
    dinfo = get_flux_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        context_dic = {}
        interesting_contexts = ['GF-Guy', 'DE-Hai', 'FR-Pue', 'US-SRM', 'FI-Hyy']
        interesting_ci = [ci for ci in dinfo.flux_ids if dinfo.flux_ids[ci] in interesting_contexts]
        interesting_cnm = [dinfo.flux_ids[ci] for ci in dinfo.flux_ids if dinfo.flux_ids[ci] in interesting_contexts]
        results = {}
        context_dic = {}
        for ii, ci in enumerate(interesting_ci):
            for i, yr in enumerate(dinfo.flux_contexts_years[ci]):
                if yr == '2006':
                    df = dinfo.flux_contexts_years[ci][yr][:365]

                    context_dic[ii] = np.array(df)
                    print(f'{ci}_{yr}')

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        links = {0:  FLUX_RESULT_LINKS}
        truths = SimpleNamespace(true_links=links)
        method = MethodType.GP_DAG

        print(f'EVALUATING  {interesting_cnm} ')
        spct = SpaceTime(method,
                         CpsInitializationStrategy.BINS,
                         truths=truths,
                         assumed_max_lag=assumed_max_lag,
                         assumed_min_dur=assumed_min_dur,
                         initial_bin_size=initial_bin_size,
                         logger=dinfo.log, verbosity=verbosity,
                         interleaving_iterations=interleaving_iterations,
                         out=out_dir)
        # spct.run(context_dic)
        params_T, params_N = context_dic[0].shape
        params = {'N': params_N, 'T': params_T}
        options = get_options(DiscrepancyTestType.KCD, spct.method_type, params_N, params_T, spct.logger,
                              spct.min_dur)
        n_bin_samples, n_bin_regions = 30, int(np.floor(params_T / 30))
        nb_chunks = int(np.floor(params_T / n_bin_samples))
        partition = partition_t(params_T, n_bin_regions, nb_chunks, n_bin_samples, True)
        windows_T = r_partition_to_windows_T(partition, spct.skip)
        # spct.run(context_dic)

        print("USING WINDOWS:", windows_T)
        r_partition, c_partition, r_partitions, c_partitions = spct.partition_under_regimes(context_dic, links,
                                                                                            windows_T)
        print(f'RESULT FOR {interesting_cnm}: ')
        print(r_partition)
        print(c_partition)

        print(f'C partitions: ')
        print(c_partitions)
        print(f'R partitions: ')
        print(r_partitions)

        results[ci] = SimpleNamespace(r_partition=r_partition, c_partition=c_partition, r_partitions=r_partitions,
                                      c_partitions=c_partitions, yrs=[yr for yr in dinfo.flux_contexts_years[ci]],
                                      loc_name=dinfo.flux_ids[ci])


def cps_discovery_loc_each_year(out_dir='cps_loc_each_year'):
    """
    CPS detection: discover cps and partition into regimes (for each location separately to plot residuals) """
    assumed_min_dur = 30  # start from the months
    dinfo = get_flux_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        context_dic = {}
        interesting_contexts = ['GF-Guy', 'DE-Hai', 'FR-Pue', 'US-SRM', 'FI-Hyy']
        interesting_ci = [ci for ci in dinfo.flux_ids if dinfo.flux_ids[ci] in interesting_contexts]
        interesting_cnm = [dinfo.flux_ids[ci] for ci in dinfo.flux_ids if dinfo.flux_ids[ci] in interesting_contexts]

        results = {}

        for ci in interesting_ci:
            context_dic = {}

            df = dinfo.flux_contexts[ci][:365]
            context_dic = {0: np.array(df)}
            print(f'EVALUATING {ci}. {dinfo.flux_ids[ci]}: ')
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            links = {0: FLUX_RESULT_LINKS}
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
            # spct.run(context_dic)
            params_T, params_N = context_dic[0].shape
            params = {'N': params_N, 'T': params_T}
            options = get_options(DiscrepancyTestType.KCD, spct.method_type, params_N, params_T, spct.logger,
                                  spct.min_dur)
            r_partition, c_partition, c_partitions, r_partitions, error_list = mdl_error_cp_search(
                context_dic, links, params, options, spct.skip, assumed_min_dur, return_residuals=True)
            for node in range(6):
                for cps in range(len(error_list)):
                    plt.scatter(np.arange(0, error_list[cps].shape[0]), error_list[cps][:, node])
                    plt.savefig(f'{ci}_{dinfo.flux_ids[ci]}_node_{node}_window_{cps}_resids')
                    plt.close()
            print(f'RESULT FOR {ci}: ')
            print(r_partition)
            print(c_partition)

            print(f'C partitions: ')
            print(c_partitions)
            print(f'R partitions: ')
            print(r_partitions)

            results[ci] = SimpleNamespace(r_partition=r_partition, c_partition=c_partition, r_partitions=r_partitions,
                                          c_partitions=c_partitions, yrs=[yr for yr in dinfo.flux_contexts_years[ci]],
                                          loc_name=dinfo.flux_ids[ci], residuals=error_list)

        # todo update
        for ci in results:
            print(
                f'LOCATION {results[ci].loc_name}, yr span {min([int(y) for y in results[ci].yrs])}-{max([int(y) for y in results[ci].yrs])}')
            for pi in np.unique([p for _, _, p in results[ci].r_partition]):
                print(f'\tGroup #{pi}')
                print('\t', [yr for i, yr in enumerate(results[ci].yrs) if results[ci].r_partition[i][2] == pi])

        for n in range(6):
            print(f'\tNODE {n}, parents {[j for j in links[0][n] if j != n]}')
            for ci in results:
                print(
                    f'\tLOCATION {results[ci].loc_name}, yr span {min([int(y) for y in results[ci].yrs])}-{max([int(y) for y in results[ci].yrs])}')
                for pi in np.unique(results[ci].r_partitions[n]):
                    print(f'\tGroup #{pi}')
                    print('\t', [yr for i, yr in enumerate(results[ci].yrs) if results[ci].r_partitions[n][i] == pi])

    return results


def regime_discovery_loc_over_years(out_dir='regime_abnormal'):
    """ Partitioning : discover abnormal years per location
    """
    assumed_min_dur = 365  # In this experiment, we compare the years
    dinfo = get_flux_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        context_dic = {}
        interesting_contexts = ['DE-Hai', 'DE-Tha', 'BR-Sa3', 'US-Wkg']
        interesting_ci = [ci for ci in dinfo.flux_ids]  # if dinfo.flux_ids[ci] in interesting_contexts]

        results = {}
        for ci in interesting_ci:
            df = None
            for yr in dinfo.flux_contexts_years[ci]:
                if df is None:
                    df = dinfo.flux_contexts_years[ci][yr][:365]
                else:
                    df = np.vstack([df, dinfo.flux_contexts_years[ci][yr][:365]])
            context_dic = {0: np.array(df)}
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            links = {0: FLUX_RESULT_LINKS}
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
            # spct.run(context_dic)
            params_T = len(context_dic[0])
            n_bin_samples, n_bin_regions = 365, int(np.floor(params_T / 365))
            nb_chunks = int(np.floor(params_T / n_bin_samples))
            partition = partition_t(params_T, n_bin_regions, nb_chunks, n_bin_samples, True)

            windows_T = r_partition_to_windows_T(partition, spct.skip)
            print("USING WINDOWS:", windows_T)
            r_partition, c_partition, r_partitions, c_partitions = spct.partition_under_regimes(context_dic, links,
                                                                                                windows_T)
            print(f'RESULT FOR {ci}: ')
            print(r_partition)
            print(c_partition)

            print(f'C partitions: ')
            print(c_partitions)
            print(f'R partitions: ')
            print(r_partitions)

            results[ci] = SimpleNamespace(r_partition=r_partition, c_partition=c_partition, r_partitions=r_partitions,
                                          c_partitions=c_partitions, yrs=[yr for yr in dinfo.flux_contexts_years[ci]],
                                          loc_name=dinfo.flux_ids[ci])

        for ci in results:
            print(
                f'LOCATION {results[ci].loc_name}, yr span {min([int(y) for y in results[ci].yrs])}-{max([int(y) for y in results[ci].yrs])}')
            for pi in np.unique([p for _, _, p in results[ci].r_partition]):
                print(f'\tGroup #{pi}')
                print('\t', [yr for i, yr in enumerate(results[ci].yrs) if results[ci].r_partition[i][2] == pi])
        dic = {}
        for ci in results:
            for i, yr in enumerate(results[ci].yrs):
                dic[f'{ci}_{yr}'] = [results[ci].r_partitions[p][i] for p in results[ci].r_partitions]

        for n in range(6):
            for ci in results:
                for i, yr in enumerate(results[ci].yrs):
                    dic[f'{ci}_{yr}'] = results[ci].r_partitions[str(n)][i]
            df = pd.DataFrame([dic])
            df.to_csv(f'FLUX_regime_yearly_node_{n}')

        for ci in results:
            for i, yr in enumerate(results[ci].yrs):
                dic[f'{ci}_{yr}'] = results[ci].r_partition[i][2]
        df = pd.DataFrame([dic])
        df.to_csv(f'FLUX_regime_yearly', index=False)

        def to_tex():
            var = 'P_F'
            dr = '../flux_krich_et_al/'
            nm = 'IT-Ren'
            ci = [cj for cj in dinfo.flux_ids if dinfo.flux_ids[cj] == nm][0]
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
                        data_frame.insert(0, "t", [i for i in range(12)], True)
                        data_frame.to_csv(f'flux_texdata/{nm}_{yr}', index=False, sep='\t')
                        # plt.plot(np.arange(0, 12, 1), data_frame[var], label=yr,
                        #         color="gray" if yr != '2003' else "blue")
            # plt.legend()
            # plt.savefig(f'atrajectory_{nm}')
            # plt.close()
    return results



def regime_discovery_loc():
    """
    Regime discovery across all locations
    @return:
    """
    out_dir='regime_loc'
    dinfo = get_flux_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        context_dic = {}
        for ci in dinfo.flux_contexts:
            context_dic[ci] = np.array(dinfo.flux_contexts[ci][:365])

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        links = {0: [((0, -1), 1, None), ((5, 0), 1, None)],
                 1: [((0, 0), 1, None),
                     ((1, -1), 1, None),
                     ((2, 0), 1, None),
                     ((5, 0), 1, None)],
                 2: [((2, -1), 1, None), ((3, 0), 1, None)],
                 3: [((0, 0), 1, None), ((3, -1), 1, None)],
                 4: [((0, 0), 1, None), ((4, -1), 1, None)],
                 5: [((3, 0), 1, None), ((4, 0), 1, None), ((5, -1), 1, None)]}
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

        spct.run(context_dic)
        print(spct.cur_regimes)


