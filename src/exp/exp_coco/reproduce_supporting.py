import numpy as np

from src.exp.exp_coco.method_types import MethodType
from src.exp.exp_coco.run_coco import run_coco


def reproduce_supporting_clustering(path, testing=False):
    reps = 20 if not testing else 2
    METHODS = [MethodType.ORACLE]  # [MethodType.COCO]
    base_n_nodes, base_n_confounders, base_n_contexts, base_shift_X, base_shift_Z = (10, 1, 10, 1, 2)

    show_result = ""
    identifier = lambda i: f'{base_n_nodes}_{i}_{base_n_contexts}_{base_shift_X}_{base_shift_Z}'
    CASE = {identifier("i"):
                {str(i): (
                base_n_nodes, i, base_n_contexts, base_shift_X, base_shift_Z, 'test_case')
                for i in range(int(np.floor(base_n_nodes/2)))}}
    res = run_coco(reps,
                   CASE, METHODS,
                   known_componets=False,
                   path=path, writing=not testing)

    for metric in ['f1']: #['f1', 'tpr', 'fpr']:
        show_result += res.write_methods_tex(identifier("i"), path, CASE[identifier], METHODS,
                                             fscore=metric, sigscore=f'{metric}sig')


def reproduce_supporting_causal(path, testing=False):
    reps = 20 if not testing else 2
    METHODS = [MethodType.ORACLE]# [MethodType.COCO]
    base_n_nodes, base_n_confounders, base_n_contexts, base_shift_X, base_shift_Z = (10, 1, 10, 1, 2)

    show_result = ""
    identifier= f'{base_n_nodes}_{base_n_confounders}_{base_n_contexts}_{base_shift_X}_{base_shift_Z}'
    CASE = {identifier:
                {'test_case': (base_n_nodes, base_n_confounders, base_n_contexts, base_shift_X, base_shift_Z, 'test_case')}}
    res = run_coco(reps, CASE, METHODS, path=path, writing=not testing)


    for metric in ['f1', 'tpr', 'fpr']:
        show_result += res.write_methods_tex(identifier, path, CASE[identifier], METHODS,
                                         fscore=metric, sigscore=f'{metric}sig')
    print('')


def reproduce_supporting_emp_significance_power(path,
                                                show_nc=True,
                                                show_nx=True,
                                                show_sx=True,
                                                show_nz=True,
                                                show_sz=True, testing=False):
    reps = 2 if testing else 20
    METHODS = [MethodType.ORACLE]
    base_n_nodes, base_n_confounders, base_n_contexts, base_shift_X, base_shift_Z = (10, 1, 10, 1, 2)
    show_result = ""

    # a) Number of contexts
    CASES_CONTEXTS = {
        f'{base_n_nodes}_{base_n_confounders}_i_{base_shift_X}_{base_shift_Z}':
            {str(i): (base_n_nodes, base_n_confounders, i, base_shift_X, base_shift_Z, 'n_c')
             for i in range(3, base_n_contexts + 1)},
        f'{base_n_nodes}_{base_n_confounders}_j_{base_shift_X}_{base_shift_Z}':
            {str(i): (base_n_nodes, base_n_confounders, i, base_shift_X, base_shift_Z, 'n_c') for i in [15, 20]}
    }
    if show_nc:
        res = run_coco(reps, CASES_CONTEXTS,
                       METHODS, path=path, writing=not testing)
        for identifier in CASES_CONTEXTS:
            for metric in ['f1', 'tpr', 'fpr']:
                show_result += "\n" + identifier + f"\nMetric: {metric}\n"
                show_result += res.write_methods_tex(identifier, path, CASES_CONTEXTS[identifier], METHODS,
                                                     fscore=metric, sigscore=f'{metric}sig')
    # b) Number of variables
    CASES_VARS = {
        f'i_{base_n_confounders}_{base_n_contexts}_{base_shift_X}_{base_shift_Z}':
            {str(i): (i, base_n_confounders, base_n_contexts, base_shift_X, base_shift_Z, 'n_x')
             for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]} }
    if show_nx:
        res = run_coco(reps, CASES_VARS, METHODS, path=path, writing=not testing)
        for identifier in CASES_VARS:
            for metric in ['f1', 'tpr', 'fpr']:
                show_result += "\n" + identifier + f"\nMetric: {metric}\n"
                show_result += res.write_methods_tex(identifier, path, CASES_VARS[identifier], METHODS,
                                                     fscore=metric, sigscore=f'{metric}sig')

    #c) Number of mechanism shifts
    CASES_SHIFTX = {
        f'{base_n_nodes}_{base_n_confounders}_{base_n_contexts}_i_i+1':
            {str(i): (base_n_nodes, base_n_confounders, base_n_contexts, i, i+1, 's_xz_') for i in
             range(0, base_n_contexts - 1)},
        f'{base_n_nodes}_{base_n_confounders}_{base_n_contexts}_9_9': {
            str(9): (base_n_nodes, base_n_confounders, base_n_contexts, 9, 9, 's_xz_')}
    }
    if show_sx:
        res = run_coco(reps, CASES_SHIFTX, METHODS, path=path, writing=not testing)
        for identifier in CASES_SHIFTX:
            for metric in ['f1', 'tpr', 'fpr']:
                show_result += "\n" + identifier +f"\nMetric: {metric}\n"
                show_result += res.write_methods_tex(identifier, path, CASES_SHIFTX[identifier], METHODS,
                                                 fscore=metric, sigscore=f'{metric}sig')

        print(show_result)

    # d) Number of confounders
    CASES_CONFOUNDERS = {
        f'{base_n_nodes}_i_{base_n_contexts}_{base_shift_X}_{base_shift_Z}':
            {str(i): (base_n_nodes, i, base_n_contexts, base_shift_X, base_shift_Z, 'n_z')
             for i in range(int(base_n_nodes / 2) + 1)}
    }
    if show_nc:
        res = run_coco(reps, CASES_CONFOUNDERS,
                       METHODS, path=path, writing=not testing)
        for identifier in CASES_CONFOUNDERS:
            for metric in ['f1', 'tpr', 'fpr']:
                show_result += "\n" + identifier + f"\nMetric: {metric}\n"
                show_result += res.write_methods_tex(identifier, path, CASES_CONFOUNDERS[identifier],
                                                     METHODS,
                                                     fscore=metric, sigscore=f'{metric}sig')

    # e) Number of mechanism shifts of the confounders
    CASES_SHIFTZ = {
        f'{base_n_nodes}_{base_n_confounders}_{base_n_contexts}_{base_shift_X}_i':
            {str(i): (base_n_nodes, base_n_confounders, base_n_contexts, base_shift_X, i, 's_z_') for i in
             range(0, base_n_contexts)}
    }
    if show_sz:
        res = run_coco(reps, CASES_SHIFTZ, METHODS, path=path, writing=not testing)
        for identifier in CASES_SHIFTZ:
            for metric in ['f1', 'tpr', 'fpr']:
                show_result += "\n" + identifier + f"\nMetric: {metric}\n"
                show_result += res.write_methods_tex(identifier, path, CASES_SHIFTZ[identifier], METHODS,
                                                     fscore=metric, sigscore=f'{metric}sig')

    print(show_result)


def reproduce_supporting_sparse_shifts(path, n_c=20, testing=False):
    METHODS = [MethodType.ORACLE]
    reps = 2 if testing else 20

    # pattern: (n_observed_vars, n_latent_vars, n_contexts, n_shifts_observed, n_shifts_latent)
    # Vary the mechanism shifts of all variables
    vary_i = [str(xi) for xi in range(n_c)]
    vary_ii = {f'5_1_{n_c}_{i}_{i}': {str(i): (5, 1, n_c, i, i, f'sx_{i}_sz_{i}') } for i in
                 range(n_c)}
    identifier_ii = f'5_1_{n_c}_ixi'
    res_ii = run_coco(reps, vary_ii, METHODS, exact_shifts=True, path=path, writing=not testing)

    res_ii.write_keys_tex(identifier_ii, path, vary_ii, vary_i)

    # Vary the mechanism shifts of observed variables
    s_x_max, s_z_max = n_c, n_c
    vary_x_x = [str(xi) for xi in range(s_x_max)]
    vary_xy_x = {f'5_1_{n_c}_sx_{i}': {str(j): (5, 1, n_c, j, i, f'sx_{i}_sz_{j}') for j in range(s_x_max)} for i in range(s_z_max)}
    identifier_x = f'5_1_{n_c}_{s_x_max}x{s_z_max}_by_sx'

    res_shiftx = run_coco(reps, vary_xy_x, METHODS, exact_shifts=True, path=path, writing=not testing)

    # Vary the mechanism shifts of latent variables
    vary_x_z = [str(xi) for xi in range(s_x_max)]
    vary_xy_z = {f'5_1_{n_c}_{i}_sz': {str(j): (5, 1, n_c, i, j, f'sx_{j}_sz_{i}') for j in range(s_x_max)} for i in
               range(s_z_max)}
    identifier_z = f'5_1_{n_c}_{s_x_max}x{s_z_max}_by_sz'

    res_shiftz = run_coco(reps, vary_xy_z, METHODS, exact_shifts=True, path=path, writing=not testing)

    for metric in ['f1', 'tpr', 'fpr']:
        res_ii.write_keys_tex(identifier_ii, path, vary_ii, vary_i, fscore=metric, sigscore=f'{metric}sig')
        res_shiftx.write_keys_tex(identifier_x, path, vary_xy_x, vary_x_x, fscore=metric, sigscore=f'{metric}sig')
        res_shiftz.write_keys_tex(identifier_z, path, vary_xy_z, vary_x_z,  fscore=metric, sigscore=f'{metric}sig')