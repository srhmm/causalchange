import os

import numpy as np

from src.coco.util.co_test_types import CoCoTestType, CoShiftTestType, CoDAGType
from src.coco.coco import CoCo
from src.coco.gen.dag_gen import _random_nonlinearity
from src.exp.exp_coco.fci import FCI_JCI
from src.coco.gen.dag_confounded import DAGConfounded
from src.coco.util.mi_sampling import Sampler
from src.exp.exp_coco.results_coco import ResultsCoCo, MethodType


def run_coco(reps,
             test_cases,
             METHODS=[m for m in MethodType],
             SHIFT_TEST=CoShiftTestType.PI_KCI,
             FCI_INDEP_TEST = 'fisherz',
             bivariate=False,
             exact_shifts=False,
             known_componets=True,
             path='',
             writing=True):

    CONFOUNDING_TEST = CoCoTestType.MI_ZTEST
    FUN_FORMS = [(_random_nonlinearity(), "NLIN")]

    sampler = Sampler()
    D_n = 500

    for fun_form, fun_str in FUN_FORMS:
        exp_identifier = str(SHIFT_TEST) + '_' + str(CONFOUNDING_TEST)  + "_biv_"+ str(bivariate)+'_known_n_z_' + str(known_componets) + '_' + fun_str + '_' + str(D_n)
        fl = f"{path}/out_coco/"

        path_write = f'{fl}{exp_identifier}_log.csv'
        path_final = f'{path}{exp_identifier}_res.csv'

        if writing:
            if not os.path.exists(fl):
                os.makedirs(fl)

            write_file = open(path_write, "a+")
            write_final_file = open(path_final, "a+")

            for fl in [write_file, write_final_file]:
                fl.write("\n\nEXPERIMENT: " + exp_identifier)
                fl.write("\nTESTS: " + str(CONFOUNDING_TEST) + " x " + str(SHIFT_TEST))
                fl.write("\nMETHODS: " + str([str(m) for m in METHODS]))
                fl.flush()
            write_file.close()
            write_final_file.close()
        res = ResultsCoCo(writing)

        # Each metric, could try other tests for confounding here, but MI_ZTEST works well
        metric = 'mi'

        # Each pattern, for example (X, ncf, nc, nsx, nsy) for varying nv X
        for pattern in test_cases:
            print(f'PATTERN: (nv, ncf, nc, nsx, nsy)={pattern}')
            # Each case, fixes (nv, ncf, nc, nsx, nsy)
            for xnm in test_cases[pattern]:
                (n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders, nm) = test_cases[pattern][xnm]
                seed = 0
                print(f'CASE: i={xnm}, (nv, ncf, nc, nsx, nsy)={res.get_key(n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders)}')# {n_nodes}_{n_confounders}_{n_contexts}_{n_shifts_observed}_{n_shifts_confounders}')

                for rep in range(reps):
                    print(f"REP. {rep}")
                    print(f'CASE: i={xnm}, (nv, ncf, nc, nsx, nsy)={res.get_key(n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders)}')# {n_nodes}_{n_confounders}_{n_contexts}_{n_shifts_observed}_{n_shifts_confounders}')
                    print(f'COMPONENTS:', n_confounders)
                    seed += 1
                    np.random.seed(seed)

                    list_n_shifts_observed = [0, n_shifts_observed]
                    if exact_shifts:
                        list_n_shifts_observed = [n_shifts_observed]
                    list_n_shifts_confounders = [n_shifts_confounders]
                    if bivariate:
                        n_components = 1
                        list_n_confounded_nodes = [2]
                        if rep % 2 == 0:  # np.random.uniform(0, 1) < 0.5:
                            list_n_confounded_nodes = [0]  # or empty
                        dag = DAGConfounded(seed, n_contexts, 2, 1, list_n_confounded_nodes,
                                            list_n_shifts_observed,
                                            list_n_shifts_confounders, is_bivariate=True
                                            )

                        D, Dobs = dag.gen_data(seed, D_n, _functional_form=fun_form, oracle_partition=True,
                                               noise_iv=False )
                    else:
                        if known_componets:
                            n_components = n_confounders
                        else:
                            n_components = None #CoCo should find the number of confounders/components
                        remaining = n_nodes - n_confounders*2
                        remaining_nodes = n_nodes
                        if remaining < 0 :
                            raise ValueError('More confounders than node pairs')
                        list_n_confounded_nodes = []
                        for _ in range(n_confounders):
                            if remaining <= 0:
                                size = 2
                            else:
                                size = 2 + np.random.choice([i for i in range(remaining + 1)], 1)[0]
                            list_n_confounded_nodes.append(size)
                            remaining = remaining - size
                            remaining_nodes = remaining_nodes - size

                        #print("CFD.", n_confounders, "over", n_nodes, "nodes:", list_n_confounded_nodes)
                        list_n_shifts_observed = [0, n_shifts_observed]
                        if exact_shifts:
                            list_n_shifts_observed = [n_shifts_observed]
                        list_n_shifts_confounders = [n_shifts_confounders]

                        dag = DAGConfounded(seed, n_contexts, n_nodes, n_confounders, #list_n_confounded_nodes, list_n_shifts_observed, list_n_shifts_confounders,
                                            n_shifts_observed, n_shifts_confounders,
                                            is_bivariate=False
                                            )

                        D, Dobs = dag.gen_data(seed, D_n, _functional_form=fun_form, oracle_partition=True,
                                               noise_iv=False )
                    #results for this rep.
                    RESULTS = {metric: {str(MethodType.ORACLE) : None}}

                    #FCI family
                    for method in METHODS:
                        if not method.is_fci():
                            continue
                        print("METHOD:", str(method))
                        fci_jci, D, Dobs, seed = run_fci(dag, D, Dobs, D_n, fun_form, method, FCI_INDEP_TEST, seed) #validates data
                        RESULTS[metric][str(method)] = fci_jci

                    #MSS
                    if MethodType.MSS in METHODS:
                        print("METHOD:", str(MethodType.MSS))

                        mss = CoCo(D, dag.G.nodes,
                                   co_test=CoCoTestType.SKIP,
                                   shift_test=CoShiftTestType.SKIP,
                                   dag_discovery=CoDAGType.MSS,
                                   sampler=sampler, n_components=n_components, dag=dag)#TODO omit arg dag and n_components?
                        RESULTS[metric][str(MethodType.MSS)] = mss
                    #Ours
                    if True in [m.is_coco() for m in METHODS]:
                        if MethodType.COCO in METHODS:
                            print(f"METHOD: {str(MethodType.COCO)}\nTESTS: {str(CoDAGType.MSS)} x {str(SHIFT_TEST)} x {CONFOUNDING_TEST}" )
                            coco = CoCo(D, dag.G.nodes,
                                        co_test=CONFOUNDING_TEST, shift_test=SHIFT_TEST, dag_discovery=CoDAGType.MSS,
                                        sampler=sampler, n_components=n_components, dag=dag)  #TODO omit arg dag?
                            RESULTS[metric][str(MethodType.ORACLE)] = coco
                            RESULTS[metric][str(MethodType.COCO)] = coco

                        if MethodType.ORACLE_DAG in METHODS :
                            print(f"METHOD: {str(MethodType.ORACLE_DAG)}\nTESTS: {str(CoDAGType.SKIP)} x {str(SHIFT_TEST)} x {CONFOUNDING_TEST}" )
                            cocoracle = CoCo(D, dag.G.nodes,
                                        co_test=CONFOUNDING_TEST, shift_test=SHIFT_TEST,
                                        dag_discovery=CoDAGType.SKIP,
                                        sampler=sampler, n_components=n_components, dag=dag)
                            RESULTS[metric][str(MethodType.ORACLE)] = cocoracle
                            RESULTS[metric][str(MethodType.ORACLE_DAG)] = cocoracle
                        elif MethodType.ORACLE in METHODS and not(MethodType.COCO in METHODS):
                            print(f"METHOD: {str(MethodType.ORACLE)}\nTESTS: {str(CoDAGType.SKIP)} x {str(CoShiftTestType.SKIP)} x {CONFOUNDING_TEST}" )
                            oracle = CoCo(D, dag.G.nodes,
                                          co_test=CONFOUNDING_TEST, shift_test=CoShiftTestType.SKIP,
                                          dag_discovery=CoDAGType.SKIP,
                                          sampler=sampler, n_components=n_components, dag=dag)   #TODO dont need dag
                            RESULTS[metric][str(MethodType.ORACLE)] = oracle

                    res.update(dag, n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders, RESULTS, METHODS)
                res.write(path_write, n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders, [metric], METHODS)
        res.write_final(path_final, [metric], METHODS)

        return res


def run_fci(dag, D, Dobs, D_n, fun_form, method,
            FCI_INDEP_TEST, seed):

    val_error = False
    try:
        fci_jci = FCI_JCI(Dobs, dag.G, dag.G_true, dag,
                                               independence_test=FCI_INDEP_TEST,
                                               method=method)
    except ValueError:
        val_error = True
    it = 0
    while (val_error and it < 1000):
        it += 1
        seed = int(np.random.uniform())
        np.random.seed(seed)
        D, Dobs = dag.gen_data(seed, D_n, _functional_form=fun_form, oracle_partition=True,
                               noise_iv=False)
        val_error = False
        try:
            fci_jci = FCI_JCI(Dobs, dag.G, dag.G_true, dag,
                                                   independence_test=FCI_INDEP_TEST,
                                                   method=method)
        except ValueError:
            val_error = True
    if val_error:
        raise Exception("singular matrix")

    seed = seed + 1
    return fci_jci, D, Dobs, seed