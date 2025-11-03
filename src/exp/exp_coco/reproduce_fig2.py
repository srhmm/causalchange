from src.exp.exp_coco.method_types import all_methods, all_coco_methods
from src.exp.exp_coco.run_coco import run_coco


def reproduce_fig2(path, METHODS=None, testing=False):
    reps = 2 if testing else 20
    if METHODS is None:
        METHODS = all_coco_methods() if testing else all_methods()
    base_n_nodes, base_n_confounders, base_n_contexts, base_shift_X, base_shift_Z = (5, 1, 10, 1, 2)
    c_min, c_max = 3, 10
    show_result = ""

    # a) Number of contexts
    CASES_CONTEXTS = {
        f'{base_n_nodes}_{base_n_confounders}_i_1_2':
            {str(i): (base_n_nodes, base_n_confounders, i, 1, 2, 'n_c')
             for i in range(c_min, c_max + 1)},
        f'{base_n_nodes}_{base_n_confounders}_j_1_2':
            {str(i): (base_n_nodes, base_n_confounders, i, 1, 2, 'n_c') for i in [15, 20]}
    }
    res = run_coco(reps, CASES_CONTEXTS,
                   METHODS, path=path, writing=not testing)
    for identifier in CASES_CONTEXTS:
        show_result +=  "\n" +identifier +"\n"
        show_result += res.write_methods_tex(identifier, path + "/tex_coco",
                                             CASES_CONTEXTS[identifier], METHODS)


    # b) Number of nodes in G
    fewer_contexts = 6
    node_max = 10
    CASES_NODES = {
        f"i_{base_n_confounders}_{fewer_contexts}_1_2":
            {str(i): (i, base_n_confounders, base_n_contexts, base_shift_X, base_shift_Z, 'n_nodes_')
             for i in range(2, node_max + 1)},
        f"j_{base_n_confounders}_{fewer_contexts}_1_2": {
            str(i): (i, base_n_confounders, base_n_contexts, 1, 2, 'n_nodes_') for i in [15, 20]}

    }
    res = run_coco(reps, CASES_NODES, METHODS, path=path, writing=not testing)
    for identifier in CASES_NODES:
        show_result += "\n" +identifier +"\n"
        show_result += res.write_methods_tex(identifier, path + "/tex_fci",
                                             CASES_NODES[identifier], METHODS)




    #c) Number of shifts (observed variables)
    # in the figure, we show these in reverse order, with increasing (n_contexts - s_x)
    # the latent variable shifts one more, s_z=s_x+1
    CASES_SHIFTX = {
        f'{base_n_nodes}_{base_n_confounders}_{base_n_contexts}_i_i+1':
            {str(i): (base_n_nodes, base_n_confounders, base_n_contexts, i, i+1, 'n_xzshift_') for i in
                  range(0, base_n_contexts-1)},
        f'{base_n_nodes}_{base_n_confounders}_{base_n_contexts}_9_9': {
            str(9): (base_n_nodes, base_n_confounders, base_n_contexts, 9, 9, 'n_xzshift_')}
    }

    res = run_coco(reps, CASES_SHIFTX, METHODS, path=path, writing=not testing)
    for identifier in CASES_SHIFTX:
        show_result += "\n" + identifier + "\n"
        show_result += res.write_methods_tex(identifier, path + "/tex_coco",
                                             CASES_SHIFTX[identifier], METHODS)
    print(show_result)

    #d) Number of shifts (latent variables)
    CASES_SHIFTZ = {
        f'{base_n_nodes}_{base_n_confounders}_{base_n_contexts}_0_i':
            {str(i): (base_n_nodes, base_n_confounders, base_n_contexts, 0, i, 'n_zshift_') for i in
                  range(0, base_n_contexts)}
      }

    res = run_coco(reps, CASES_SHIFTZ, METHODS, path=path, writing=not testing)
    for identifier in CASES_SHIFTZ:
        show_result += "\n" +identifier +"\n"
        show_result += res.write_methods_tex(identifier, path + "/tex_coco",
                                             CASES_SHIFTZ[identifier], METHODS)
