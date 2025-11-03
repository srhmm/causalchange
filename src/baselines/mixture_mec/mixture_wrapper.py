import itertools
import multiprocessing as mp
from mixture_solver import *

def wrapper_get_expt_config(num_nodes, adj_dense_prop, num_samples):
    '''
    '''
    all_expt_config = dict(
        #Graph related parameters
        run_list = list(range(5)), #for random runs with same config, needed?
        num_nodes = num_nodes, #[6],
        max_edge_strength = [1.0,],
        graph_sparsity_method=["adj_dense_prop",],#[adj_dense_prop, use num_parents]
        num_parents = [None],
        adj_dense_prop = adj_dense_prop, # [0.1,0.4,0.6,1.0],
        noise_type=["gaussian"], #"gaussian", or gamma
        obs_noise_mean = [0.0],
        obs_noise_var = [1.0],
        obs_noise_gamma_shape = [None],
        #Intervnetion related related parameretrs
        new_noise_mean= [1.0],
        intv_targets = ["all", "half",], #all, half
        intv_type = ["do"], #hard,do,soft
        new_noise_var = [None],
        #Sample and other statistical parameters
        sample_size = num_samples, #[2**idx for idx in range(10,21)],
        gmm_tol = [1e-3], #1e-3 default #10000,5000,1000 for large nodes
        cutoff_drop_ratio=[0.07]
    )

    #save_dir="all_expt_logs/expt_logs_sim_compsel_backwardbugfixed_cameraready_half_density_var"
    #pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True)
    #wrapper_jobber(all_expt_config,save_dir,num_parallel_calls=1)#64)


#   EXPERIMENT RUNNER (SIMULATION)
def wrapper_mixture_jobber(all_expt_config, save_dir, num_parallel_calls):
    '''
    '''
    # First of all we have to generate all possible experiments
    flatten_args_key = []
    flatten_args_val = []
    for key, val in all_expt_config.items():
        flatten_args_key.append(key)
        flatten_args_val.append(val)

    # Getting all the porblem configs
    problem_configs = list(itertools.product(*flatten_args_val))
    # Now generating all the experimetns arg
    all_expt_args = []

    expt_args_list = []
    for cidx, config in enumerate(problem_configs):
        config_dict = {
            key: val for key, val in zip(flatten_args_key, config)
        }

        args = dict(
            save_dir=save_dir,
            exp_name="{}".format(cidx),
            num_nodes=config_dict["num_nodes"],
            num_tgt_prior=config_dict["num_nodes"] + 1,
            obs_noise_mean=config_dict["obs_noise_mean"],
            obs_noise_var=config_dict["obs_noise_var"],
            obs_noise_gamma_shape=config_dict["obs_noise_gamma_shape"],
            noise_type=config_dict["noise_type"],
            max_edge_strength=config_dict["max_edge_strength"],
            graph_sparsity_method=config_dict["graph_sparsity_method"],
            adj_dense_prop=config_dict["adj_dense_prop"],
            num_parents=config_dict["num_parents"],
            new_noise_mean=config_dict["new_noise_mean"],
            mix_samples=config_dict["sample_size"],
            stage2_samples=config_dict["sample_size"],
            gmm_tol=config_dict["gmm_tol"],
            intv_type=config_dict["intv_type"],
            new_noise_var=config_dict["new_noise_var"],
            dtype="simulation",
            cutoff_drop_ratio=config_dict["cutoff_drop_ratio"],
        )
        if config_dict["intv_targets"] == "all":
            args["intv_targets"] = list(range(config_dict["num_nodes"]))
        elif config_dict["intv_targets"] == "half":
            # Here we will only interven on half the nodes
            nodes_list = list(range(config_dict["num_nodes"]))
            np.random.shuffle(nodes_list)
            half_node_list = nodes_list[0:config_dict["num_nodes"] // 2]
            args["intv_targets"] = half_node_list
        elif type(config_dict["intv_targets"]) == type(1):
            nodes_list = list(range(config_dict["num_nodes"]))
            np.random.shuffle(nodes_list)
            some_node_list = nodes_list[0:config_dict["intv_targets"]]
            args["intv_targets"] = some_node_list
        else:
            raise NotImplementedError
        expt_args_list.append(args)

        # If we want to run sequentially
        # intv_args_dict,metric_dict = run_mixture_disentangle(args)
        # print("=================================================")
        # print("\n\n\n\n\n\n")

    # Running the experiment parallely
    # run_mixture_disentangle(expt_args_list[0])
    with mp.Pool(num_parallel_calls) as p:
        p.map(run_wrapper_mixture_disentangle, expt_args_list)
    print("Completed the whole experiment!")


# SINGLE EXPERIMENT KERNEL
def run_wrapper_mixture_disentangle(args):
    '''
    '''
    # Collecting the metrics to evaluate the results later
    metric_dict = {}

    if args["dtype"] == "simulation":
        # Creating the SCM
        gargs = {}
        gargs["noise_mean_list"] = [args["obs_noise_mean"], ] * args["num_nodes"]
        gargs["noise_var_list"] = [args["obs_noise_var"], ] * args["num_nodes"]
        scmGen = RandomSCMGenerator(num_nodes=args["num_nodes"],
                                    max_strength=args["max_edge_strength"],
                                    num_parents=args["num_parents"],
                                    args=args,
                                    )
        gSCM = scmGen.generate_gaussian_scm(scm_args=gargs)

        # Step 0: Generating the samples and interventions configs
        print("Generating mixture samples!")
        intv_args_dict, mixture_samples = gSCM.generate_gaussian_mixture(
            args["intv_type"],
            args["intv_targets"],
            args["new_noise_mean"],
            args["new_noise_var"],
            args["mix_samples"],
            args["noise_type"],
            args["obs_noise_gamma_shape"],
        )
    elif args["dtype"] == "sachs":
        # Getting the samples and the intervention configs
        print("Generating the mixture sample")
        intv_args_dict, mixture_samples, num_nodes = generate_mixture_sachs(
            args["dataset_path"],
            args["mix_samples"],
        )
        args["num_nodes"] = num_nodes
    else:
        raise NotImplementedError()

    # Step 1: Running the disentanglement
    print("Step 1: Disentangling Mixture")
    gSolver = GaussianMixtureSolver(args["dtype"])
    # We will allow number of component = n+1 (hopefully it will find zero weight)
    err, intv_args_dict, weight_precision_error, est_num_comp, gm_score_dict \
        = gSolver.mixture_disentangler(
        args["num_tgt_prior"],
        intv_args_dict,
        mixture_samples,
        args["gmm_tol"],
        args["cutoff_drop_ratio"],
    )
    metric_dict["param_est_rel_err"] = err
    metric_dict["est_num_comp"] = est_num_comp
    metric_dict["weight_precision_error"] = weight_precision_error
    metric_dict["gm_score_dict"] = gm_score_dict
    print("error:", err)
    print("weight precision error:", metric_dict["weight_precision_error"])

    # Step 2: Finding the graph for each component
    print("Stage 2: Estimating individual graph using PC")
    # gSolver.run_pc_over_each_component(intv_args_dict,args["stage2_samples"])
    est_dag, intv_args_dict, oracle_est_dag, igsp_est_dag, intv_base_est_dag \
        = gSolver.identify_intervention_utigsp(
        intv_args_dict, args["stage2_samples"])
    metric_dict["est_dag"] = est_dag
    metric_dict["oracle_est_dag"] = oracle_est_dag
    metric_dict["igsp_dag"] = igsp_est_dag
    metric_dict["intv_base_est_dag"] = intv_base_est_dag

    # Evaluation:
    print("==========================")
    print("Estimated Target List")
    for comp in intv_args_dict.keys():
        if "left_comp" in comp:
            print("actual_tgt:{}\test_tgt:{}".format(
                comp,
                intv_args_dict[comp]["est_tgt"],
            )
            )
        elif "est_tgt" in intv_args_dict[comp]:
            print("actual_tgt:{}\test_tgt:{}\toracle_tgt:{}".format(
                comp,
                intv_args_dict[comp]["est_tgt"],
                intv_args_dict[comp]["oracle_est_tgt"]
            )
            )
        else:
            # All the targets will have the oracle targets
            print("actual_tgt:{}\toracle_tgt:{}".format(
                comp,
                intv_args_dict[comp]["oracle_est_tgt"]
            )
            )
    # Computing the SHD
    print("==========================")
    metric_dict = compute_shd(intv_args_dict, metric_dict)
    print("SHD:", metric_dict["shd"])
    print("Oracle-SHD:", metric_dict["oracle_shd"])
    print("Intv Base-SHD:", metric_dict["intv_base_shd"])
    print("IGSP-SHD:", metric_dict["igsp_shd"])
    print("actgraph:\n", metric_dict["act_dag"])
    print("est graph:\n", metric_dict["est_dag"])
    print("oracle est graph:\n", metric_dict["oracle_est_dag"])
    print("igsp est graph:\n", metric_dict["igsp_dag"])

    # Computing the js of target
    print("==========================")
    metric_dict = compute_target_jaccard_sim(intv_args_dict, metric_dict, args["dtype"])
    print("Avg JS:", metric_dict["avg_js"])
    print("Avg Oracle JS:", metric_dict["avg_oracle_js"])
    # print("Avg Intv Base JS:",metric_dict["avg_intv_base_js"])

    # Dumping the experiment
    pickle_experiment_result_json(args, intv_args_dict, metric_dict)
    return intv_args_dict, metric_dict
