import random
from enum import Enum

import networkx as nx
import numpy as np
from numpy.random import SeedSequence

from src.exp.exp_change.gen.synthetic.data_gen_context import DataGenContext
from src.exp.exp_change.gen.synthetic.data_gen_mixing import DataGen

""" synthetic data generation """

class GenDataType(Enum):
    """type of synthetic data: synthetic data from iid distribution, mult context, w unknown mixing"""

    IID = 'iid'
    MULTI_CONTEXT = 'context'
    TIME = 'time'
    MIXING = 'mixing'

    def __eq__(self, other): return self.value == other.value


class DagType(Enum):
    ERDOS = 'erdos_renyi'
    SCALE_FREE = 'scale_free'
    RANDOM = 'random'

    def __eq__(self, other): return self.value == other.value


class GSType(Enum):
    """ preset graph structures """
    GRAPH = 'graph'  # DagType
    BIV_CAUSAL = 'biv_causal_only'
    BIV_CAUSAL_CONFD = 'biv_causal_confd'
    BIV_CAUSAL_CHANGEY = 'biv_causal_chy'
    BIV_CAUSAL_CHANGEX = 'biv_causal_chx'
    BIV_CAUSAL_INDEPCHANGE = 'biv_causal_chxchy'
    BIV_REV = 'biv_rev_only'
    BIV_REV_CONFD = 'biv_rev_confd'
    BIV_REV_CHANGEY = 'biv_rev_chy'
    BIV_REV_CHANGEX = 'biv_rev_chx'
    BIV_REV_INDEPCHANGE = 'biv_rev_chxchy'
    BIV_CONFD = 'biv_confd_only'
    BIV_CHANGEY = 'biv_chy_only'
    BIV_CHANGEX = 'biv_chx_only'
    BIV_INDEPCHANGE = 'biv_chxchy_only'
    BIV_INDEP = 'biv_indep'
    MV_CAUSAL = 'mv_causal'
    MV_CAUSAL_CONFD = 'mv_causal_confd'
    MV_CONFD = 'mv_confd'
    MV_INDEP = 'mv_indep'

    # could add MV rev but adds too much clutter, rather evaluate when generating random dags.
    def __eq__(self, other): return self.value == other.value
    def __str__(self): return self.value

    # deciding edges among system variables:
    def is_bivariate(self): return self.value.startswith("biv")

    def is_mvariate(self): return self.value.startswith("mv")

    def is_causal(self): return self.value in [
        GSType.BIV_CAUSAL.value, GSType.BIV_CAUSAL_CONFD.value,
        GSType.BIV_CAUSAL_CHANGEY.value, GSType.BIV_CAUSAL_CHANGEX.value, GSType.BIV_CAUSAL_INDEPCHANGE.value,
        GSType.MV_CAUSAL.value, GSType.MV_CAUSAL_CONFD.value
        # GSType.MV_CAUSAL_CHANGEX.value, GSType.MV_CAUSAL_CHANGEY.value
    ]

    def is_anticausal(self): return self.value in [
        GSType.BIV_REV.value, GSType.BIV_REV_CONFD.value,
        GSType.BIV_REV_CHANGEY.value, GSType.BIV_REV_CHANGEX.value, GSType.BIV_REV_INDEPCHANGE.value]

    # deciding edges of interv variables to system variables:
    def only_Y_changes(self): return self.value in [GSType.BIV_CAUSAL_CHANGEY.value, GSType.BIV_REV_CHANGEY.value,
                                                    GSType.BIV_CHANGEY.value]

    def only_X_changes(self): return self.value in [GSType.BIV_CAUSAL_CHANGEX.value, GSType.BIV_REV_CHANGEX.value,
                                                    GSType.BIV_CHANGEX.value]

    def both_change(self): return self.value in [GSType.BIV_CONFD.value, GSType.BIV_CAUSAL_CONFD.value,
                                                 GSType.BIV_REV_CONFD.value, GSType.MV_CAUSAL_CONFD.value,
                                                 GSType.MV_CONFD.value]

    def indep_change(self): return self.value in [GSType.BIV_CAUSAL_INDEPCHANGE.value, GSType.BIV_REV_INDEPCHANGE.value,
                                                  GSType.BIV_INDEPCHANGE.value]

    def neither_change(self): return self.value in [GSType.BIV_INDEP.value, GSType.BIV_CAUSAL.value,
                                                    GSType.BIV_REV.value,
                                                    GSType.MV_INDEP.value, GSType.MV_CAUSAL.value]

    def is_confounded(self): return self.value in [GSType.BIV_CAUSAL_CONFD.value, GSType.BIV_REV_CONFD.value,
                                                   GSType.BIV_CONFD.value,
                                                   GSType.MV_CONFD.value, GSType.MV_CAUSAL_CONFD.value]


class IvType(Enum):
    COEF = 'coef'
    FLIP = 'flip'
    SHIFT = 'shift'

    def __eq__(self, other): return self.value == other.value


class FunType(Enum):
    LIN = 'lin'
    QUAD = 'quad'
    CUB = 'cub'
    EXP = 'exp'
    LOG = 'log'
    SIN = 'sin'
    MIX = 'mix'

    def __eq__(self, other): return self.value == other.value


class NoiseType(Enum):
    GAUSS = 'normal'
    EXP = 'exp'
    GUMBEL = 'gumbel'
    UNIF = 'unif'
    MIX = 'mix'

    def __eq__(self, other): return self.value == other.value


def gen_data_type(dataparams, seed, vb=0, lg=None, ret_params=False):
    params = dataparams.copy()  # change the number of nodes

    seedseq = SeedSequence(seed)
    random_state = np.random.default_rng(seedseq)

    iv_mode = params["IVM"]

    graph = gen_graph_structure_or_chain(params)

    data, truths, params = gen_synthetic_data(iv_mode, params, graph, random_state, seed, vb=vb, lg=lg)
    return (data, truths, params) if ret_params else (data, truths)  # return params to be transparent?


def gen_random_intervention_targets(params, graph, random_state):
    """ random setting, each node affected at most one Z but mult nodes can be affected by same Z """

    frac_confounded = params["PZ"]
    n_confounded = np.floor(frac_confounded * params["N"]).astype(int)
    n_confounders = params["NZ"]

    if n_confounders == 0: return []
    confd_nodes = sorted(random.sample(range(params["N"]), n_confounded))
    confd_sets = []
    base_size = n_confounded // n_confounders
    extra = n_confounded % n_confounders

    start_idx = 0
    for i in range(n_confounders):
        set_size = base_size + (1 if i < extra else 0)
        # Z can also confound single nodes ("intervention variables")
        confd_sets.append(confd_nodes[start_idx:start_idx + set_size])
        start_idx += set_size

    intervention_target_nodes = confd_sets
    return intervention_target_nodes


def gen_atomic_intervention_targets(params, graph, random_state):
    """ controlled setting: assures each node intervened on in at least one context (if possible) """

    if params["K"] != 2: print(f"Warning: K={params['K']} ignored, K=2 for {params['C']} contexts")
    SKIP_OBSERVATIONAL = 1
    one_per_node = True
    intervention_nb = 1
    intervention_targets = dict.fromkeys(set(range(params["C"])))
    intervention_target_nodes = dict.fromkeys(set(range(params["C"])))
    print(f"atomic setting: PZ {params['PZ']} ignored")

    # choose an intervened context for each node
    choices = random_state.choice(
        list(range(params["N"])),
        size=min(intervention_nb * (params["C"] - SKIP_OBSERVATIONAL), params["N"]),
        replace=False,
    )

    # for the remaining contexts, distribute nodes arbitrarily
    rest = intervention_nb * (params["C"] - SKIP_OBSERVATIONAL) - params["N"]
    remaining_choices = []
    if rest > 0:
        remaining_choices = random_state.choice(
            list(range(params["N"])),
            size=rest,
            replace=True,
        )
        choices = np.concatenate([choices, remaining_choices])
    ct = 0
    if one_per_node:
        SKIP_OBSERVATIONAL += len(remaining_choices)
    for c in range(params["C"]):
        intervention_targets[c] = []
        intervention_target_nodes[c] = []
    for c in range(SKIP_OBSERVATIONAL, params["C"]):
        for ib in range(intervention_nb):
            if choices[ct] not in intervention_target_nodes[c]:
                intervention_target_nodes[c].append(choices[ct])
            for arc in graph.edges:
                if arc[1] == choices[ct]:
                    intervention_targets[c].append(arc)
            ct += 1
    return intervention_target_nodes


def gen_erdos_graph(params: dict, depth=1000) -> nx.DiGraph:
    if depth <= 0: raise ValueError("graph cannot be generated")

    p_connectivity = params['P']
    assert 0 <= p_connectivity <= 1
    nodes = params["N"]
    causal_order = np.random.permutation(np.arange(nodes))
    adj_mat = np.zeros((nodes, nodes))
    for i in range(nodes - 1):
        node = causal_order[i]
        possible_parents = causal_order[(i + 1):]
        num_parents = np.random.binomial(
            n=nodes - i - 1, p=p_connectivity)
        parents = np.random.choice(
            possible_parents, size=num_parents, replace=False)
        adj_mat[parents, node] = 1

    try:
        g = nx.DiGraph(adj_mat)
        assert not list(nx.simple_cycles(g))

    except AssertionError:
        return gen_erdos_graph(params, depth=depth - 1)

    # import cdt.data as tb
    # generator = tb.AcyclicGraphGenerator( #causal mechanism and noise not relevant here.
    #    'polynomial', nodes=params["N"], npoints=1,
    #    noise='gaussian',
    #    noise_coeff=0.3, dag_type='erdos', expected_degree=exp_deg)
    # _, graph = generator.generate()
    # graph = nx.relabel_nodes(graph, mapping={f"V{i}": i for i in range(params["N"])})

    # import causaldag as cd
    # arcs = cd.rand.directed_erdos(params['N'], params['P'])
    # nodes = list(range(params["N"]))
    # graph = nx.DiGraph()
    # graph.add_nodes_from(nodes)
    # _ = [graph.add_edge(n1, n2) for n1 in nodes for n2 in nodes if (n1, n2) in arcs.arcs]

    return g


def gen_graph_structure_bivariate(params: dict) -> nx.DiGraph:
    if params["GS"] == GSType.GRAPH:
        return gen_causal_graph(params)
    elif params["GS"].is_bivariate():
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1])
        if params["GS"].is_causal():
            graph.add_edge(0, 1)
        elif params["GS"].is_anticausal():
            graph.add_edge(1, 0)

    elif params["GS"].is_mvariate():
        graph = nx.DiGraph()
        n_nodes = np.random.randint(3, params["N"])
        graph.add_nodes_from(range(n_nodes))
        if params["GS"].is_causal():
            for n in range(n_nodes - 1): graph.add_edge(n, n_nodes - 1)
    else:
        raise ValueError(params["GS"])
    return graph


def gen_graph_structure_or_chain(params: dict) -> nx.DiGraph:
    if params["GS"] == GSType.GRAPH:
        return gen_causal_graph(params)
    elif params["GS"].is_bivariate():
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edge(0, 1)
        if params["GS"].is_causal():
            graph.add_edge(1, 2)
        elif params["GS"].is_anticausal():
            graph.add_edge(2, 1)

    elif params["GS"].is_mvariate():
        graph = nx.DiGraph()
        n_nodes = np.random.randint(3, params["N"])
        graph.add_nodes_from(range(n_nodes))
        graph.add_edge(0, 1)
        if params["GS"].is_causal():
            for n in range(n_nodes - 1): graph.add_edge(n, n_nodes - 1)
    else:
        raise ValueError(params["GS"])
    return graph


def gen_causal_graph(params: dict) -> nx.DiGraph:
    if params['DG'] == DagType.ERDOS:
        return gen_erdos_graph(params)

    elif params['DG'] == DagType.SCALE_FREE:
        G = nx.directed.scale_free_graph(
            params["N"],
            alpha=0.41, beta=0.54, gamma=0.05,
            delta_in=0.2, delta_out=0)
        G = G.to_directed()
        _G = nx.DiGraph()
        for u, v, _ in G.edges:
            if (u, v) not in _G.edges:
                _G.add_edge(u, v)
        try:
            while True:
                cycle = nx.find_cycle(_G)
                e = cycle.pop()
                _G.remove_edge(*e)
        except nx.NetworkXNoCycle:
            pass
        graph = _G
    elif params['DG'] == DagType.RANDOM:
        n_nodes = params['N']
        avg_edges = n_nodes // 3
        n_edges = (n_nodes * avg_edges) // 2
        nodes = list(range(n_nodes))
        random.shuffle(nodes)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        while len(graph.edges) < n_edges:
            node1, node2 = random.sample(nodes, 2)
            if not nx.has_path(graph, node2, node1):
                graph.add_edge(node1, node2)
    else:
        raise ValueError(f"{params['DG']}")
    return graph


def gen_t_n_Z_graph_or_bivariate(structure, iv_mode, params, graph, random_state, seed):
    if structure == GSType.GRAPH:
        pass
    else:
        params["N"] = len(graph.nodes)

    if structure == GSType.GRAPH:
        nodesets_affected_byeach_Z =  gen_random_intervention_targets(params, graph, random_state)
            #gen_atomic_intervention_targets(params, graph, random_state) if iv_mode == IvMode.ATOMIC else \


        t_n_Z = [set([i for i in range(params["N"]) if i in nodesets_affected_byeach_Z[z]])
                 for z in range(len(nodesets_affected_byeach_Z))]

    elif structure.only_Y_changes():
        t_n_Z = [{max(graph.nodes)}]
        params["NZ"] = 1
    elif structure.only_X_changes():
        t_n_Z = [{i for i in graph.nodes if i != max(graph.nodes)}]
        params["NZ"] = 1
    elif structure.both_change():
        t_n_Z = [set(graph.nodes)]
        params["NZ"] = 1
    elif structure.indep_change():
        t_n_Z = [{i for i in graph.nodes if i != max(graph.nodes)}, {max(graph.nodes)}]
        params["NZ"] = 2
    else:
        assert structure.neither_change()
        t_n_Z = []
        params["NZ"] = 0
    return params, t_n_Z


def gen_t_n_Z_graph_or_chains(structure, iv_mode, params, graph, random_state, seed):
    if structure == GSType.GRAPH:
        pass
    else:
        params["N"] = len(graph.nodes)

    if structure == GSType.GRAPH:
        nodesets_affected_byeach_Z = gen_random_intervention_targets(params, graph, random_state)
           # gen_atomic_intervention_targets(params, graph, random_state) if iv_mode == IvMode.ATOMIC else \

        t_n_Z = [set([i for i in range(params["N"]) if i in nodesets_affected_byeach_Z[z]])
                 for z in range(len(nodesets_affected_byeach_Z))]
    elif structure.only_Y_changes():
        t_n_Z = [{max(graph.nodes)}]
        params["NZ"] = 1
    elif structure.only_X_changes():
        t_n_Z = [{i for i in graph.nodes if i != max(graph.nodes) and i != 0}]  # only middle node?
        params["NZ"] = 1
    elif structure.both_change():
        t_n_Z = [set(graph.nodes)]
        params["NZ"] = 1
    elif structure.indep_change():
        t_n_Z = [{i for i in graph.nodes if i != max(graph.nodes) and i != 0}, {max(graph.nodes)}]
        params["NZ"] = 2
    else:
        assert structure.neither_change()
        t_n_Z = []
        params["NZ"] = 0

    return params, t_n_Z

def gen_synthetic_data(iv_mode, params, graph, random_state, seed, vb, lg=None):
    if iv_mode == GenDataType.IID:
        return gen_synthetic_data_iid(iv_mode, params, graph, random_state, seed, vb, lg)
    elif iv_mode==GenDataType.MULTI_CONTEXT:
        return gen_synthetic_data_context(iv_mode, params, graph, random_state, seed, vb, lg)
    if iv_mode==GenDataType.MIXING:
        return gen_synthetic_data_mixing(iv_mode, params, graph, random_state, seed, vb, lg)
    else:
        raise NotImplementedError("not implemented")


def gen_synthetic_data_context(iv_mode, params, graph, random_state=None, seed=0, vb=0, lg=None):
    #defaults
    if 'NS' not in params: params['NS'] = NoiseType.MIX
    if 'F' not in params: params['F'] = FunType.MIX
    if 'IVT' not in params: params['IVT'] = IvType.COEF
    # (iv_mode, random_state, lg are kept for parity/hooks; not used here directly)
    dg = DataGenContext(params, graph=graph, seed=seed, vb=vb)
    X, contexts = dg.gen_X()

    truths = dict(
        true_g=graph,
        context_labels=contexts,  # length S, ints in [0..C-1]
        is_true_edge=lambda node: (lambda other: 'causal' if graph.has_edge(node, other)
                                   else ('anticausal' if graph.has_edge(other, node) else 'spurious')),
        mechanism_partitions=dg.partitions,   # per-node: context -> group id
        mechanism_meta=dg.mech,               # per-node, per-group specs (weights, nonlins, sigma)
        _dg=dg
    )
    labels = np.unique(contexts)
    Xs_by_c  = [X[contexts == c] for c in labels]  # list aligned with `labels`
    # optional: dict form
    Xs_by_c_dict = {int(c): X[contexts == c] for c in labels}
    return Xs_by_c_dict, truths, params


def gen_synthetic_data_iid(iv_mode, params, graph, random_state, seed, vb, lg=None):

    dg = DataGen(params, graph=graph, seed=seed, vb=vb - 1)
    X = dg.gen_unconfounded_X()

    truths = dict(
        true_g=graph,
        is_true_edge=lambda node: lambda other: 'causal' if graph.has_edge(node, other) else (
            'anticausal' if graph.has_edge(other, node) else 'spurious'),
        _dg=dg
    )
    return X, truths, params


def gen_synthetic_data_mixing(iv_mode, params, graph, random_state, seed, vb, lg=None):
    _max_n_nodes = params["N"]
    structure = params["GS"]

    # sample a set of nodes affected by mechanism shifts, dep on the considered graph structure
    # reset parameters based on graph structure if neccessary
    params, t_n_Z = gen_t_n_Z_graph_or_chains(structure, iv_mode, params, graph, random_state, seed)

    # sample the class-label confounders
    t_Z = [np.random.choice(params["C"], size=params["S"]) for _ in range(params["NZ"])]
    t_Z = [[i + 1 if i + 1 < params["K"] else 0 for i in Z] for Z in t_Z]

    # remove confds that do not have any nodes due to the data generation
    # -> warning if NZ reduced?
    t_n_Z = [nodeset for nodeset in t_n_Z if len(nodeset) > 0]
    t_Z = [confd for (nodeset, confd) in zip(t_n_Z, t_Z) if len(nodeset) > 0]
    # -> per Z, k subsamples of expected size N/C are affected by mechanism shifts

    dg = DataGen(params, graph=graph, seed=seed, vb=vb - 1)
    X = dg.gen_X(t_n_Z, t_Z)

    if vb > 0:
        samp = params['S']
        for zi, Z1 in enumerate(t_Z): lg.info(
            f"*GEN: Z_{zi}: {len(np.unique(Z1))} cls, {[f'{len(np.where(np.array(Z1) == k)[0])}/{samp}' for k in np.unique(Z1)]} samples")
        # for (z1, z2) in itertools.combinations(t_Z, 2):  lg.info(f"*GEN: ami(Za,Zb) {adjusted_mutual_info_score(z1, z2):.2f}")

    truths = dict(
        true_g=graph,
        t_A=nx.to_numpy_array(graph),
        t_n_Z=dg.conf_ind_sets,
        t_Z=dg.Zs,
        is_true_edge=lambda node: lambda other: 'causal' if graph.has_edge(node, other) else (
            'anticausal' if graph.has_edge(other, node) else 'spurious'),
        _dg=dg
    )

    check_synthetic_data(graph, dg, X, truths, params["GS"], params["N"], params["S"], params["NZ"], _max_n_nodes)

    if vb > 0: lg.info("*GEN: Confounders: " + "/" if len(t_n_Z) == 0 else \
                           ", ".join([f"Z_{zi}: targets  {nodeset}" for zi, nodeset in enumerate(t_n_Z)]))
    if vb > 0: lg.info(
        f"*GEN: Graph structure {params['GS']}\t{len(graph.nodes)} Nodes\t{len(graph.edges)} Edges\tConfd {truths['t_n_Z']}")
    return X, truths, params


def check_synthetic_data(graph, dg, X, truths, graph_structure: GSType, n_nodes: int, n_samples: int,
                         n_confounders: int, max_n_nodes: int):
    assert truths["t_A"].shape == (n_nodes, n_nodes)
    assert len(truths["t_n_Z"]) == len(truths["t_Z"])
    assert X.shape == (n_samples, n_nodes)
    assert truths["t_A"].sum() == len(graph.edges)
    if graph_structure.is_bivariate():
        assert n_nodes == 3
    elif graph_structure.is_mvariate():
        assert 3 <= n_nodes <= max_n_nodes
    else:
        assert n_nodes == max_n_nodes

    if graph_structure.is_causal():
        assert all([(pre, n_nodes - 1) in graph.edges for pre in range(1, n_nodes - 1)])
        if graph_structure.is_bivariate(): assert (1, 2) in graph.edges and truths["t_A"][
            1, 2] == 1  # and len(graph.edges) == 1
    elif graph_structure.is_anticausal() and graph_structure.is_bivariate():
        if graph_structure.is_bivariate(): assert (2, 1) in graph.edges and truths["t_A"][
            2, 1] == 1  # and len(graph.edges) == 1


def gen_interventional_mixture(params):
    from src.baselines.mixture_mec.scm_module import RandomSCMGenerator
    assert params["PZ"] == 1 or params["PZ"] == 0.5, "either half or all nodes should be intervened in this exp."

    sample_size = params["S"]
    n_nodes = params["N"]
    args = dict(
        num_nodes=n_nodes,
        num_nodesp=n_nodes,
        num_tgt_prior=n_nodes + 1,
        obs_noise_mean=0.0,
        obs_noise_var=1.0,
        obs_noise_gamma_shape=None,
        noise_type="gaussian",  # "gaussian", or gamma
        max_edge_strength=1.0,
        graph_sparsity_method="adj_dense_prop",  # [adj_dense_prop, use num_parents]
        adj_dense_prop=params["P"],  # [0.1,0.4,0.6,1.0],
        num_parents=None,
        new_noise_mean=1.0,
        mix_samples=sample_size,
        stage2_samples=sample_size,
        gmm_tol=1e-3,
        intv_type="do",  # hard,do,soft
        new_noise_var=None,
        dtype="simulation",
        cutoff_drop_ratio=0.07
    )
    if params["PZ"] == 1:  # config_dict["intv_targets"] == "all":
        args["intv_targets"] = list(range(n_nodes))
    elif params["PZ"] == 0.5:  # config_dict["intv_targets"] == "half":
        nodes_list = list(range(n_nodes))
        np.random.shuffle(nodes_list)
        half_node_list = nodes_list[0:n_nodes // 2]
        args["intv_targets"] = half_node_list
    # or a given nb of interventions:
    # elif type(config_dict["intv_targets"]) == type(1):
    #    nodes_list = list(range(n_nodes))
    #    np.random.shuffle(nodes_list)
    #    some_node_list = nodes_list[0:config_dict["intv_targets"]]
    #    args["intv_targets"] = some_node_list
    else:
        raise ValueError(params["PZ"])

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
    print("Generated mixture samples")
    intv_args_dict, mixture_samples = gSCM.generate_gaussian_mixture(
        args["intv_type"],
        args["intv_targets"],
        args["new_noise_mean"],
        args["new_noise_var"],
        args["mix_samples"],
        args["noise_type"],
        args["obs_noise_gamma_shape"],
    )
    # if args["dtype"] == "simulation": #the above
    # elif  args["dtype"] == "sachs":
    #    print("Generating the mixture sample")
    #    intv_args_dict, mixture_samples, num_nodes = generate_mixture_sachs(
    #        args["dataset_path"],
    #        args["mix_samples"],
    #    )
    # else: raise ValueError(params["dtype"])
    Z1 = []
    for tgt_i, (tgt) in enumerate(intv_args_dict):
        Z1.append(tgt_i * np.ones(intv_args_dict[tgt]["samples"].shape[0]))
    Z1 = np.hstack(Z1)
    Z1 = np.pad(Z1, (0, max(0, params["S"] - len(Z1))), 'maximum')
    truths = dict(
        true_g=nx.from_numpy_array(gSCM.A.T, create_using=nx.DiGraph),
        t_A=gSCM.A.T,
        t_Z=[Z1],
        t_n_Z=[args["intv_targets"]],
        intv_args_dict=intv_args_dict,
        args=args
    )

    return mixture_samples, truths
