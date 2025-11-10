from __future__ import annotations
from typing import Optional, Any, Callable

import itertools
import networkx as nx
import numpy as np
import sklearn

from src.causalchange.dag.edge_memoized import EdgeMemoized
from src.causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.scoring.scoring_context import ScoreType, GPType, DataMode, GraphSearch
from src.causalchange.search.dag_search import dag_tree_search
from src.causalchange.util.utils import is_insignificant
from src.causalchange.util.utils_idl import exp_mutual_info_score, pi_xor_pessimistic, \
    expected_sampled_mutual_info_score, expected_sampled_adjusted_mutual_info_score, expected_smi, \
    soft_mutual_information, get_true_idl_Z
from src.causalchange.util.old.visu import visu_pair_mi, visu_node_pproba, visu_node_idls, \
    visu_node_idl, visu_node_true_idl, visu_pproba_dens

from numpy.typing import NDArray

XArray = NDArray[np.number]
XType = XArray | dict[int, XArray]
#ScoreRes = dict[str, Any]


class CausalChange:
    #external
    # data
    X: XType
    D: int
    N: int
    # algo and score
    data_mode: DataMode
    graph_search: GraphSearch
    score_type: ScoreType | GPType
    mixing_type: MixingType
    #hyperparams and kwargs
    k_max: int
    mi_thresh: float
    smi_thresh: float
    lambda_mix: float
    bic_thresh: float
    extra_refinement: bool
    oracle_Z: bool
    oracle_K: bool
    oracle_G: bool
    oracle_order: bool
    lg: Optional[Any]
    vb: int
    truths: dict[str, Any]
    true_graph: nx.DiGraph | None
    true_top_order: list[int] | None

    #internal
    _add_greedily: bool
    _info: Callable[[str, int], None]
    is_true_edge: Callable[[int], Callable[[int], str]]
    # graph state
    graph_state: nx.DiGraph
    edges_state: EdgeMemoized
    candidates: list[int]
    topological_order: list[int]
    # mix state
    e_Z_n: dict[int, list[int]]
    e_Zp_n: dict[int, XArray]
    e_Z: list[XArray]
    e_n_Z: list[set[int]]
    Z_pairs: list[tuple[int, int]]
    Z_pairs_scores: dict[str, dict[str, float]]
    confd_A: XArray
    confd_targets_alt: list[set[int]]
    # flags
    fitted_graph: bool
    fitted_mixing: bool

    def __init__(self, **kwargs):
        r""" CausalChange: Causal Discovery Algorithms under Distribution Change (iid version using TOPIC, multi-context data, time series, mixtures of causal mechanisms)
        :param optargs: optional arguments

        :Keyword Arguments:
        * *data_mode* (``DataMode``) -- input data type, one iid dataset, multi-context data, mixed data, or TS data
        * *score_type* (``MixingType``) -- regressor
        * *mixing_type* (``MixingType``) -- for mixed data, type of mixture model inference (EM algo), ow skip
        * *truths* (``nx.DiGraph``) -- for mixed data, oracle versions, w entries 't_A', 't_Z', 't_n_Z'
        * *oracle_G* (``bool``) -- known graph
        * *oracle_order* (``bool``) -- known top order
        * *oracle_K* (``bool``) -- known n mixtures
        * *oracle_Z* (``bool``) -- known assignments per node (debug)
        * *k_max* (``int``) -- max n mixtures
        * *alpha_gp_mdl* (``float``) -- sigificance thresh GP-MDL score
        * *mi_thresh* (``float``) -- sigificance thresh MI pairs of nodes
        * *smi_thresh* (``float``) -- sigificance thresh soft MI pairs of nodes
        * *bic_thresh* (``float``) -- sigificance thresh edge score
        * *lambda_mix* (``float``) -- regularization param for mixing penalty in score
        * *use_smi* (``float``) -- use soft MI instead of MI to detect confounding between node pair
        * *extra_refinement* (``float``) -- slower parent subset search at the end
        * *lg* (``logging``) -- logger if verbosity>0
        * *vb* (``int``) -- verbosity level
        """
        self.defaultargs = {
            "data_mode": DataMode.CONTEXTS,
            "graph_search": GraphSearch.TOPIC,
            "score_type": GPType.EXACT,
            "mixing_type": MixingType.SKIP,
            "truths": dict(),
            "oracle_G": False,
            "oracle_order": False,
            "oracle_K": False,
            "oracle_Z": False,
            "k_max": 5,
            "alpha_gp_mdl": 0.05,
            "mi_thresh": 0.03,
            "smi_thresh": 0.01,
            "bic_thresh": 0,
            "lambda_mix": 1,
            "use_smi": False,
            "extra_refinement": True,
            "lg": None,
            "vb": 0}

        self.__dict__.update((k, v) for k, v in self.defaultargs.items() if k not in kwargs.keys())
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.defaultargs.keys())

        assert self.mixing_type != MixingType.SKIP if self.data_mode == DataMode.MIXED else self.mixing_type == MixingType.SKIP
        assert not self.oracle_order or 'true_g' in self.truths

        def _info(st, strength=0):
            (self.lg.info(st) if self.lg is not None else print(st)) if self.vb + strength > 0 else None
        self._info = _info
        self.is_true_edge = (lambda i: lambda j: "") if 'true_g' not in self.truths else \
            (lambda node: lambda other: 'causal' if self.truths['true_g'].has_edge(node, other) else (
                'rev' if self.truths['true_g'].has_edge(other, node) else 'spurious'))

        if self.oracle_G or self.oracle_order: assert 'true_g' in self.truths, 'need truths if oracle_G'
        self.graph_state =  nx.DiGraph() if not self.oracle_G else self.truths['true_g']
        if self.oracle_Z: self.use_smi = False
        self.topological_order = [] if not self.oracle_order else list(nx.topological_sort(self.truths['true_g']))
        self.true_top_order = [] if 'true_g' not in self.truths else list(
            nx.topological_sort(self.truths['true_g']))
        self._add_greedily = False

        self.fitted_graph, self.fitted_mixing = False, False

    def initialize(self):
        assert self.X is not None
        print(self.data_mode, self.score_type, self.mixing_type)
        self.edges_state = EdgeMemoized(self.X, self.data_mode, self.score_type, self.mixing_type,
                                        **self.get_scoring_params())

    def init_and_check_X(self, X: XType):
        self.X = X
        if isinstance(X, dict):
            first = next(iter(X.values()))
            self.D, self.N = first.shape
        else:
            self.D, self.N = X.shape

        import warnings
        if not (0 < self.N < self.D):
            warnings.warn("n. samples < n. nodes", RuntimeWarning)
        assert self.D > 0 and self.N > 0


    #%% Graph search
    def fit(self, X: XType) -> nx.DiGraph:
        self.init_and_check_X(X)
        self.initialize()
        if self.data_mode == DataMode.IID:
            return self.fit_graph_iid()
        elif self.data_mode == DataMode.CONTEXTS:
            return self.fit_graph_over_contexts()
        elif self.data_mode == DataMode.TIME:
            return self.fit_graph_over_time()
        elif self.data_mode == DataMode.TIME_CONTEXTS:
            return self.fit_graph_over_time_and_contexts()
        elif self.data_mode == DataMode.CONFOUNDED:
            return self.fit_graph_and_confounders()
        elif self.data_mode == DataMode.MIXED:
            return self.fit_graph_and_mixtures()
        else: raise ValueError(self.data_mode)

    #%% Single continuous-valued dataset
    def fit_graph_iid(self) -> nx.DiGraph:
        assert self.data_mode == DataMode.IID
        self._info(f"\t*** CausalChange ({self.score_type.value}) ***")
        self._fit_graph()
        self.fitted_graph = True
        return self.graph_state

    #%% Time series
    def fit_graph_over_time(self) -> nx.DiGraph:
        assert self.data_mode == DataMode.TIME
        self._info(f"\t*** CausalChange, time series setting ({self.score_type.value}) ***")
        self._fit_graph()
        self.fitted_graph = True
        return self.graph_state

    #%% Multi-context datasets
    def fit_graph_over_contexts(self) -> nx.DiGraph:
        assert self.data_mode == DataMode.CONTEXTS
        self._info(f"\t*** CausalChange, multi-context setting ({self.score_type.value}) ***")
        self._fit_graph()
        self.fitted_graph = True
        self._fit_changes()
        return self.graph_state

    def _fit_changes(self) -> None:
        assert self.fitted_graph
        changes = {}
        _res_each_node = {}
        for node_i in self.graph_state.nodes:
            parents_i = list(self.graph_state.predecessors(node_i))
            score, res = self._score(parents_i, node_i, ret_full_result=True)
            changes[node_i] = {"groups": res["groups"], "partition":  res["partition"]}
            _res_each_node[node_i] = res
        self.changes = changes
        self._res_each_node = _res_each_node #for debugging


    #%% Mixture of causal mechanisms
    def fit_graph_and_mixtures(self) -> nx.DiGraph:
        assert self.data_mode == DataMode.MIXED
        self._info(f"\t*** CausalMixtures ({self.mixing_type.value}) ***")


        self._fit_graph()
        self._fit_Z_nodes()
        self._fit_Z_sets()

        self.fitted_graph, self.fitted_mixing = True, True
        return self.graph_state


    #%% CoCo with TOPIC
    def fit_graph_and_confounders(self) -> nx.DiGraph:
        raise NotImplementedError("integrate Coco here")

    #%% SPACETIME with TOPIC
    def fit_graph_over_time_and_contexts(self) -> nx.DiGraph:
        raise NotImplementedError("integrate Stime here")


    #%% Graph search algorithms
    def _fit_graph(self) -> nx.DiGraph:
        if self.oracle_G: return self.graph_state
        if self.graph_search == GraphSearch.TOPIC:
            return self._fit_graph_topological()
        elif self.graph_search == GraphSearch.GLOBE:
            return self._fit_graph_edge_greedy()
        else: raise ValueError(self.graph_search)


    def _fit_graph_topological(self) -> nx.DiGraph:
        """ TOPIC (Xu et al., 2025) """
        #self.edges_state = EdgeMemoized(self.X, self.data_mode, self.score_type, self.mixing_type)

        self.graph_state.add_nodes_from(range(self.N))
        self.candidates = list(range(self.N))
        self.order_nodes()

        self.fitted_graph = True
        return self.graph_state

    def _fit_graph_edge_greedy(self) -> nx.DiGraph:
        """ GLOBE (Mian et al., 2021) """
        #self.edges_state = EdgeMemoized(self.X, self.data_mode, self.score_type, self.mixing_type)
        self.graph_state.add_nodes_from(range(self.N))

        dag_model = dag_tree_search(self.X, self.N, self.edges_state)
        self.graph_state = nx.from_numpy_array(dag_model.get_adj(), create_using=nx.DiGraph)

        self.fitted_graph = True
        return self.graph_state

    #%% Mixing Variable Search
    def fit_Z_given_G(self, graph_adj: XArray, skip_pruning=False, skip_sets=False) -> None:
        graph = nx.from_numpy_array(graph_adj, create_using=nx.DiGraph)
        self._info(f"\t*** Causal Mixture Modeling (given DIgraph w {len(graph.edges())} edges) ***")

        self.graph_state = graph.copy()
        self.initialize()
        #self.edges_state = EdgeMemoized(self.X, self.data_mode, self.score_type, self.mixing_type)
        self.fitted_mixing = True

        self._fit_Z_nodes()
        if not skip_sets: self._fit_Z_sets()
        if skip_pruning: return

        #self.edges_state = EdgeMemoized(self.X, self.data_mode, self.score_type, self.mixing_type)
        self.initialize()
        self.prune_spurious_edges(self.graph_state.copy())
        self._fit_Z_nodes()
        if not skip_sets: self._fit_Z_sets()

    def _fit_Z_nodes(self) -> None:
        X = self.X
        self.e_Z_n = {}
        self.e_Zp_n = {}
        for nodei in self.graph_state.nodes:
            cov = [i for i in self.graph_state.predecessors(nodei)] if nx.is_directed_acyclic_graph(self.graph_state)  else [i for i in self.graph_state.neighbors(nodei)]
            score, res =  self._score(cov, nodei, ret_full_result=True)
            self.e_Z_n[nodei], self.e_Zp_n[nodei] =  res.get("idl", []),res.get("pproba", [])
            if all([ky in self.truths for ky in ['t_A', 't_Z', 't_n_Z']]):
                true_idl = get_true_idl_Z(np.where(self.truths['t_A'][:, nodei] != 0)[0], nodei, self.truths['t_A'],
                                           self.truths['t_Z'], self.truths['t_n_Z'], X.shape[0])
                ami = sklearn.metrics.adjusted_mutual_info_score(true_idl, self.e_Z_n[nodei])
                self._info(
                    f"\t\tNode {nodei} | {cov}: {ami:.2f} k^={len(np.unique(self.e_Z_n[nodei]))} k*={len(np.unique(true_idl))}",
                    -1)  # -1
                if self.vb >= 2: self.visu_scatter_mixing_assignment_node(nodei)

    def _fit_Z_sets(self, n_nodes=None) -> None:
        if n_nodes is None: n_nodes = len(self.graph_state.nodes)
        # adj_A = nx.to_numpy_array(self.topic_graph)
        self.Z_pairs = []
        self.Z_pairs_scores = {}
        self.confd_A = np.zeros((n_nodes, n_nodes))
        alt_A = np.zeros((n_nodes, n_nodes))

        # Jointly mixed nodes (confounding)
        for (i, j) in itertools.combinations(set(range(n_nodes)), 2):
            Za, Zb = self.e_Z_n[i], self.e_Z_n[j]
            pa, pb = self.e_Zp_n[i], self.e_Zp_n[j]

            mi = sklearn.metrics.mutual_info_score(Za, Zb)
            ami = sklearn.metrics.adjusted_mutual_info_score(Za, Zb)
            smi = soft_mutual_information(pa, pb) if not self.oracle_Z else mi
            # asmi = soft_adjusted_mutual_information(pa, pb)
            emi, _ = exp_mutual_info_score(Za, Zb)
            _ = expected_sampled_mutual_info_score(Za, Zb)
            _ = expected_sampled_adjusted_mutual_info_score(Za, Zb)
            esmi, _ = expected_smi(pa, pb, len(Za)) if not self.oracle_Z else (emi, 0)

            mi_val = 0 if len(np.unique(Za)) == 1 or len(np.unique(Zb)) == 1 else \
                mi if mi > emi and mi > self.mi_thresh else 0
            smi_val = 0 if len(np.unique(Za)) == 1 or len(np.unique(Zb)) == 1 else \
                smi if smi > esmi and smi > self.smi_thresh else 0

            val = smi_val if self.use_smi else mi_val
            self.confd_A[i, j] = self.confd_A[j, i] = val
            alt_A[i][j] = mi_val if self.use_smi else smi_val
            is_confounded = self.confd_A[i][j] > 0
            if is_confounded: self.Z_pairs.append((i, j))
            self.Z_pairs_scores[f"{i}-{j}"] = {"ami": ami, "mi": mi, "smi": smi, "emi": emi, "esmi": esmi}

        # Dependently mixed nodes (confounding)
        componentG = nx.from_numpy_array(self.confd_A, create_using=nx.Graph)
        confd_targets = [set(n_set) for n_set in nx.connected_components(componentG) if len(n_set) > 1]

        self.confd_targets_alt = [set(n_set) for n_set in
                                  nx.connected_components(nx.from_numpy_array(alt_A, create_using=nx.Graph)) if
                                  len(n_set) > 1]

        # Independently mixed nodes (independent change)
        for i in list(range(n_nodes)):
            if not any([i in n_set for n_set in confd_targets]) and len(np.unique(self.e_Z_n[i])) > 1:
                confd_targets.append({i})

            if not any([i in n_set for n_set in self.confd_targets_alt]) and len(np.unique(self.e_Z_n[i])) > 1:
                self.confd_targets_alt.append({i})

        # Simple aggregation of confounder labels per node set
        confd_idls = []
        for n_set in confd_targets:
            confd_avg = np.zeros(self.X.shape[0])
            for node_i in n_set:
                confd_idl = self.e_Z_n[node_i]  # todoself.get_edge_confd_A(node_i, adj_A)
                confd_avg = pi_xor_pessimistic(confd_idl, confd_avg)
            confd_idls.append(confd_avg)

        # per node set
        self.e_Z = confd_idls
        self.e_n_Z = confd_targets
        assert len(self.e_n_Z) == len(self.e_Z)

        # per node
        assert len(self.e_Z_n) == len(self.e_Zp_n)
        self._info(f"\t** Confd node sets: {self.e_n_Z}")

    # %% GRAPH PRUNING
    def local_pruning(self, node, parents) -> list:
        parent_subset = []
        for parent in parents:
            # assert (parent, node) in self.topic_graph.edges, f"edge {parent}->{node} not in graph"
            if (parent, node) in self.graph_state.edges:
                self.graph_state.remove_edge(parent, node)
        for parent in parents:  # try score-pruning under the node's true partition
            gain = self._addition_gain(node, parent)  # during scoring&fitting: values of Z will be used
            if self._significant(gain) or self._add_greedily:
                self._add_edge(parent, node, gain=float(gain))
                parent_subset.append(parent)
        return parent_subset

    def prune_spurious_edges(self, graph) -> None:
        self.edges_state.hybrid = True  # enforce that we score each domain separately here
        self._info(f"\t** Edge Pruning ")
        for _, node in enumerate(graph.nodes()):
            pre = graph.predecessors(node) if nx.is_directed_acyclic_graph(graph) else graph.neighbors(node)
            parents_spurious = [nodem for nodem in pre]
            #parent_subset = self.local_pruning(nodei, parents_spurious)
            parent_subset = self.local_pruning(node, parents_spurious)

            for nodej in parents_spurious:
                pair_confounded = ((nodej, node) in self.Z_pairs or (node, nodej) in self.Z_pairs)
                prune = nodej not in parent_subset
                if 'true_g' in self.truths:
                    correct = "(keeping true edge correctly)" if not prune and (nodej, node) in self.truths[
                        'true_g'].edges else \
                        "(pruning ncaus. edge correctly)" if prune and (nodej, node) not in self.truths[
                            'true_g'].edges else \
                            "(pruning true edge erroneously)" if prune and (nodej, node) in self.truths[
                                'true_g'].edges else "(keeping ncaus. edge erroneously)"
                else:
                    correct = ""
                self._info(
                    f"\t\tPair {nodej}->{node}: {'cfd' if pair_confounded else '1. uncfd'}  {'PRUNED' if prune else 'keep'} {correct}",
                    -2)

    # %% GRAPH SEARCH: TOPOLOGICAL ORDERING
    def order_nodes(self):
        it = 0
        while it < self.N:
            source = self.get_next_node(self.candidates if not self.oracle_order else self.true_top_order[it])
            self.candidates.remove(source)
            self.topological_order.append(source)
            it += 1
            self._info(f"\t{it}. Source: {source}\t current {self.topological_order}, true {self.true_top_order}", -2)

            self.add_edges(source)
            self.refine_edges(source)

        if self.extra_refinement:
            self.refinement_phase()

    def add_edges(self, source):
        for node in self.candidates:
            if node in self.topological_order or node == source or self.has_cycle(source, node):
                continue
            gain = self._addition_gain(node, source)
            if self._significant(gain) or self._add_greedily:
                self._add_edge(source, node, gain=float(gain))

    def _addition_gain(self, node, source):
        parents = list(self.graph_state.predecessors(node)).copy() if nx.is_directed_acyclic_graph(self.graph_state) else  list(self.graph_state.neighbors(node)).copy()
        old_score = self._score(parents, node)
        parents.append(source)
        new_score = self._score(parents, node)
        gain = self._gain(new_score, old_score)
        return gain

    def refine_edges(self, source):
        parents = list(self.graph_state.predecessors(source))
        n_removed = 0
        while n_removed < len(parents):
            removed_found, removed_parent = self.refine_step(source, parents)

            if removed_parent is not None:
                self._remove_edge(removed_parent, source)
                parents.remove(removed_parent)
                n_removed += 1
            else:
                break

    def refine_step(self, source, parents):
        removed_found, best_parent, best_diff = False, None, -np.inf
        old_score = self._score(parents, source)

        for parent in parents:
            new_parents = parents.copy()
            new_parents.remove(parent)
            if len(new_parents) == 0: continue
            new_score = self._score(new_parents, source)
            diff = old_score - new_score  # >0 means removing parent improved the score

            if diff > best_diff and is_insignificant(abs(diff)):#self._significant(diff):
                best_diff = diff
                best_parent = parent
                removed_found = True
        return removed_found, best_parent



    # %% GRAPH SEARCH: EDGE-GREEDY


    # %% GRAPH AND SCORING UTILS
    def _add_edge(self, parent, child, vb=-2, gain=None):
        self.graph_state.add_edge(parent, child)
        self._info(
            f"\tAdding {self.is_true_edge(parent)(child)} edge {parent} -> {child} {'' if gain is None else f': gain {gain:.2f}'}",
            vb)

    def _info_graph(self, vb=-1):
        if 'true_g' in self.truths:
            self._info(
                #f"\tResult: {', '.join([f'{ky}:{val:.2f}' for ky, val in self.get_metrics_graph(self.truths['true_g']).items()])}"
                f"\nEdges:")
        for (parent, child) in self.graph_state.edges:
            self._info(f"\t\t{self.is_true_edge(parent)(child)} edge {parent} -> {child}", vb)

    def _remove_edge(self, parent, child, vb=-2):
        self.graph_state.remove_edge(parent, child)
        self._info(f"\tRemoving {self.is_true_edge(parent)(child)} edge {parent} -> {child}", vb)

    def _score(self, parents, child, ret_full_result=False, vb=-3):
        score, res = self.edges_state.score_edge(child, parents) #, **self.get_scoring_params())
        if len(parents):
            self._info(
                f"\tScoring {'&'.join([self.is_true_edge(parent)(child) for parent in parents])} edge {parents} -> {child}\t{score:.2f}",
                vb)

        if ret_full_result: return score, res
        return score

    @staticmethod
    def _gain(new_score, old_score):
        return old_score - new_score

    @staticmethod
    def _improvement(new_score, old_score):
        return new_score - old_score

    def _significant(self, gain):
        return gain > self.bic_thresh

    def has_cycle(self, source, node):
        G_hat = self.graph_state.copy()
        G_hat.add_edge(source, node)
        try:
            _ = nx.find_cycle(G_hat, orientation="original")
        except nx.exception.NetworkXNoCycle:
            return False
        return True

    def get_next_node(self, candidates):
        if self.oracle_order:
            n = len(self.topological_order)
            self._info(f"\tTrue Next Node: {self.true_top_order[n]}", -2)
            return self.true_top_order[n]

        improvement = self.get_improvement_matrix(self.graph_state, candidates)
        delta = improvement - improvement.T
        # find the node with the smallest possible delta
        np.fill_diagonal(delta, -np.inf)
        best_delta = np.max(delta, axis=1)
        worst = np.argmin(best_delta)

        self._info(f"\tNext Node: {candidates[worst]}, order {self.topological_order} ", -2)
        k = len(improvement)
        top_k_ind = np.argsort(-best_delta)[-k:][::-1]
        self._info(f"\tbest {k} next nodes:", -3)
        for i in top_k_ind: self._info(f"\t  node  {candidates[i]}, best_delta: {-best_delta[i]:.4f}", -3)

        return candidates[worst]

    def get_improvement_matrix(self, graph, candidates):
        improvement_matrix = np.zeros((len(candidates), len(candidates)))
        for cause in candidates:
            for effect in candidates:
                if cause == effect:
                    continue
                parents = list(graph.predecessors(effect))
                old_score = self._score(
                    parents, effect)
                parents.append(cause)
                new_score = self._score(parents, effect)
                improvement_matrix[candidates.index(cause), candidates.index(effect)] = \
                    self._improvement(new_score, old_score)
        return improvement_matrix

    def refinement_phase(self, min_parent_set_size=0):
        # smallest subset of parents with insignificant score gain
        for j in self.graph_state.nodes:
            parents = list(self.graph_state.predecessors(j))
            if len(parents) == 0:
                continue

            best_size = np.inf
            arg_max = None

            old_score = self._score(parents, j)
            old_parents = parents.copy()

            for k in range(min_parent_set_size, len(parents) + 1 - 1):
                parent_sets = itertools.combinations(parents, k)
                for parent_set in parent_sets:

                    new_score = self._score(parent_set, j)
                    gain = self._gain(new_score, old_score)

                    if is_insignificant(np.abs(gain)) and len(parent_set) < best_size:  # favor smaller parent sets
                        best_size = len(parent_set)
                        arg_max = parent_set

            if arg_max is None:
                continue
            self._info(f'\trefine {parents} to {arg_max} -> {j}', -2)
            for p in old_parents:
                if p not in arg_max:
                    self._remove_edge(p, j)
    def get_results_graph(self):
        for j in self.graph_state.nodes:
            self._score(list(self.graph_state.predecessors(j)), j, True)



    # %% VISU
    def visu_pproba_dens(self, nodei):
        if len(np.unique(self.e_Z_n[nodei])) > 1: visu_pproba_dens(self.e_Zp_n[nodei])

    def visu_heatmatrix_nodepair_MI(self, **kwargs):
        visu_pair_mi(self.e_Z_n, self.e_Zp_n, soft=False, **kwargs)
        if not self.oracle_Z: visu_pair_mi(self.e_Z_n, self.e_Zp_n, soft=True, **kwargs)

    def visu_scatter_mixing_assignments(self, **kwargs):
        visu_node_idls(self.truths['true_g'], self.X, self.e_Z_n, **kwargs)

    def visu_scatter_true_assignments(self, nodei, **kwargs):
        assert 'true_g' in self.truths and 't_n_Z' in self.truths and 't_Z' in self.truths
        visu_node_true_idl(self.X, nodei, list(self.truths['true_g'].predecessors(nodei)), self.truths['t_n_Z'],
                           self.truths['t_Z'], method_idf='_trueGtrueZ', **kwargs)

    def visu_scatter_mixing_confidence(self, **kwargs):
        visu_node_pproba(self.truths['true_g'], self.X, self.e_Z_n, self.e_Zp_n, **kwargs)

    def visu_scatter_mixing_assignment_node(self, nodei, **kwargs):
        visu_node_idl(self.X, nodei, list(self.truths['true_g'].predecessors(nodei)), self.e_Z_n[nodei], **kwargs)
        visu_node_true_idl(self.X, nodei, list(self.truths['true_g'].predecessors(nodei)), self.truths['t_n_Z'],
                           self.truths['t_Z'],
                           method_idf='_trueGtrueZ', **kwargs)
    def visu_scatter_mixing_assignment_pair(self, nodei, nodej, **kwargs):
        visu_node_idl(self.X, nodei, [nodej], self.e_Z_n[nodei], **kwargs)

    def visu_scatter_mixing_fit(self, node, parents, k_range, **kwargs):
        from src.causalchange.scoring.fit_cond_mixture import  fit_functional_mixture

        res_dict = fit_functional_mixture(self.X, node, parents, k_range, None, None, lg=None)
        visu_node_idl(self.X, node, parents, res_dict["idl"], method_idf="_givenPAestimZ", **kwargs)

    # %% all scoring hyperparameters to pass down to Memoized edge score
    def get_scoring_params(self):
        return dict(
            oracle_Z=self.oracle_Z,
            oracle_K=self.oracle_K,
            t_A=self.truths.get('t_A', None),
            t_Z=self.truths.get('t_Z', None),
            t_n_Z=self.truths.get('t_n_Z', None),
            k_max=self.k_max,
            lambda_mix=self.lambda_mix,
            lg=self.lg,
        )