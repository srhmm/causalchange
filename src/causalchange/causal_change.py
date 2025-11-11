from __future__ import annotations
from typing import Optional, Any, Callable

import itertools
import networkx as nx
import numpy as np
import sklearn

from src.causalchange.dag.dag import DAG
from src.causalchange.dag.edge_memoized import EdgeMemoized
from src.causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.cc_types import ScoreType, GPType, DataMode, GraphSearch, XArray, XType
from src.causalchange.util.upq import UPQ
from src.causalchange.util.utils import is_insignificant
from src.causalchange.util.utils_idl import exp_mutual_info_score, pi_xor_pessimistic, \
    expected_sampled_mutual_info_score, expected_sampled_adjusted_mutual_info_score, expected_smi, \
    soft_mutual_information, get_true_idl_Z
from src.causalchange.util.old.visu import visu_pair_mi, visu_node_pproba, visu_node_idls, \
    visu_node_idl, visu_node_true_idl, visu_pproba_dens

from numpy.typing import NDArray



class CausalChange:
    #external
    X: XType
    D: int
    N: int
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
        r""" CausalChange: Causal Discovery Algorithms under Distribution Change (continuous data, multi-context continuous data, continuous-valued time series, mixtures of causal mechanisms)
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
            "data_mode": DataMode.IID,
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
    def is_score_insignificant(self, gain):
        if self.score_type.is_scorebased():
            return is_insignificant(gain)
        else: return gain < 0
    def initialize(self):
        assert self.X is not None
        print(self.data_mode, self.score_type, self.mixing_type)
        self.edges_state = EdgeMemoized(
            self.X, self.data_mode, self.score_type, self.mixing_type,**self.get_scoring_params())

    def init_and_check_X(self, X: XType):
        self.X = X
        if isinstance(X, dict):
            assert self.data_mode.is_dict_like(), "expected np array"
            first = next(iter(X.values()))
            self.D, self.N = first.shape
        else:
            assert isinstance(X, np.ndarray), "expected np array"
            assert not self.data_mode.is_dict_like(), "expected dict of np arrays"
            self.D, self.N = X.shape

        import warnings
        if not (0 < self.N < self.D):
            warnings.warn("n. samples < n. nodes", RuntimeWarning)
        assert self.D > 0 and self.N > 0


    #%% Graph search
    def fit(self, X: XType) -> nx.DiGraph:
        """ Discover a causal DAG
       :param X: ``XArray | dict[int, XArray]``: input data
       :return: ``nx.DiGraph``: causal DAG over nodes in X
        """
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
        self._graph_search()
        self.fitted_graph = True
        return self.graph_state

    #%% Time series
    def fit_graph_over_time(self) -> nx.DiGraph:
        assert self.data_mode == DataMode.TIME
        self._info(f"\t*** CausalChange, time series setting ({self.score_type.value}) ***")
        self._graph_search()
        self.fitted_graph = True
        return self.graph_state

    #%% Multi-context datasets
    def fit_graph_over_contexts(self) -> nx.DiGraph:
        assert self.data_mode == DataMode.CONTEXTS
        self._info(f"\t*** CausalChange, multi-context setting ({self.score_type.value}) ***")
        self._graph_search()
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
        """ Discover a causal DAG and discrete latent "mixing" variables
       :return: ``nx.DiGraph``: causal DAG
        """
        assert self.data_mode == DataMode.MIXED
        self._info(f"\t*** CausalMixtures ({self.mixing_type.value}) ***")


        self._graph_search()
        self._fit_latent_discrete_per_node()
        self._fit_latent_discrete_per_set()

        self.fitted_graph, self.fitted_mixing = True, True
        return self.graph_state


    #%% CoCo with TOPIC
    def fit_graph_and_confounders(self) -> nx.DiGraph:
        """ Discover a causal DAG and latent confounding variables
       :return: ``nx.DiGraph``: causal DAG
        """
        raise NotImplementedError("integrate Coco here")

    #%% SPACETIME with TOPIC
    def fit_graph_over_time_and_contexts(self) -> nx.DiGraph:
        """ Discover a causal DAG and causal changepoints
       :return: ``nx.DiGraph``: causal DAG
        """
        raise NotImplementedError("integrate Stime here")


    #%% Graph search algorithms
    def _graph_search(self) -> nx.DiGraph:
        if self.oracle_G: return self.graph_state
        if self.graph_search == GraphSearch.TOPIC:
            return self._graph_search_topological()
        elif self.graph_search == GraphSearch.GLOBE:
            return self._graph_search_edgegreedy()
        else: raise ValueError(self.graph_search)


    #%% Graph search - TOPIC
    def _graph_search_topological(self) -> nx.DiGraph:
        """ TOPIC (Xu et al., 2025) """

        self.graph_state.add_nodes_from(range(self.N))
        self.candidates = list(range(self.N))
        self._graph_search_topological_ordering()

        self.fitted_graph = True
        return self.graph_state


    def _graph_search_topological_ordering(self):
        it = 0
        while it < self.N:
            source = self._graph_search_topological_next(self.candidates if not self.oracle_order else self.true_top_order[it])
            self.candidates.remove(source)
            self.topological_order.append(source)
            it += 1
            self._info(f"\t{it}. Source: {source}\t current {self.topological_order}, true {self.true_top_order}", -2)

            self._graph_search_topological_add_outgoing(source)
            self._graph_search_topological_remove_ingoing(source)

        if self.extra_refinement:
            self._graph_search_topological_refine_extra()

    def _graph_search_topological_add_outgoing(self, source):
        for node in self.candidates:
            if node in self.topological_order or node == source or self.has_cycle(source, node):
                continue
            gain = self._addition_gain(node, source)
            if self._significant(gain) or self._add_greedily:
                self._add_edge(source, node, gain=float(gain))


    def _graph_search_topological_remove_ingoing(self, source):
        parents = list(self.graph_state.predecessors(source))
        n_removed = 0
        while n_removed < len(parents):
            removed_found, removed_parent = self._graph_search_topological_refine_step(source, parents)

            if removed_parent is not None:
                self._remove_edge(removed_parent, source)
                parents.remove(removed_parent)
                n_removed += 1
            else:
                break

    def _graph_search_topological_refine_step(self, source, parents):
        removed_found, best_parent, best_diff = False, None, -np.inf
        old_score = self._score(parents, source)

        for parent in parents:
            new_parents = parents.copy()
            new_parents.remove(parent)
            if len(new_parents) == 0: continue
            new_score = self._score(new_parents, source)
            diff = old_score - new_score  # >0 means removing parent improved the score

            if diff > best_diff and self.is_score_insignificant(abs(diff)):#self._significant(diff):
                best_diff = diff
                best_parent = parent
                removed_found = True
        return removed_found, best_parent

    def _graph_search_topological_next(self, candidates):
        if self.oracle_order:
            n = len(self.topological_order)
            self._info(f"\tTrue Next Node: {self.true_top_order[n]}", -2)
            return self.true_top_order[n]

        improvement = self._graph_search_topological_improvement_matrix(self.graph_state, candidates)
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

    def _graph_search_topological_improvement_matrix(self, graph, candidates):
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

    def _graph_search_topological_refine_extra(self, min_parent_set_size=0):
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

                    if self.is_score_insignificant(np.abs(gain)) and len(parent_set) < best_size:  # favor smaller parent sets
                        best_size = len(parent_set)
                        arg_max = parent_set

            if arg_max is None:
                continue
            self._info(f'\trefine {parents} to {arg_max} -> {j}', -2)
            for p in old_parents:
                if p not in arg_max:
                    self._remove_edge(p, j)

    # %% Graph search - GLOBE
    def _graph_search_edgegreedy(self) -> nx.DiGraph:
        """ GLOBE (Mian et al., 2021) """

        self._info(f"\t*** Greedy DAG search (phase 0) ***", -1)

        edge_q = UPQ()
        dag_model = DAG(self.X, self.N, self.edges_state)

        edge_q = dag_model.initial_edges(edge_q)
        edge_q, dag_model = self._graph_search_edgegreedy_forward(edge_q, dag_model)
        #edge_q, dag_model = self._graph_search_edgegreedy_backward(edge_q, dag_model)

        self._info(f'DAG search result:' + ', '.join(
            f"{i}->{j}" for i, j in set(itertools.product(set(range(self.N)), set(range(self.N)))) if
            dag_model.get_adj()[i][j] != 0))
        self.fitted_graph = True

        self.graph_state = nx.from_numpy_array(dag_model.get_adj(), create_using=nx.DiGraph)
        return self.graph_state

    def _graph_search_edgegreedy_forward(self, edge_q: UPQ, dag_model: DAG) -> [UPQ, DAG]:
        self._info(f"\t*** Greedy DAG search (phase 1) ***", -1)

        while edge_q.pq:
            try: edge_q, dag_model = self._graph_search_edgegreedy_forward_next(edge_q, dag_model)
            except KeyError: pass  # empty or all remaining entries are tagged as removed
        return edge_q, dag_model

    def _graph_search_edgegreedy_backward(self, edge_q: UPQ, dag_model: DAG) -> [UPQ, DAG]:
        self._info(f"\t*** Greedy DAG search (phase 2) ***", -1)
        for node in dag_model.nodes:
            edge_q, dag_model = self._graph_search_edgegreedy_backward_refine(node, edge_q, dag_model)
        return edge_q, dag_model

    def _graph_search_edgegreedy_backward_refine(self, node, edge_q, dag_model):
        P = list(self._dm_parents_of(dag_model, node))
        if len(P) <= 1:
            return edge_q, dag_model

        # search over all subsets of current parents
        max_gain, arg_max = -np.inf, None
        old_score = self._score(P, node)
        for k in range(0, len(P) + 1):
            for parent_set in itertools.combinations(P, k):
                new_score = self._score(list(parent_set), node)
                gain = self._gain(new_score, old_score)  # old - new
                if gain > max_gain:
                    max_gain, arg_max = gain, parent_set

        if (arg_max is not None) and (not self.is_score_insignificant(max_gain)):
            # apply the best parent subset: remove dropped parents, add missing ones
            keep = set(arg_max)
            for p in list(P):
                if p not in keep:
                    self._dm_remove_edge(dag_model, p, node, vb=-2)
            for p in keep:
                if not self._dm_is_edge(dag_model, p, node):
                    self._dm_add_edge(dag_model, p, node, score=None, gain=None, vb=-2)

            for op in dag_model.nodes:
                edge_q, dag_model = self._graph_search_edgegreedy_update_parents(op, node, -1, edge_q, dag_model)

        return edge_q, dag_model

    """
    def _graph_search_edgegreedy_forward(self, edge_q: UPQ, dag_model: DAG) -> [UPQ, DAG]:
        self._info(f"\t*** Greedy DAG search (phase 1) ***", -1)

        while edge_q.pq:
            try: edge_q, dag_model = self._graph_search_edgegreedy_forward_next(edge_q, dag_model)
            except KeyError: pass  # empty or all remaining entries are tagged as removed
        return edge_q, dag_model

    def _graph_search_edgegreedy_backward(self, edge_q: UPQ, dag_model: DAG) -> [UPQ, DAG]:
        self._info(f"\t*** Greedy DAG search (phase 2) ***", -1)

        for node in dag_model.nodes:
            edge_q, dag_model = self._graph_search_edgegreedy_backward_refine(node, edge_q, dag_model)

        return edge_q, dag_model

    def _graph_search_edgegreedy_backward_refine(self, node, edge_q, dag_model):
        P = list(dag_model.parents_of(node))
        if len(P) <= 1:
            return edge_q, dag_model
        max_gain, arg_max = -np.inf, None
        min_size = 0  # allow empty set if desired
        for k in range(min_size, len(P) + 1):
            for parent_set in itertools.combinations(P, k):
                gain, _ = dag_model.eval_edges(node, parent_set)
                if gain > max_gain:
                    max_gain, arg_max = gain, parent_set
        if (arg_max is not None) and (not is_score_insignificant(max_gain)):
            dag_model.update_edges(node, arg_max)
            # optionally refresh queue entries affected by node’s parents
            for op in dag_model.nodes:
                edge_q, dag_model = self._graph_search_edgegreedy_update_parents(op, node, -1, edge_q, dag_model)
        return edge_q, dag_model

    def _graph_search_edgegreedy_forward_next(self, edge_q: UPQ, dag_model: DAG) -> [UPQ, DAG]:
        #Greedy tree search, forward phase -- evaluate the next causal edge in the priority queue.  
        pi_edge = edge_q.pop_task()
        node, parent = pi_edge.j, pi_edge.pa
        if dag_model.has_cycle(parent, node) or dag_model.exists_anticausal_edge(parent, node):  return edge_q, dag_model
        gain, score = dag_model.eval_edge_addition(node, parent, return_score=True)
        if is_score_insignificant(gain): return edge_q, dag_model

        dag_model.add_edge(parent, node, score, gain)
        # Reconsider children under current model and remove if reversing the edge improves score
        for child in dag_model.nodes:
            edge_q, dag_model = self._graph_search_edgegreedy_update_children(child, node, edge_q, dag_model)

        # Reconsider edges Xk->Xj in q given the current model as their score changed upon adding Xi->Xj
        for other_parent in dag_model.nodes:
            edge_q, dag_model = self._graph_search_edgegreedy_update_parents(other_parent, node, parent, edge_q, dag_model)

        return edge_q, dag_model

    def _graph_search_edgegreedy_update_children(self, child, node, edge_q, dag_model):
        ch = child
        if not dag_model.is_edge(node, ch):
            return edge_q, dag_model

        gain = dag_model.eval_edge_flip(node, ch)
        if is_score_insignificant(gain):
            return edge_q, dag_model

        dag_model.remove_edge(node, ch)

        edge_fw = dag_model.pair_edges[node][ch]
        edge_bw = dag_model.pair_edges[ch][node]
        if edge_q.exists_task(edge_bw):
            edge_q.remove_task(edge_bw)

        gain_bw = dag_model.eval_edge_addition(edge_bw.i, edge_bw.j)
        gain_fw = dag_model.eval_edge_addition(edge_fw.i, edge_fw.j)

        if not is_score_insignificant(gain_bw):
            edge_q.add_task(edge_bw, gain_bw * 100)
        if not is_score_insignificant(gain_fw):
            edge_q.add_task(edge_fw, gain_fw * 100)
        return edge_q, dag_model

    def _graph_search_edgegreedy_update_parents(self, other_parent, node, parent, edge_q, dag_model):
        if node == other_parent or parent == other_parent \
                or dag_model.is_edge(other_parent, node) or dag_model.is_edge(node, other_parent):
            return edge_q, dag_model

        edge_candidate = dag_model.pair_edges[other_parent][node]
        gain_mom = dag_model.eval_edge_addition(node, other_parent)
        if edge_q.exists_task(edge_candidate):
            edge_q.remove_task(edge_candidate)
        if not is_score_insignificant(gain_mom):
            edge_q.add_task(edge_candidate, gain_mom * 100)
        return edge_q, dag_model
    """

    def _dm_has_cycle(self, dag_model, parent, child):
        return self.has_cycle(parent, child)

    def _dm_exists_anticausal(self, dag_model, parent, child):
        try:
            return dag_model.is_edge(child, parent)
        except AttributeError:
            return self.graph_state.has_edge(child, parent)

    def _dm_parents_of(self, dag_model, node):
        try:
            return list(dag_model.parents_of(node))
        except AttributeError:
            return list(self.graph_state.predecessors(node))

    def _dm_is_edge(self, dag_model, u, v):
        try:
            return dag_model.is_edge(u, v)
        except AttributeError:
            return self.graph_state.has_edge(u, v)

    def _dm_add_edge(self, dag_model, parent, child, score=None, gain=None, vb=-2):
        try:
            dag_model.add_edge(parent, child, score, gain)
        except AttributeError:
            pass
        self._add_edge(parent, child, vb=vb, gain=gain)

    def _dm_remove_edge(self, dag_model, parent, child, vb=-2):
        try:
            dag_model.remove_edge(parent, child)
        except AttributeError:
            pass
        if self.graph_state.has_edge(parent, child):
            self._remove_edge(parent, child, vb=vb)


    def _dm_eval_edge_addition(self, dag_model, target, parent, return_score=False):
        """
        gain = old_score - new_score (higher better)
        """
        parents = self._dm_parents_of(dag_model, target)
        if parent in parents:
            gain = 0.0
            sc = self._score(parents, target)
            return (gain, sc) if return_score else gain
        old_score = self._score(parents, target)
        new_score = self._score(parents + [parent], target)
        gain = self._gain(new_score, old_score)
        if return_score:
            return gain, new_score
        return gain

    def _dm_eval_edge_flip(self, dag_model, u, v):
        Pv = self._dm_parents_of(dag_model, v)
        Pu = self._dm_parents_of(dag_model, u)
        if u not in Pv:
            return 0.0
        old = self._score(Pv, v) + self._score(Pu, u)
        Pv_new = [p for p in Pv if p != u]
        Pu_new = Pu + [v] if v not in Pu else Pu
        new = self._score(Pv_new, v) + self._score(Pu_new, u)
        return self._gain(new, old)

    def _graph_search_edgegreedy_forward_next(self, edge_q: UPQ, dag_model: DAG) -> [UPQ, DAG]:
        pi_edge = edge_q.pop_task()
        node, parent = pi_edge.j, pi_edge.pa
        if self._dm_has_cycle(dag_model, parent, node) or self._dm_exists_anticausal(dag_model, parent, node):
            return edge_q, dag_model

        gain, score = self._dm_eval_edge_addition(dag_model, node, parent, return_score=True)
        if self.is_score_insignificant(gain):
            return edge_q, dag_model

        self._dm_add_edge(dag_model, parent, node, score, gain)

        for child in dag_model.nodes:
            edge_q, dag_model = self._graph_search_edgegreedy_update_children(child, node, edge_q, dag_model)

        for other_parent in dag_model.nodes:
            edge_q, dag_model = self._graph_search_edgegreedy_update_parents(other_parent, node, parent, edge_q,
                                                                             dag_model)

        return edge_q, dag_model

    def _graph_search_edgegreedy_update_children(self, child, node, edge_q, dag_model):
        ch = child
        if not self._dm_is_edge(dag_model, node, ch):
            return edge_q, dag_model

        gain = self._dm_eval_edge_flip(dag_model, node, ch)
        if self.is_score_insignificant(gain):
            return edge_q, dag_model

        self._dm_remove_edge(dag_model, node, ch)

        edge_fw = dag_model.pair_edges[node][ch]  # keep your queue item
        edge_bw = dag_model.pair_edges[ch][node]

        if edge_q.exists_task(edge_bw):
            edge_q.remove_task(edge_bw)

        gain_bw = self._dm_eval_edge_addition(dag_model, edge_bw.j, edge_bw.i)
        gain_fw = self._dm_eval_edge_addition(dag_model, edge_fw.j, edge_fw.i)

        if not self.is_score_insignificant(gain_bw):
            edge_q.add_task(edge_bw, gain_bw * 100)
        if not self.is_score_insignificant(gain_fw):
            edge_q.add_task(edge_fw, gain_fw * 100)
        return edge_q, dag_model

    def _graph_search_edgegreedy_update_parents(self, other_parent, node, parent, edge_q, dag_model):
        if node == other_parent or parent == other_parent \
                or self._dm_is_edge(dag_model, other_parent, node) or self._dm_is_edge(dag_model, node, other_parent):
            return edge_q, dag_model

        edge_candidate = dag_model.pair_edges[other_parent][node]
        gain_mom = self._dm_eval_edge_addition(dag_model, node, other_parent)

        if edge_q.exists_task(edge_candidate):
            edge_q.remove_task(edge_candidate)
        if not self.is_score_insignificant(gain_mom):
            edge_q.add_task(edge_candidate, gain_mom * 100)
        return edge_q, dag_model

    def _dm_eval_parent_set(self, dag_model, node, parent_set):
        old_parents = self._dm_parents_of(dag_model, node)
        old = self._score(old_parents, node)
        new = self._score(list(parent_set), node)
        return self._gain(new, old)  # old - new

    #%% Mixing Variable Search
    def fit_latent_discrete_given_DAG(self, graph_adj: XArray, skip_pruning=False, skip_sets=False) -> None:
        graph = nx.from_numpy_array(graph_adj, create_using=nx.DiGraph)
        self._info(f"\t*** Causal Mixture Modeling (given DIgraph w {len(graph.edges())} edges) ***")

        self.graph_state = graph.copy()
        self.initialize()
        #self.edges_state = EdgeMemoized(self.X, self.data_mode, self.score_type, self.mixing_type)
        self.fitted_mixing = True

        self._fit_latent_discrete_per_node()
        if not skip_sets: self._fit_latent_discrete_per_set()
        if skip_pruning: return

        #self.edges_state = EdgeMemoized(self.X, self.data_mode, self.score_type, self.mixing_type)
        self.initialize()
        self.prune_spurious_edges(self.graph_state.copy())
        self._fit_latent_discrete_per_node()
        if not skip_sets: self._fit_latent_discrete_per_set()

    def _fit_latent_discrete_per_node(self) -> None:
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

    def _fit_latent_discrete_per_set(self, n_nodes=None) -> None:
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
    def prune_spurious_edges_local(self, node, parents) -> list:
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
            parent_subset = self.prune_spurious_edges_local(node, parents_spurious)

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

    def _addition_gain(self, node, source):
        parents = list(self.graph_state.predecessors(node)).copy() if nx.is_directed_acyclic_graph(self.graph_state) else  list(self.graph_state.neighbors(node)).copy()
        old_score = self._score(parents, node)
        parents.append(source)
        new_score = self._score(parents, node)
        gain = self._gain(new_score, old_score)
        return gain

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