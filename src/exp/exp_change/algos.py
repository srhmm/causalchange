import time
from abc import ABC, abstractmethod
from enum import Enum

import networkx as nx
import numpy as np
import pandas as pd

from src.baselines.sep_distances.codebase import mixed_graph as graph_lmg
from src.baselines.sep_distances.codebase.mixed_graph import LabelledMixedGraph
from src.causalchange.greedy_equivalent_causal_mixture import ges_causal_mixture
from src.causalchange.mixing.mixing import MixingType
from src.causalchange.mixing.regression import ScoreType, GPType, DataMode
from src.causalchange.causal_change_topo import CausalChangeTopological
from src.causalchange.util.util import compare_lmg_DAG, compare_lmg_CPDAG, nxdigraph_to_lmg, general_graph_to_lmg, \
    causaldag_to_lmg, general_graph_to_directed_edge_adj, general_graph_to_undirected_edge_adj, augmented_graph_to_lmg, \
    lmg_to_directed_edge_adj


class DAGType(Enum):
    """ result that a method returns, DAG or (C)PDAG """
    DAG = 0
    CPDAG = 1
    PAG = 2



class CD(Enum):
    """ causal discovery methods. """
    SKIP = 'skip'
     # iid setting
    TOPIC = 'topic'
    # multi-conext data
    TopicContextsGP = 'topic-contexts-gp'
    TopicContextsRFF = 'topic-contexts-rff'
    # mix-of-mechanisms data
    CausalMixtures = 'causal-mixtures'
    CausalMixturesGES = 'causal-mixtures-ges'
    # baselines, iid
    PC_PC = 'pc-partial-correl'
    PC_KCI = 'pc-kci-partial-correl'
    FCI_PC = 'fci'
    FCI_KCI = 'fci-kci'
    GES = 'ges'
    CAM = 'cam'
    LINGAM = 'lingam'
    SCORE = 'score'
    DAS = 'das'
    NOGAM = 'nogam'
    CAM_UV = 'cam-uv'
    R2SORT = 'r2sort'
    RANDSORT = 'randsort'
    VARSORT = 'varsort'
    # baselines, mix-of-mechanisms
    MixtureUTIGSP = 'mix-utigsp'
    # baselines, multi-conext
    CDNOD_PC = 'cdnod-pc'
    CDNOD_KCI = 'cdnod-kci'
    JCI_PC_PC = 'jci-pc-pc'
    JCI_PC_KCI = 'jci-pc-kci'
    JCI_FCI_PC = 'jci-fci-pc'
    JCI_FCI_KCI = 'jci-fci-kci'
    UTIGSP = 'utigsp'

    def __str__(self):
        return str(self.value) if self.value != CD.SKIP.value else ''

    def get_method(self):
        """ all implemented methods"""
        if self.value  == CD.TopicContextsGP.value:
            return TopicContextsGPMethod(self)
        elif self.value  == CD.TopicContextsRFF.value:
            return TopicContextsRFFMethod(self)
        elif self.value == CD.CausalMixtures.value:
            return CausalMixtureMethod(self)
        elif self.value == CD.CausalMixturesGES.value:
            return CausalMixtureMethodGES(self)
        elif self.value in [CD.PC_PC.value, CD.PC_KCI.value]:
            return PCMethod(self)
        if self.value in [CD.FCI_PC.value, CD.FCI_KCI.value]:
            return FCIMethod(self)
        elif self.value in [CD.GES.value]:
            return GESMethod(self)
        elif self.value in [CD.R2SORT.value, CD.RANDSORT.value, CD.VARSORT.value]:
            return SortingMethod(self)
        # elif self.value == CD.RESIT.value:
        #    return RESITMethod(self)
        elif self.value == CD.CAM_UV.value:
            return CAMUVMethod(self)
        elif self.value == CD.TOPIC.value:
            return TOPICMethod(self)
        elif self.value == CD.LINGAM.value:
            return DirectLINGAMMethod(self)
        elif self.value in [CD.CDNOD_PC.value, CD.CDNOD_KCI.value]:
            return CDNODMethod(self)
        elif self.value in [
            CD.SCORE.value,
            CD.CAM.value,
            CD.NOGAM.value,
            CD.DAS.value,
        ]:
            return DoDiscoverMethod(self)
        elif self.value in [CD.JCI_PC_PC.value, CD.JCI_PC_KCI.value]:
            return JCIPCMethod(self)
        elif self.value in [CD.JCI_FCI_PC.value, CD.JCI_FCI_KCI.value]:
            return JCIFCIMethod(self)
        elif self.value == CD.UTIGSP.value:
            return UTIGSPMethod(self)
        elif self.value == CD.MixtureUTIGSP.value:
            return MixtureUTIGSPMethod(self)
        elif self.value == CD.SKIP.value:
            raise ValueError("placeholder when causal discovery is skipped")
        raise ValueError("unsupported method")

    def discovers_mixture_assignments(self):
        return self.value in [CD.CausalMixtures.value, CD.CausalMixturesGES.value, CD.MixtureUTIGSP.value]

    def discovers_context_changes(self):
        return self.value in [CD.TopicContextsGP.value,CD.TopicContextsRFF.value]  #,
                             # TODO CD.UTIGSP.value, CD.CDNOD_KCI.value, CD.CDNOD_PC.value,
                             # CD.JCI_FCI_PC.value, CD.JCI_FCI_KCI.value,
                              #CD.JCI_PC_PC.value, CD.JCI_PC_KCI.value]

class OracleType(Enum):
    """Decides between different types of oracles for our method"""
    trueGtrueZ = 'trueGtrueZ'
    # Find Z
    trueGtrueK = 'trueGtrueK'
    trueGhatZ = 'trueGhatZ'
    emptyGhatZ = 'emptyGhatZ'
    fullGhatZ = 'fullGhatZ'
    # Find G
    hatGhatZ = 'hatGhatZ'
    hatGtrueZ = 'hatGtrueZ'
    SKIP = ''

    def __str__(self):
        return str(self.value) if self.value != OracleType.SKIP.value else ''

    def is_G_known(self):
        return self.value in [
            OracleType.trueGtrueZ.value, OracleType.trueGtrueK.value, OracleType.trueGhatZ.value, OracleType.SKIP.value
        ]

    def is_G_empty(self): return self.value in [OracleType.emptyGhatZ.value]

    def is_G_dense(self): return self.value in [OracleType.fullGhatZ.value]

    def is_Z_known(self): return self.value in [OracleType.trueGtrueZ.value, OracleType.hatGtrueZ.value]

    def is_K_known(self): return self.value in [OracleType.trueGtrueK.value]

    def haveto_discover_G(self):
        return self.value.startswith('hatG')



class CausalDiscoveryMthd(ABC):

    def __init__(self, ty: CD):
        self.ty = ty
        self.metrics: dict = {}
        self.dag: nx.DiGraph = nx.DiGraph()
        self.lmg: graph_lmg.LabelledMixedGraph = graph_lmg.LabelledMixedGraph()
        self.pag: graph_lmg.LabelledMixedGraph = graph_lmg.LabelledMixedGraph()
        self.model = None
        self.e_n_Z = {}
        self.e_Z_n = {}

    @staticmethod
    @abstractmethod
    def dag_ty() -> DAGType:
        pass

    def nm(self) -> str:
        return self.ty.value

    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs):
        """ causal discovery of PAG or DAG
        :param data: data
        :param kwargs: method parameters
        :return:
        """
        pass

    def get_directed_graph(self):
        assert self.dag_ty() == DAGType.DAG
        return nx.to_numpy_array(self.dag)

    def get_labelled_mixed_graph(self):
        return self.lmg

    def get_labelled_mixed_pag(self):
        assert self.dag_ty() == DAGType.PAG
        return self.pag

    def get_mixture_assignment_node(self):
        assert self.ty.discovers_mixture_assignments()
        return self.e_n_Z

    def get_mixed_node_sets(self):
        assert self.ty.discovers_mixture_assignments()
        return self.e_Z_n

    def get_graph_metrics(self, true_nxg):
        true_lmg = nxdigraph_to_lmg(true_nxg)
        est_lmg = self.get_labelled_mixed_graph()

        if self.dag_ty() == DAGType.DAG:
            return compare_lmg_DAG(true_lmg, est_lmg)
        elif self.dag_ty() == DAGType.CPDAG:
            try: return compare_lmg_CPDAG(true_lmg, est_lmg)
            except nx.NetworkXError:
                est_lmg = nxdigraph_to_lmg(nx.from_numpy_array(lmg_to_directed_edge_adj(est_lmg)))
                return compare_lmg_CPDAG(true_lmg, est_lmg)
        elif self.dag_ty() == DAGType.PAG:
            est_cpdag = self.get_labelled_mixed_graph()
            est_pag = self.get_labelled_mixed_graph()
            return compare_lmg_CPDAG(true_lmg, est_cpdag)
        else:
            raise ValueError(self.dag_ty())

    def get_target_metrics(self, truths):
        assert self.ty.discovers_context_changes()
        if self.changes is not None: pass
        if self.targets is not None: pass
        if self.targets_each_context is not None: pass

# %% Our Methods ############

# IID Data
class TOPICMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from src.causalchange.causal_change_topo import CausalChangeTopological
        kwargs["data_mode"] = DataMode.IID
        top = CausalChangeTopological(**kwargs)
        time_st = time.perf_counter()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = top.fit_graph_iid(X)
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = top

# Multi-Context Data
class TopicContextsGPMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from src.causalchange.causal_change_topo import CausalChangeTopological
        #kwargs["hybrid"] = False
        kwargs["data_mode"] = DataMode.CONTEXTS
        kwargs["mixing_type"] = MixingType.SKIP
        kwargs["score_type"] = GPType.EXACT
        top = CausalChangeTopological(**kwargs)
        time_st = time.perf_counter()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = top.fit_graph_over_contexts(X)
        self.changes, self.targets, self.targets_each_context  = top.changes, top.targets, top.targets_each_context
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = top

class TopicContextsRFFMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from src.causalchange.causal_change_topo import CausalChangeTopological
        #kwargs["hybrid"] = False
        kwargs["data_mode"] = DataMode.CONTEXTS
        kwargs["mixing_type"] = MixingType.SKIP
        kwargs["score_type"] = GPType.FOURIER
        top = CausalChangeTopological(**kwargs)
        time_st = time.perf_counter()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = top.fit_graph_over_contexts(X)
        self.changes, self.targets, self.targets_each_context  = top.changes, top.targets, top.targets_each_context
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = top


# Mixetures of Mechanisms/Populations
class CausalMixtureMethod(CausalDiscoveryMthd, ABC):
    allowed_params = ['truths', 'hybrid', 'pruning_G', 'oracle_G', 'oracle_K', 'oracle_Z', 'lg', 'k_max', 'vb', 'mixing_type']
    e_Z = {}
    Z_pairs = {}
    pprobas = {}
    idls = {}

    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        params = {ky: val for ky, val in kwargs.items() if ky in self.allowed_params}
        params["data_mode"] = DataMode.MIXED
        params["score_type"] = ScoreType.LIN
        params["mixing_type"] = MixingType.MIX_LIN
        top = CausalChangeTopological(**params)
        time_st = time.perf_counter()
        top.fit_graph_and_mixtures(X)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = top.topic_graph
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = top

        # information on reconstructed mixing variables, targeted observed variable sets
        self.e_n_Z = top.e_n_Z
        self.e_Z_n = top.e_Z_n
        self.e_Z = top.e_Z
        self.Z_pairs = top.Z_pairs
        self.pprobas = top.pprobas
        self.idls = top.idls

class CausalMixtureMethodGES(CausalDiscoveryMthd, ABC):
    allowed_params = ['oracle_K', 'oracle_Z', 'k_max', 'vb']
    e_Z = {}
    Z_pairs = {}
    pprobas = {}
    idls = {}

    @staticmethod
    def dag_ty(): return DAGType.CPDAG

    def fit(self, X, **kwargs):
        from causallearn.graph import GeneralGraph
        params = {ky: val for ky, val in kwargs.items() if ky in self.allowed_params}
        ges_score = "local_score_latent_BIC"

        time_st = time.perf_counter()
        ges_obj = ges_causal_mixture(X, ges_score, parameters=params)
        # -------
        # ges_obj['G']: learned causal graph, where ges_obj['G'].graph[j,i]=1 and ges_obj['G'].graph[i,j]=-1 indicates  i --> j ,
        #            ges_obj['G'].graph[i,j] = ges_obj['G'].graph[j,i] = -1 indicates i --- j.

        gg: GeneralGraph = ges_obj['G']
        self.lmg = general_graph_to_lmg(gg)
        self.model = ges_obj

        # reconstruct the mixing variables under G
        adj = general_graph_to_undirected_edge_adj(gg)
        hypparams = dict(oracle_Z=False, oracle_K=False, oracle_G=False, k_max=params.get('k_max', 5),
                         data_mode=DataMode.MIXED, score_type=ScoreType.LIN, mixing_type=MixingType.MIX_LIN)

        top = CausalChangeTopological(**hypparams)
        top.fit_Z_given_G(X, adj, skip_pruning=True) #pruning is only for the power-speci experiments
        self.metrics = {'time': time.perf_counter() - time_st}

        self.model = {'ges-with-latent-bic': ges_obj, 'mixture-variable-extraction': top}

        # information on reconstructed mixing variables, targeted observed variable sets
        self.e_n_Z = top.e_n_Z
        self.e_Z_n = top.e_Z_n
        self.e_Z = top.e_Z
        self.Z_pairs = top.Z_pairs
        self.pprobas = top.pprobas
        self.idls = top.idls


# %% BASELINES ############
class MixtureUTIGSPMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        if "intv_args_dict" not in kwargs or "args" not in kwargs: raise Warning(
            "Usage (MixtureUTIGSP): provide hyperparameters in 'args'")
        model = None
        time_st = time.perf_counter()

        # Step 1: Disentanglement/mixture modelling
        from src.baselines.mixture_mec.mixture_solver import GaussianMixtureSolver
        mixture_samples = X
        intv_args_dict = kwargs.get("intv_args_dict", {})
        args = kwargs.get("args", {})

        gSolver = GaussianMixtureSolver(args["dtype"])
        err, intv_args_dict, weight_precision_error, est_num_comp, gm_score_dict, gm \
            = gSolver.mixture_disentangler(
            args["num_tgt_prior"],
            intv_args_dict,
            mixture_samples,
            args["gmm_tol"],
            args["cutoff_drop_ratio"],
        )

        # Step 2: structure learning and intervention target identification
        est_dag, intv_args_dict, oracle_est_dag, igsp_est_dag, intv_base_est_dag \
            = gSolver.identify_intervention_utigsp(
            intv_args_dict, args["stage2_samples"])
        self.metrics = {'time': time.perf_counter() - time_st}

        # Result extraction
        est_tgts = [
            node_i for node_i in range(mixture_samples.shape[1]) if
            any(["est_tgt" in intv_args_dict[ky] and node_i in intv_args_dict[ky]["est_tgt"] and ky != "obs" for ky in
                 intv_args_dict])]
        self.lmg = causaldag_to_lmg(est_dag)
        self.model = model
        self.e_Z_n = [gm.predict(mixture_samples) if node_i in est_tgts else np.zeros(mixture_samples.shape[0]) for
                      node_i in range(mixture_samples.shape[1])]
        self.e_n_Z = [est_tgts]


class UTIGSPMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester
        from causaldag import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester

        model = None
        time_st = time.perf_counter()
        from causaldag import unknown_target_igsp

        nnodes = X[0].shape[1]
        nodes = set(range(nnodes))
        num_settings = len(X)-1
        targets_list = [None for _ in range(num_settings)]

        obs_samples = X[0]
        iv_samples_list =  [X[c_i] for c_i in range(1, len(X))]

        # Form sufficient statistics
        obs_suffstat = partial_correlation_suffstat(obs_samples)
        invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)

        # Create conditional independence tester and invariance tester
        alpha = 1e-3
        alpha_inv = 1e-3
        ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

        # Run UT-IGSP
        setting_list = [dict(known_interventions=[]) for _ in targets_list]
        est_dag, est_targets_list = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
        print(est_targets_list)

        self.metrics = {'time': time.perf_counter() - time_st}
        self.targets_each_context = est_targets_list
        self.targets = [n_i for n_i in range(nnodes) if any ([n_i in est_targets_list[c_i] for c_i in range(len(est_targets_list)) ]) ]
        self.lmg = causaldag_to_lmg(est_dag)
        self.causaldag_dag = est_dag
        self.model = model


def remove_cycles(lmg):
    Gcy = nx.from_numpy_array(lmg_to_directed_edge_adj(lmg))
    G = nx.DiGraph()
    G.add_nodes_from(Gcy.nodes)
    G.add_edges_from(Gcy.edges)
    cycles = list(nx.simple_cycles(G))
    edges_to_remove = []
    sccs = [s for s in nx.strongly_connected_components(G) if len(s) > 1]
    if len(sccs) == 0:
        return False, lmg
    for s in sccs:
        sub = G.subgraph(s).copy()
        e = nx.find_cycle(sub)[0][:2]
        edges_to_remove.append(e)
    G_acyclic = G.copy()
    G_acyclic.remove_edges_from(edges_to_remove)
    assert nx.is_directed_acyclic_graph(G_acyclic)

    lmg = nxdigraph_to_lmg(G_acyclic)
    return True, lmg


class CDNODMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.CPDAG

    def fit(self, X, **kwargs):
        from causallearn.graph.GeneralGraph import GeneralGraph
        model = None
        time_st = time.perf_counter()

        from causallearn.search.ConstraintBased.CDNOD import cdnod
        contexts = sorted(X.keys())
        blocks = [np.asarray(X[k]) for k in contexts]
        data = np.vstack(blocks)
        n_contexts = len(X)
        c_indx = np.concatenate([np.full(len(b), k, dtype=int) for k, b in zip(contexts, blocks)]).reshape(-1, 1)
        assert(len(np.unique(c_indx)) == n_contexts)
        from causallearn.search.ConstraintBased.CDNOD import cdnod

        # default parameters
        indep_test = 'mv_fisherz' if self.ty == CD.CDNOD_PC else 'kci'
        cg = cdnod(data, c_indx, indep_test=indep_test)
        # or customized parameters
        #cg = cdnod(data, c_indx, alpha, indep_test, stable, uc_rule, uc_priority, mvcdnod,
        #           correction_name, background_knowledge, verbose, show_progress)
        self.metrics = {'time': time.perf_counter() - time_st}
        gg:  GeneralGraph = cg.G

        self.lmg = augmented_graph_to_lmg(gg) # important to use the augmented function here.

        has_cy, lmg_acy = remove_cycles(self.lmg)
        if has_cy: self.lmg = nxdigraph_to_lmg(lmg_acy)

        ix_augmented_node = len(gg.nodes) - 1
        targets = [ix_node for ix_node in range(len(gg.nodes) - 1) if
                   (gg.graph[ix_augmented_node][ix_node] == -1 and gg.graph[ix_node][ix_augmented_node] == 1)]

        self.targets = targets
        self.model = cg



class DoDiscoverMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from src.baselines.dodiscover import make_context

        from src.baselines.dodiscover.toporder import SCORE, CAM, NoGAM, DAS
        model = SCORE() if self.ty.value == CD.SCORE.value else \
            CAM() if self.ty.value == CD.CAM.value else \
                NoGAM() if self.ty.value == CD.NOGAM.value else \
                    DAS() if self.ty.value == CD.DAS.value \
                        else None
        score_context = make_context().variables(data=pd.DataFrame(X)).build()
        time_st = time.perf_counter()
        model.learn_graph(pd.DataFrame(X), score_context)
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = model.graph_
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = model



class _JCIFCIMethod_R(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty():
        return DAGType.PAG

    def fit(self, X, **kwargs):
        """
        X: dict {context_key -> np.ndarray (n_k, p)} with mutually-exclusive contexts.

        Sets:
          - self.lmg : LabelledMixedGraph over ONLY the system variables (no context nodes)
          - self.targets : set of system node indices j (0..p-1) with any context -> j (arrowhead at j)
          - self.targets_per_context : list of lists; per context column, nodes j with that context -> j
        """
        from rpy2 import robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()

        alpha = float(kwargs.get("alpha", 0.05))
        jci = kwargs.get("jci", "123")
        selection_bias = bool(kwargs.get("selectionBias", False))
        verbose_r = bool(kwargs.get("verbose_r", False))
        indep_test = kwargs.get(
            "indep_test",
            "gauss" if getattr(self, "ty", None) == CD.JCI_FCI_PC else "kernel"
        )
        # Keep all context columns so every context can generate targets
        one_hot_drop_last = bool(kwargs.get("one_hot_drop_last", False))
        contexts = sorted(X.keys())
        blocks = [np.asarray(X[k]) for k in contexts]
        data = np.vstack(blocks)  # (N, p)
        N, p = data.shape
        K = len(contexts)

        # ---- build one-hot for contexts (optionally drop last) ----
        oh = np.zeros((N, K), dtype=float)
        r = 0
        for idx, b in enumerate(blocks):
            n_k = len(b)
            oh[r:r + n_k, idx] = 1.0
            r += n_k
        if one_hot_drop_last and K > 1:
            C = oh[:, :K - 1]
            C_labels = [f"C{j}" for j in range(K - 1)]
        else:
            C = oh
            C_labels = [f"C{j}" for j in range(K)]

        data_aug = np.hstack([data, C])  # (N, p + q)
        q = C.shape[1]
        labels = [f"X{i}" for i in range(p)] + C_labels

        # 1-based indices in R for the context columns:
        context_cols_r = list(range(p + 1, p + q + 1))

        # -------- R: run FCI-JCI --------
        ro.r('suppressMessages({ library(pcalg) })')
        r_data = ro.r.matrix(data_aug, nrow=N, ncol=p + q)
        ro.r.assign("X_py", r_data)
        ro.r.assign("labels_py", ro.StrVector(labels))
        ro.r.assign("contextVars_py", ro.IntVector(context_cols_r))
        ro.r("colnames(X_py) <- labels_py")

        if indep_test == "gauss":
            ro.r("""
                suffStat <- list(C = cor(X_py), n = nrow(X_py))
                indepTestFun <- gaussCItest
            """)
        elif indep_test == "kernel":
            ro.r('suppressMessages({ '
                 'if (!"kpcalg" %in% rownames(installed.packages())) '
                 'stop("Install R package \'kpcalg\' for kernelCItest"); '
                 'library(kpcalg) })')
            ro.r("""
                suffStat <- list(data = X_py)
                indepTestFun <- function(x, y, S, suffStat) {
                  kpcalg::kernelCItest(x, y, S, suffStat$data)
                }
            """)
        else:
            raise ValueError("indep_test must be 'gauss' or 'kernel'")

        ro.r(f"""
            set.seed(1)
            res <- fci(
                suffStat      = suffStat,
                indepTest     = indepTestFun,
                alpha         = {alpha},
                labels        = labels_py,
                p             = ncol(X_py),
                jci           = "{jci}",
                contextVars   = contextVars_py,
                selectionBias = {'TRUE' if selection_bias else 'FALSE'},
                verbose       = {'TRUE' if verbose_r else 'FALSE'}
            )
            amat <- res@amat
        """)

        amat = np.array(ro.r("amat"), dtype=int)  # (p+q) x (p+q)

        # pcalg mark codes per (i,j): 0 none, 1 circle at j, 2 arrow at j, 3 tail at j
        def _project_pag_to_cpdag_lmg(M, p):
            G = LabelledMixedGraph(nodes=set(range(p)))
            for i in range(p):
                for j in range(i + 1, p):
                    a, b = M[i, j], M[j, i]
                    # directed only if (arrow,tail) or (tail,arrow)
                    if a == 2 and b == 3:  # i -> j
                        G.add_directed(i, j);
                        continue
                    if a == 3 and b == 2:  # j -> i
                        G.add_directed(j, i);
                        continue
                    # undirected only if (tail,tail)
                    if a == 3 and b == 3:  # i -- j
                        G.add_undirected(i, j);
                        continue
                    # drop everything else: (2,2)=↔, any 1=circle, (2,1),(1,2)=*->, etc.
            return G

        ctx_py_indices = list(range(p, p + q))
        targets_per_context = []
        targets_set = set()
        for cpi in ctx_py_indices:
            js = [j for j in range(p) if amat[cpi, j] == 2]
            targets_per_context.append(js)
            targets_set.update(js)

        time_st = time.perf_counter()
        lmg_system = _project_pag_to_cpdag_lmg(amat, p)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.lmg = lmg_system
        self.targets = sorted(targets_set)
        self.targets_per_context = targets_per_context
        self.model = {
            "amat": amat,
            "labels": labels,
            "system_p": p,
            "context_cols_r": context_cols_r,
            "context_cols_py": ctx_py_indices,
        }


class PCMethod(CausalDiscoveryMthd, ABC):
    """causal-learn implementation"""

    @staticmethod
    def dag_ty(): return DAGType.CPDAG

    def fit(self, X, **kwargs):
        from causallearn.search.ConstraintBased.PC import pc
        indep_test = 'mv_fisherz' if self.ty == CD.PC_PC else 'kci'
        time_st = time.perf_counter()
        pc_obj = pc(X, indep_test=indep_test)
        # -------
        # G : a CausalGraph object, where G.graph[j,i]=1 and G.graph[i,j]=-1 indicates  i --> j ,
        #                G.graph[i,j] = G.graph[j,i] = -1 indicates i --- j,
        #                G.graph[i,j] = G.graph[j,i] = 1 indicates i <-> j.
        gg = pc_obj.G
        self.lmg = general_graph_to_lmg(gg)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.model = pc_obj


class GESMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.CPDAG

    def fit(self, X, **kwargs):
        # cdt implementation:
        # import cdt.causality.graph as algs
        # obj = algs.GES()
        # datafr = pd.DataFrame(data)
        # self.untimed_graph = obj.predict(datafr)

        from causallearn.search.ScoreBased.GES import ges
        from causallearn.graph import GeneralGraph
        ges_score = "local_score_BIC" #if self.ty == CD.GES else None
            #"local_score_CV_multi" if self.ty == CD.GGES_CV else \
            #    "local_score_marginal_multi" if self.ty == CD.GGES_MARG else None

        time_st = time.perf_counter()
        ges_obj = ges(X, ges_score)
        # -------
        # ges_obj['G']: learned causal graph, where ges_obj['G'].graph[j,i]=1 and ges_obj['G'].graph[i,j]=-1 indicates  i --> j ,
        #            ges_obj['G'].graph[i,j] = ges_obj['G'].graph[j,i] = -1 indicates i --- j.

        gg: GeneralGraph = ges_obj['G']
        self.lmg = general_graph_to_lmg(gg)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.model = ges_obj


class SortingMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from src.baselines.CausalDisco.baselines import (
            r2_sort_regress, var_sort_regress, random_sort_regress
        )
        fun = r2_sort_regress if self.ty == CD.R2SORT else \
            var_sort_regress if self.ty == CD.VARSORT else \
                random_sort_regress if self.ty == CD.RANDSORT else None
        time_st = time.perf_counter()
        self.dag = nx.from_numpy_array(fun(X), create_using=nx.DiGraph)
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.metrics = {'time': time.perf_counter() - time_st}


class FCIMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.PAG

    def fit(self, X, **kwargs):
        from causallearn.search.ConstraintBased.FCI import fci
        indep_test = 'fisherz' if self.ty == CD.FCI_PC else 'kci'
        time_st = time.perf_counter()
        graph, edges = fci(X, independence_test_method=indep_test)
        # from causallearn.search.ConstraintBased.FCI:
        #     graph : a GeneralGraph object, where graph.graph[j,i]=1 and graph.graph[i,j]=-1 indicates  i --> j ,
        #                     graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j,
        #                     graph.graph[i,j] = graph.graph[j,i] = 1 indicates i <-> j,
        #                     graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
        # currently don't evaluate the following edges that could point to latent mixing variables: graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
        gg = graph
        self.pag = general_graph_to_lmg(gg, add_bidirected=True)
        self.lmg = general_graph_to_lmg(gg, add_bidirected=False) # this method returns bidir edges as well.
        self.metrics = {'time': time.perf_counter() - time_st}
        self.model = graph, edges



class JCIPCMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty():
        return DAGType.PAG

    def fit(self, X, **kwargs):
        from causallearn.search.ConstraintBased.PC import pc as cl_pc
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        from causallearn.graph.GraphNode import GraphNode

        alpha = float(kwargs.get("alpha", 0.05))
        indep_test =  'fisherz' if self.ty.value == CD.JCI_PC_PC.value else 'kci'
        print(indep_test)
        stable = bool(kwargs.get("stable", True))
        verbose = bool(kwargs.get("verbose", False))
        show_progress = bool(kwargs.get("show_progress", False))

        contexts = sorted(X.keys())
        blocks = [np.asarray(X[k]) for k in contexts]
        data = np.vstack(blocks)
        N, p = data.shape
        K = len(contexts)

        oh = np.zeros((N, K), dtype=float)
        r = 0
        for idx, b in enumerate(blocks):
            n_k = len(b)
            oh[r:r + n_k, idx] = 1.0
            r += n_k

        data_aug = np.hstack([data, oh])
        q = K
        node_names = [f"X{i}" for i in range(p)] + [f"C{j}" for j in range(q)]

        bk = BackgroundKnowledge()
        nodes_bk = [GraphNode(nm) for nm in node_names]
        for j in range(q):
            bk.add_node_to_tier(nodes_bk[p + j], 0)
        for i in range(p):
            bk.add_node_to_tier(nodes_bk[i], 1)
        bk.add_forbidden_by_pattern(r"^X\d+$", r"^C\d+$")
        bk.add_forbidden_by_pattern(r"^C\d+$", r"^C\d+$")

        t0 = time.perf_counter()
        cg = cl_pc(
            data_aug,
            alpha=alpha,
            indep_test=indep_test,
            stable=stable,
            background_knowledge=bk,
            verbose=verbose,
            show_progress=show_progress,
            node_names=node_names,
        )
        M = np.asarray(cg.G.graph)

        ctx_idx = list(range(p, p + q))
        targets_per_context = []
        targets_set = set()
        for cpi in ctx_idx:
            js = [j for j in range(p) if M[j, cpi] == 1]
            targets_per_context.append(js)
            targets_set.update(js)
        self.targets = sorted(targets_set)
        self.targets_per_context = targets_per_context


        self.pag = _cpdag_from_pc_fci_M_over_system(M, p, add_bidirected=True)
        self.lmg = _cpdag_from_pc_fci_M_over_system(M, p, add_bidirected=False)

        has_cy, lmg_acy = remove_cycles(self.lmg)
        if has_cy: self.lmg = nxdigraph_to_lmg(lmg_acy)


        self.metrics = {'time': time.perf_counter() - t0}
        self.model = {
            "adj": M,
            "node_names": node_names,
            "system_p": p,
            "context_count": q,
            "indep_test": indep_test,
            "alpha": alpha,
            "stable": stable,
        }



class JCIFCIMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.PAG   # optionally, DAGType.PAG.

    def fit(self, X, **kwargs):
        from causallearn.search.ConstraintBased.FCI import fci as cl_fci
        alpha = float(kwargs.get("alpha", 0.05))
        depth = int(kwargs.get("depth", -1))
        max_path_length = int(kwargs.get("max_path_length", -1))
        verbose = bool(kwargs.get("verbose", False))
        indep_test =  'fisherz' if self.ty.value == CD.JCI_FCI_PC.value else 'kci'

        print(indep_test)
        contexts = sorted(X.keys())
        blocks = [np.asarray(X[k]) for k in contexts]
        data = np.vstack(blocks)  # (N, p)
        N, p = data.shape
        K = len(contexts)

        # one-hot contexts
        oh = np.zeros((N, K), dtype=float)
        r = 0
        for idx, b in enumerate(blocks):
            n_k = len(b)
            oh[r:r + n_k, idx] = 1.0
            r += n_k

        data_aug = np.hstack([data, oh])  # (N, p+K)
        q = K  # number of context columns kept
        node_names = [f"X{i}" for i in range(p)] + [f"C{j}" for j in range(q)]

        bk = build_bk_jci123(node_names, p, q)
        # run FCI
        time_st = time.perf_counter()
        graph, edges = cl_fci(
            dataset=data_aug,
            independence_test_method=indep_test,
            alpha=alpha,
            depth=depth,
            max_path_length=max_path_length,
            verbose=verbose,
            background_knowledge=bk,
            show_progress=False,
            node_names=node_names
        )

        M = np.asarray(graph.graph)
        ctx_py_indices = list(range(p, p + q))
        targets_per_context = []
        targets_set = set()
        for cpi in ctx_py_indices:
            js = [j for j in range(p) if M[j, cpi] == 1]
            targets_per_context.append(js)
            targets_set.update(js)
        self.targets = sorted(targets_set)
        self.targets_per_context = targets_per_context


        self.pag = _cpdag_from_pc_fci_M_over_system(M, p, add_bidirected=True)
        self.lmg = _cpdag_from_pc_fci_M_over_system(M, p, add_bidirected=False)

        has_cy, lmg_acy = remove_cycles(self.lmg)
        if has_cy: self.lmg = nxdigraph_to_lmg(lmg_acy)

        self.metrics = {'time': time.perf_counter() - time_st}
        self.model = {"adj": M,"node_names": node_names,"system_p": p,
            "context_count": q,"indep_test": indep_test}

# helpers
def build_bk_jci123(node_names, p, q):
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.graph.GraphNode import GraphNode
    bk = BackgroundKnowledge()
    nodes = [GraphNode(nm) for nm in node_names]

    # JCI-1: forbid system -> context
    for j in range(q): bk.add_node_to_tier(nodes[p + j], 0)  # contexts
    for i in range(p): bk.add_node_to_tier(nodes[i], 1)      # system
    bk.add_forbidden_by_pattern(r"^X\d+$", r"^C\d+$")

    # JCI-3 (directed part only): forbid directed context->context
    bk.add_forbidden_by_pattern(r"^C\d+$", r"^C\d+$")

    return bk


def _cpdag_from_pc_fci_M_over_system(M, p, add_bidirected=False):
    G = LabelledMixedGraph(nodes=set(range(p)))
    for i in range(p):
        for j in range(i + 1, p):
            a, b = M[i, j], M[j, i]
            if b == 1 and a == -1:
                G.add_directed(i, j)
                continue
            if a == 1 and b == -1:
                G.add_directed(j, i)
                continue
            if a == -1 and b == -1:
                G.add_undirected(i, j)
                continue
            if not add_bidirected:
                continue
            if a == 1 and b == 1:
                G.add_bidirected(i, j)
                continue
            if b == 1 and a == 2:       # i o-> j
                G.add_semidirected(i, j)
                continue
            if a == 1 and b == 2:       # j o-> i
                G.add_semidirected(j, i)
                continue
            if a == 2 and b == 2:       # i o-o j
                G.add_undirected(i, j)
                continue
    return G



class CAMUVMethod(CausalDiscoveryMthd, ABC):
    """ causal discovery toolbox implementation """

    @staticmethod
    def dag_ty():
        return DAGType.DAG

    def fit(self, X, **kwargs):
        num_explanatory_vals = kwargs.get("num_explanatory_vals", 3)
        alpha = kwargs.get("alpha", 0.05)
        print("CAM-UV: Setting num_explanatory_vals to: ", num_explanatory_vals, ", alpha: ", alpha)

        from causallearn.search.FCMBased.lingam import CAMUV

        time_st = time.perf_counter()

        # Usage
        # P: P[i] contains the indices of the parents of Xi
        # U: The indices of variable pairs having UCPs or UBPs

        P, U = CAMUV.execute(X, alpha, num_explanatory_vals)
        self.metrics = {'time': time.perf_counter() - time_st}

        dag = nx.DiGraph()
        dag.add_nodes_from(set(range(len(P))))
        for i, result in enumerate(P):
            if not len(result) == 0:
                print("child: " + str(i) + ",  parents: " + str(result))
                for j in result:
                    dag.add_edge(j, i)
        print("CAM-UV: evaluate indices U")
        self.dag = dag
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = P, U


class GLOBEMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from src.baselines.globe import GlobeWrapper

        max_interactions = kwargs.get("max_interactions", 3)
        print("Setting max interactions to: ", max_interactions)

        model = GlobeWrapper(max_interactions, False, True)
        data = pd.DataFrame(X)
        data.to_csv("temp.csv", header=False, index=False)
        model.loadData("temp.csv")
        time_st = time.perf_counter()
        adjacency = model.run()
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = model


class ICALINGAMMethod(CausalDiscoveryMthd, ABC):
    """causallearn implementation"""

    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from causallearn.search.FCMBased import lingam

        model = lingam.ICALiNGAM()
        time_st = time.perf_counter()
        model.fit(X)
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = nx.from_numpy_array(model.adjacency_matrix_, create_using=nx.DiGraph)
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = model


class DirectLINGAMMethod(CausalDiscoveryMthd, ABC):
    """causallearn implementation"""

    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from causallearn.search.FCMBased import lingam

        model = lingam.DirectLiNGAM()  # random_state, prior_knowledge, apply_prior_knowledge_softly, measure)
        time_st = time.perf_counter()
        model.fit(X)
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = nx.from_numpy_array(model.adjacency_matrix_, create_using=nx.DiGraph)
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = model


class LINGAMMethod(CausalDiscoveryMthd, ABC):
    """cdt implementation"""

    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from cdt.causality.graph import LiNGAM
        model = LiNGAM()
        time_st = time.perf_counter()
        dag = model.predict(pd.DataFrame(X))
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = dag
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = model
