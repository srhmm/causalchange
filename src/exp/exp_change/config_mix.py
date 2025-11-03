import logging
from enum import Enum

import numpy as np

from src.exp.exp_change.algos import CD, OracleType
from src.exp.exp_change.gen.generate import GSType
from src.causalchange.mixing.mixing import MixingType

""" parameters relevant to the experiment runs are specified here"""


class ExpType(Enum):
    """ our main experiment types """
    CLUSTERING = 0
    POWER_SPECIFICITY = 1
    CAUSAL_DISCOVERY = 2
    CLUSTERING_GRAPHSTRUCTURES = 3
    INTERVENTIONAL_MIXTURE = 4

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return ["MIX", "CD-PS", "CD", "MIX-GS", "CD-IV"][self.value]

    def long_nm(self): return ["mixing structure (Fig 4)", "ablation study on pruning spurious edges (Fig D1)",
                               "causal discovery (Fig 5)", "graph structures", "interventional mixture (Fig 6)"][self.value]

def exp_base_parameters(exp_type: ExpType):
    """ parameter config held fixed in each experiment """
    defaults = exp_default_parameters(exp_type)
    return {ky: val[0] for ky, val in defaults.items()}


def exp_default_parameters(exp_type: ExpType, only_base_params=False):
    """  parameter configs to iterate over """
    defaults = {"N": [10], "S": [500], "P": [0.4], "C": [5], "K": [2], "PZ": [0.5], "NZ": [4], "L": [2]}
    # note: "N", "PZ", "NZ" are used if exp["GS"]: GSType is a graph, otherwise they depend on the graph structure (bivariate etc.).
    # note: at pos. 0 of each list is the "base configuration" held fixed in the experiments

    # %% A: CLUSTERING
    defaults_clustering = defaults.copy()
    defaults_clustering["PZ"] = list(np.linspace(0.0, 1.0, 10))
    defaults_clustering["PZ"].insert(0, 0.4),  # at pos 0 is base config
    defaults_clustering["NZ"] = list(range(10))
    defaults_clustering["NZ"].insert(0, 2)  # at pos 0 is base config
    defaults_clustering["N"] = [10, 5, 20]
    defaults_clustering["S"] = [1000, 100, 500]
    defaults_clustering["C"] = [5, 2, 7, 10]
    defaults_clustering["K"] = [2, 3, 4, 5, 1]

    # %% B: CAUSAL DISCOVERY
    # Atomic interventional setting: experiment parameters used in MixtureUTIGSP: "Learning Mixtures of Interventions", Kumar et al. 24.
    defaults_cd_iv = defaults.copy()
    defaults_cd_iv["N"] = [6, 4, 8]
    defaults_cd_iv["S"] = [1000, 100, 200, 500]  # , 10000] # skip huge datasets
    defaults_cd_iv["NZ"] = [1]  # always one latent Z ("environment", split into atomic interventions)
    defaults_cd_iv["PZ"] = [0.5, 1]  # [1, 0.5]  # #make sure all or half intervened as in mix-utigsp experiment setup

    # More general setting: consider a range of parameters
    defaults_cd = defaults.copy()
    defaults_cd["N"] = [10, 3, 6]
    defaults_cd["NZ"] = [4, 2, 3, 5]
    defaults_cd["PZ"] = [0.5, 0, 0.25, 0.75, 1]
    defaults_cd["P"] = [0.4, 0, 0.2, 0.6, 0.8, 1]
    defaults_cd["S"] = [500, 100, 200, 1000]

    # ablation study: effect of the latents is of interest
    defaults_ps = defaults.copy()
    defaults_ps["NZ"] = [2]
    defaults_ps["PZ"] = [0.5, 0, 0.25, 0.75, 1]

    defaults_clustering_gs = defaults.copy()
    defaults_ret = defaults_clustering if exp_type == ExpType.CLUSTERING else \
        defaults_ps if exp_type == ExpType.POWER_SPECIFICITY else \
            defaults_cd if exp_type == ExpType.CAUSAL_DISCOVERY else \
                defaults_clustering_gs if exp_type == ExpType.CLUSTERING_GRAPHSTRUCTURES \
                    else defaults_cd_iv if exp_type == ExpType.INTERVENTIONAL_MIXTURE else None
    assert defaults_ret is not None
    if only_base_params:
        for ky in defaults_ret: defaults_ret[ky] = [defaults_ret[ky][0]]
    assert all([isinstance(obj, list) for obj in defaults_ret.values()])
    return defaults_ret


class Options:
    exp_type: ExpType
    fixed: dict
    logger: logging.Logger
    verbosity: int
    inclges: bool
    onlybase: bool
    KMAX: int
    param_info_for_reference = {
        "N": "num_vars", "NZ": "num_mixing_vars", "S": "num_samples", "K": "num_mixing_clusters",
        "P": "p_directed_edge_in_dag", "PZ": "p_observed_var_is_mixed", "C": "frac_samples_with_mixing",
        "L": "hypparam_score"
    }

    def __init__(self, **kwargs):
        _allowed_keys = {
            "methods", "exp_type", "logger", "reps", "seed", "n_jobs", "enable_SID_call", "verbosity", "read_dir",
            "plot_dir", "KMAX", "inclges", "onlybase"
        }
        assert all([k in _allowed_keys for k in kwargs])
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in _allowed_keys)

        # Get experiment setup
        self.attrs = exp_default_parameters(kwargs.get("exp_type", ExpType.CLUSTERING), self.onlybase)
        self.methods = kwargs.get("methods", self.get_cd_algos())

    def get_cases(self):
        self.fixed = {attr: val[0] for (attr, val) in self.attrs.items()}

        combos = [
            ({nm: (self.attrs[nm][i] if nm == fixed_nm else self.fixed[nm]) for nm in self.attrs})
            for fixed_nm in self.fixed
            for i in range(len(self.attrs[fixed_nm]))
        ]
        # Keep one attribute fixed and get all combos of the others
        test_cases = {"_".join(f"{arg}_{val}" for arg, val in combo.items()): combo for combo in combos}
        # small runs first
        test_cases = dict(sorted(test_cases.items(), key=lambda dic: (dic[1]["N"], dic[1]["S"])))
        return test_cases

    @staticmethod
    def get_all_experiments():
        """ other possible experiments for reference """
        IVT = ['flip', 'shift']
        IVM = ['mixing', 'sergio']
        NS = ['normal', 'unif', 'exp', 'gumbel']
        GEN = ['lin', 'quad', 'cub', 'exp', 'log', 'sin', 'mix']
        DAG = ['erdos_renyi', 'scale_free', 'random']
        return [
            {"F": gen, "NS": ns, "DG": dg, "IVT": ivt, "IVM": ivm}
            for gen in GEN for ns in NS for dg in DAG for ivt in IVT for ivm in IVM
        ]

    def get_experiments(self):
        IVT = ['flip']
        IVM = ['mixing']
        NS = ['normal']
        GEN =  [ 'cub', 'lin',  'quad', 'sin', 'exp', 'log' ]
        DAG = ['erdos_renyi']
        GS = [i for i in list(GSType) if
              i != GSType.GRAPH and i.is_bivariate() and not i.is_confounded()] if self.exp_type == ExpType.CLUSTERING_GRAPHSTRUCTURES else \
            [GSType.GRAPH] if self.exp_type in [ExpType.CLUSTERING, ExpType.CAUSAL_DISCOVERY,
                                                ExpType.INTERVENTIONAL_MIXTURE,
                                                ExpType.POWER_SPECIFICITY] else []  # list(GSType)
        return [
            {"F": gen, "NS": ns, "DG": dg, "IVT": ivt, "IVM": ivm, "GS": gs}
            for gen in GEN for ns in NS for dg in DAG for ivt in IVT for ivm in IVM for gs in GS
        ]

    def get_oracles(self):
        """ get oracles for the experiment (known vs. unknown causal graph)"""
        # ORACLES = list(OracleType) # all
        ORACLES = [OracleType.trueGhatZ] if self.exp_type == ExpType.CLUSTERING_GRAPHSTRUCTURES else \
            [OracleType.trueGhatZ, OracleType.hatGhatZ
             ] if self.exp_type == ExpType.CLUSTERING else   \
            [OracleType.trueGtrueZ, OracleType.hatGhatZ
             ] if self.exp_type == ExpType.POWER_SPECIFICITY else \
                [ OracleType.hatGhatZ ]

        return ORACLES

    def get_mixing_algos(self):
        """ get clustering method used locally for each causal edge (cmm vs. gmm) """
        # MMs = list(MixingType) # all
        mixing_algos = [
            MixingType.MIX_LIN, #MixingType.MIX_QUAD, MixingType.MIX_CUB,
            MixingType.MIX_NS, #MixingType.MIX_BS,
            MixingType.BASE_RANDOM_SPLIT, MixingType.BASE_GMM_GLOB, MixingType.BASE_GMM,
            MixingType.BASE_DBSCAN, MixingType.BASE_SPECTRAL, MixingType.BASE_HDBSCAN, MixingType.BASE_KMEANS
        ]
        return mixing_algos if self.exp_type in [ExpType.CLUSTERING_GRAPHSTRUCTURES, ExpType.CLUSTERING] else \
            [MixingType.MIX_LIN, MixingType.MIX_QUAD, MixingType.MIX_CUB, MixingType.MIX_NS, MixingType.MIX_BS]  # different MMs for graph structures and clustering experiments, otherwise only ours

    #@staticmethod
    def get_cd_algos(self):
        """ get causal discovery algorithms of interest """
        cd_algos = [
            CD.FCI_PC, CD.LINGAM, CD.SCORE,
            CD.CAM, CD.GES, CD.PC_PC,
            CD.R2SORT, CD.RANDSORT, CD.VARSORT
        ]
        cd_algos_ours = cd_algos.copy()
        if self.exp_type == ExpType.INTERVENTIONAL_MIXTURE: cd_algos_ours.insert(0, CD.MixtureUTIGSP)
        if self.inclges: cd_algos_ours.insert(0, CD.CausalMixturesGES)
        cd_algos_ours.insert(0, CD.CausalMixtures)
        return [CD.CausalMixtures] if self.exp_type in [ExpType.CLUSTERING_GRAPHSTRUCTURES, ExpType.CLUSTERING] else \
            cd_algos if self.exp_type == ExpType.POWER_SPECIFICITY else \
                cd_algos_ours if self.exp_type == ExpType.CAUSAL_DISCOVERY else \
                    cd_algos_ours if self.exp_type == ExpType.INTERVENTIONAL_MIXTURE else []

    def get_base_attribute_idf(self):
        return '_'.join([f'{ky}_{vl}' for ky, vl in self.fixed.items()])
