import logging

from src.exp.exp_change.algos import CD, OracleType
from src.exp.exp_change.config_mix import ExpType
from src.exp.exp_change.gen.generate import GSType
from src.causalchange.mixing.mixing import MixingType

""" parameters relevant to the experiment runs are specified here"""
def get_exp_parameters(exp_type: ExpType, only_base_params=False):
    """  parameter configs to iterate over """
    # note: at pos. 0 of each list is the "base configuration" held fixed in the experiments

    # %% CAUSAL DISCOVERY
    defaults_cd = {}
    defaults_cd["N"] = [6, 4, 2, 10]
    defaults_cd["C"] = [4, 2, 1, 6, 10, 15 ]
    defaults_cd["K"] = [3]
    defaults_cd["S"] = [200, 50, 100]
    defaults_cd["P"] = [0.4, 0.2, 0.6]
    defaults_cd["PC"] = [0.4, 0.2, 0.6, 0.8, 0, 1]

    if only_base_params:
        for ky in defaults_cd: defaults_cd[ky] = [defaults_cd[ky][0]]
    assert all([isinstance(obj, list) for obj in defaults_cd.values()])
    return defaults_cd


def get_cd_algos():
    """ get causal discovery algorithms of interest """
    cd_algos = [
      CD.TopicContextsRFF,
      CD.TopicContextsGP,
      CD.JCI_FCI_PC,
      CD.JCI_FCI_KCI,
      CD.CDNOD_KCI,
      CD.CDNOD_PC,
      CD.UTIGSP
    ]
    cd_algos_ours = cd_algos.copy()
    return cd_algos_ours


def exp_base_parameters(exp_type: ExpType):
    """ parameter config held fixed in each experiment """
    defaults = get_exp_parameters(exp_type)
    return {ky: val[0] for ky, val in defaults.items()}


class OptionsContext:
    exp_type: ExpType
    fixed: dict
    logger: logging.Logger
    verbosity: int
    inclges: bool
    onlybase: bool
    KMAX: int
    param_info_for_reference = {
        "N": "num_vars",
        "S": "num_samples",
        "C": "num_contexts",
        "K": "num_mechanisms",
        "P": "p_directed_edge_in_dag",
        "PC": "p_variant",
    }

    def __init__(self, **kwargs):
        _allowed_keys = {
            "methods", "exp_type", "logger", "reps", "seed", "n_jobs", "enable_SID_call", "verbosity", "read_dir",
            "plot_dir", "KMAX",  "onlybase"
        }
        assert all([k in _allowed_keys for k in kwargs])
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in _allowed_keys)

        # Get experiment setup
        self.attrs = get_exp_parameters(kwargs.get("exp_type", ExpType.CAUSAL_DISCOVERY), self.onlybase)
        self.methods = kwargs.get("methods", get_cd_algos())

    def get_cases(self):
        self.fixed = {attr: val[0] for (attr, val) in self.attrs.items()}

        combos = [
            ({nm: (self.attrs[nm][i] if nm == fixed_nm else self.fixed[nm]) for nm in self.attrs})
            for fixed_nm in self.fixed
            for i in range(len(self.attrs[fixed_nm]))
        ]
        # Keep one attribute fixed and get all combos of the others
        test_cases = {"_".join(f"{arg}_{val}" for arg, val in combo.items()): combo for combo in combos}
        # long runs last
        test_cases = dict(sorted(test_cases.items(), key=lambda dic: (dic[1]["N"], dic[1]["S"], dic[1]["C"])))
        return test_cases

    @staticmethod
    def get_all_experiments():
        """ other possible experiments for reference """
        IVT = ['coef' ]#not used here
        NS = ['mix']#not used here
        IVM = ['context' ]
        GEN = ['mix'] #not used here
        DAG = ['erdos_renyi', 'scale_free', 'random']
        return [
            {"F": gen, "NS": ns, "DG": dg, "IVT": ivt, "IVM": ivm}
            for gen in GEN for ns in NS for dg in DAG for ivt in IVT for ivm in IVM
        ]

    def get_experiments(self):
        IVT = ['coef' ]#not used here
        IVM = ['context'] # 'iid'
        NS = ['mix'] #not used here
        GEN = ['mix'] #not used here
        DAG = ['erdos_renyi']
        GS =   [GSType.GRAPH] if self.exp_type in [ ExpType.CAUSAL_DISCOVERY ] else []   #[i for i in list(GSType) if
             # i != GSType.GRAPH and i.is_bivariate() and not i.is_confounded()] if self.exp_type == ExpType.GRAPHSTRUCTURES else \

        return [
            {"F": gen, "NS": ns, "DG": dg, "IVT": ivt, "IVM": ivm, "GS": gs}
            for gen in GEN for ns in NS for dg in DAG for ivt in IVT for ivm in IVM for gs in GS
        ]

    def get_oracles(self):
        """ no oracles needed rn """
        ORACLES = [ OracleType.hatGhatZ ]
        return ORACLES

    def get_mixing_algos(self):
        """ no unknown mixing """
        mixing_algos = [ MixingType.SKIP ]
        return mixing_algos

    def get_base_attribute_idf(self):
        return '_'.join([f'{ky}_{vl}' for ky, vl in self.fixed.items()])
