from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import networkx as nx

from src.causalchange.gen.generate import gen_data_type, GenDataType, DagType, GSType
from src.causalchange.gen.synthetic.gen_types import FunType, NoiseType, IvType


@dataclass(frozen=True)
class SynthSpec:
    n_nodes: int = 5
    edge_prob: float = 0.4
    fun_type: FunType = FunType.MIX
    noise_type: NoiseType = NoiseType.GAUSS
    iv_type: IvType = IvType.MIX
    dag_type: DagType = DagType.ERDOS
    gs_type: GSType = GSType.GRAPH


    pc: float = 1.0
    kmn: int = 1
    kmx: int = 3


def _base_params(spec: SynthSpec, S: int, C: int, ivm: GenDataType) -> dict:
    return {
        "N": int(spec.n_nodes),
        "S": int(S),
        "P": float(spec.edge_prob),
        "C": int(C),
        "PC": float(spec.pc),
        "Kmn": int(spec.kmn),
        "Kmx": int(spec.kmx),
        "IVM": ivm,
        "IVT": spec.iv_type,
        "GS": spec.gs_type,
        "DG": spec.dag_type,
        "F": spec.fun_type,
        "NS": spec.noise_type,
    }


def sample_iid(spec: SynthSpec, n_samples: int, seed: int) -> tuple[np.ndarray, nx.DiGraph]:
    params = _base_params(spec, S=n_samples, C=1, ivm=GenDataType.IID)
    X, truths = gen_data_type(params, seed)
    true_g = truths["true_g"]
    return X, true_g


def sample_contexts(
    spec: SynthSpec,
    n_contexts: int,
    n_samples_per_context: int,
    seed: int,
) -> tuple[Dict[int, np.ndarray], nx.DiGraph]:
    S = n_contexts * n_samples_per_context
    params = _base_params(spec, S=S, C=n_contexts, ivm=GenDataType.MULTI_CONTEXT)
    X_ctxs, truths = gen_data_type(params, seed)
    true_g = truths["true_g"]
    return X_ctxs, true_g


def sample_mixed(
    spec: SynthSpec,
    n_samples: int,
    seed: int,
    *,
    n_confounders: int = 1,
    frac_confounded: float = 0.4,
    n_classes: int = 5,
    K: int = 2,
) -> tuple[np.ndarray, nx.DiGraph]:
    params = _base_params(spec, S=n_samples, C=n_classes, ivm=GenDataType.MIXING)

    params["NZ"] = int(n_confounders)
    params["PZ"] = float(frac_confounded)
    params["K"] = int(K)

    X, truths = gen_data_type(params, seed)
    true_g = truths["true_g"]
    return X, true_g


def sample_time_series(*args, **kwargs):
    raise NotImplementedError(
        "TIME generation is not implemented via src.causalchange.gen.generate.gen_data_type "
        "(at least in the current generator API)."
    )


def sample_time_series_contexts(*args, **kwargs):
    raise NotImplementedError(
        "TIME_CONTEXTS generation is not implemented via src.causalchange.gen.generate.gen_data_type "
        "(at least in the current generator API)."
    )
