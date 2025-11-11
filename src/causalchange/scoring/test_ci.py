import numpy as np
from causallearn.utils.cit import FisherZ as _CL_FisherZ
from causallearn.utils.cit import MV_FisherZ as _CL_MV_FisherZ
from causallearn.utils.cit import MC_FisherZ as _CL_MC_FisherZ
from causallearn.utils.cit import KCI as _CL_KCI
from causallearn.utils.cit import Chisq_or_Gsq as _CL_ChiGsq
from causallearn.utils.cit import D_Separation as _CL_DSep

def _pack_data(Y, S, Z):
    Y = np.asarray(Y, float).reshape(-1, 1)
    S = np.asarray(S, float).reshape(-1, 1)
    Z = np.asarray(Z, float) if Z is not None else np.empty((Y.shape[0], 0))
    assert Y.shape[0] == S.shape[0] == Z.shape[0], "Y, S, Z must share the same number of rows"
    return np.hstack([Y, S, Z])

def _idx_sets(d):
    # columns: 0=Y, 1=S, 2..(d+1)=Z
    return (0, 1, [] if d == 0 else list(range(2, 2 + d)))

def test_fun_kci(Y, S, Z, **kwargs):
    """Kernel Conditional Independence: KCI (continuous)"""
    data = _pack_data(Y, S, Z)
    X_idx, Y_idx, cond = _idx_sets(data.shape[1] - 2)
    kci = _CL_KCI(data, **kwargs)  # forwards KCI args: kernelX/kernelY/kernelZ/null_ss/approx/use_gp/est_width/polyd/kwidth*
    return float(kci(X_idx, Y_idx, cond))

def test_fun_fisherz(Y, S, Z, **kwargs):
    """Partial correlation (Fisher-Z)"""
    data = _pack_data(Y, S, Z)
    X_idx, Y_idx, cond = _idx_sets(data.shape[1] - 2)
    tz = _CL_FisherZ(data, **kwargs)
    return float(tz(X_idx, Y_idx, cond))

def test_fun_mv_fisherz(Y, S, Z, **kwargs):
    """Fisher-Z with test-wise deletion (handles missing values)"""
    data = _pack_data(Y, S, Z)
    X_idx, Y_idx, cond = _idx_sets(data.shape[1] - 2)
    tz = _CL_MV_FisherZ(data, **kwargs)
    return float(tz(X_idx, Y_idx, cond))

def test_fun_mc_fisherz(Y, S, Z, **kwargs):
    """Fisher-Z with missingness correction (needs `skel` and `prt_m`; otherwise falls back to MV_FisherZ)."""
    data = _pack_data(Y, S, Z)
    X_idx, Y_idx, cond = _idx_sets(data.shape[1] - 2)
    skel = kwargs.pop("skel", None)
    prt_m = kwargs.pop("prt_m", None)
    if skel is None or prt_m is None:
        tz = _CL_MV_FisherZ(data, **kwargs)
        return float(tz(X_idx, Y_idx, cond))
    tz = _CL_MC_FisherZ(data, **kwargs)
    return float(tz(X_idx, Y_idx, cond, skel, prt_m))

def test_fun_chisq(Y, S, Z, **kwargs):
    """Chi-square test (categorical; data must be integer-coded)."""
    data = _pack_data(Y, S, Z).astype(np.int64, copy=False)
    X_idx, Y_idx, cond = _idx_sets(data.shape[1] - 2)
    ct = _CL_ChiGsq(data, method_name="chisq", **kwargs)
    return float(ct(X_idx, Y_idx, cond))

def test_fun_gsq(Y, S, Z, **kwargs):
    """G-square test (categorical; data must be integer-coded)."""
    data = _pack_data(Y, S, Z).astype(np.int64, copy=False)
    X_idx, Y_idx, cond = _idx_sets(data.shape[1] - 2)
    ct = _CL_ChiGsq(data, method_name="gsq", **kwargs)
    return float(ct(X_idx, Y_idx, cond))

def test_fun_dseparation(Y, S, Z, **kwargs):
    """d-separation oracle (needs `true_dag`)."""
    data = _pack_data(Y, S, Z)
    X_idx, Y_idx, cond = _idx_sets(data.shape[1] - 2)
    true_dag = kwargs.get("true_dag", None)
    ds = _CL_DSep(data, true_dag=true_dag)
    return float(ds(X_idx, Y_idx, cond))

# Optional: a registry for convenience
TEST_FUN_REGISTRY = {
    "kci": test_fun_kci,
    "fisherz": test_fun_fisherz,
    "mv_fisherz": test_fun_mv_fisherz,
    "mc_fisherz": test_fun_mc_fisherz,
    "chisq": test_fun_chisq,
    "gsq": test_fun_gsq,
    "d_separation": test_fun_dseparation,
}
