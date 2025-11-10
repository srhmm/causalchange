from enum import Enum

from pygam import GAM
from pygam import LinearGAM
from scipy.interpolate import make_lsq_spline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.causalchange.scoring.fit_time_space import fit_gaussian_process
from src.causalchange.util.utils import data_scale
from src.causalchange.util.sttypes import TimeseriesScoringFunction


class GPType(Enum):
    EXACT = 'gp'
    FOURIER = 'ff'
    def __eq__(self, other):
        return self.value == other.value


class MIType(Enum):
    TC = 'tc'
    MSS = 'mss'

    def __eq__(self, other):
        return self.value == other.value

class ScoreType(Enum):
    LIN = 'lin'
    GAM = 'gam'
    SPLINE = 'spline'
    GP = GPType
    MI = MIType

    def get_model(self, **kwargs):
        raise ValueError(f"Only valid for ScoreType.GP, not {self.value}")

    def is_mi(self):
        return self.value in MIType._member_map_
    def __eq__(self, other):
        return self.value == other.value

class DataMode(Enum):
    IID = 'iid'
    CONTEXTS = 'contexts'
    TIME = 'time'
    TIME_CONTEXTS = 'time-contexts'
    CONFOUNDED = 'confounded'
    MIXED = 'mixed'

    def __eq__(self, other):
        return self.value == other.value
class GraphSearch(Enum):
    TOPIC = 'topological'
    GLOBE = 'edge-greedy'

    def __eq__(self, other):
        return self.value == other.value


def fit_functional_model_context(
        X, pa, target,
    fun_gp,       # callable: (Xtr, ytr, params_score_type_dict) -> (model, score_bits)
    alpha=0.05,   # significance thresh for no-hypercompression
    vb=0
):
    """
    Connected-components partitioning using only per-context and pairwise joint GP MDL scores.
    Clustering step uses pairwise Δ only (no 3+ context fits). Final scoring refits exactly
    one GP per discovered group (optional but cheap).

    Returns:
      {
        'partition': dict context->cluster_id,
        'groups': list of lists of contexts,
        'labels_pred': np.ndarray shape (C,),
        'contexts': list of context ids,
        'models_per_context': dict c -> model,
        'scores_per_context': dict c -> L_c (bits),
        'pairwise_delta': 2D np.array (C x C),
        'group_models': list of (model, L_group_bits) aligned with 'groups',
        'total_score_bits': float,
        'k_bits': float,
        'edges': list of (c,d) used to connect components
      }
    """
    import numpy as np
    import networkx as nx
    from functools import lru_cache

    # --- Prepare per-context design/response ---
    X_parents = {c: np.random.normal(size=X[c][:, [target]].shape) for c in range(len(X))} if len(pa) == 0 else {
        c: X[c][:, pa] for c in range(len(X))}
    X_target = {c: X[c][:, target] for c in range(len(X))}

    assert callable(fun_gp), "fun_gp must be a callable: (Xtr, ytr, params_dict) -> (model, score_bits)"

    contexts = sorted(X_parents.keys())
    C = len(contexts)
    idx_of = {c: i for i, c in enumerate(contexts)}

    models_c, Lc = {}, {}
    for c in contexts:
        Xtr = X_parents[c]
        ytr = X_target[c]
        models_c[c], Lc[c] = fun_gp(Xtr, ytr, dict())
    if vb >= 1:
        msg = ", ".join([f"{c}: {Lc[c]:.3f}" for c in contexts])
        print(f"[per-context] L_c bits: {msg}")

    def _concat_ctx(ctx_list):
        Xs = [X_parents[c] for c in ctx_list]
        ys = [X_target[c]  for c in ctx_list]
        return np.vstack(Xs), np.concatenate(ys, axis=0)

    # cache pairwise fits to avoid duplicates
    @lru_cache(maxsize=None)
    def _fit_pair_sorted(c_small, c_big):
        Xtr, ytr = _concat_ctx([c_small, c_big])
        model, L = fun_gp(Xtr, ytr, dict())
        return model, float(L)

    @lru_cache(maxsize=None)
    def _fit_group_tuple(ctx_tuple_sorted):
        # For pairs, reuse pairwise cache to avoid redundant work
        if len(ctx_tuple_sorted) == 2:
            c_small, c_big = ctx_tuple_sorted
            return _fit_pair_sorted(c_small, c_big)
        Xtr, ytr = _concat_ctx(list(ctx_tuple_sorted))
        model, L = fun_gp(Xtr, ytr, dict())
        return model, float(L)

    #pairw joint fits and delta
    Delta = np.zeros((C, C), dtype=float)  # Δ_{c,d} = L_{c∪d} - (L_c + L_d)
    for i, c in enumerate(contexts):
        for j, d in enumerate(contexts):
            if j <= i:
                continue
            _, L_cd = _fit_pair_sorted(*sorted((c, d)))
            Delta[i, j] = L_cd - (Lc[c] + Lc[d])
            Delta[j, i] = Delta[i, j]
    if vb >= 2:
        print(f"[pairwise] Δ stats: min={Delta.min():.3f}, max={Delta.max():.3f}, mean={Delta.mean():.3f}")

    # Connected components from significant negative delta
    k_bits = -np.log2(alpha) if alpha is not None else 0.0
    edges = []
    for i in range(C):
        for j in range(i+1, C):
            if Delta[i, j] <= -k_bits:
                edges.append((contexts[i], contexts[j]))

    G = nx.Graph()
    G.add_nodes_from(contexts)
    G.add_edges_from(edges)
    comps = list(nx.connected_components(G))
    groups = [sorted(list(comp)) for comp in comps]
    part = {c: gid for gid, grp in enumerate(groups) for c in grp}
    if vb >= 1:
        print(f"[cc] alpha={alpha} -> k={k_bits:.3f} bits | edges={len(edges)} | groups={groups}")

    # final score over group
    group_models = []
    total_score_bits = 0.0
    for Gc in groups:
        ct = tuple(sorted(Gc))
        if len(ct) == 1:
            c = ct[0]
            model, Lg = models_c[c], Lc[c]
        elif len(ct) == 2:
            # use cached pairwise refit
            model, Lg = _fit_pair_sorted(*ct)
        else:
            # one-time multiway refit for final scoring only
            model, Lg = _fit_group_tuple(ct)
        group_models.append((model, Lg))
        total_score_bits += Lg

    labels_pred = np.array([part[c] for c in contexts], dtype=int)

    results = {
        "partition": part, # dict: context -> group id
        "groups": groups, # list of lists of contexts
        "labels_pred": labels_pred,
        "contexts": contexts,
        "models_per_context": models_c,
        "scores_per_context": Lc,
        "pairwise_delta": Delta,
        "group_models": group_models,
        "total_score_bits": float(total_score_bits),
        "k_bits": float(k_bits),
        "edges": edges,
    }

    if vb >= 1:
        print(f"[final] target {target} groups={groups} | total_score_bits={float(total_score_bits):.2f}")
    return float(total_score_bits), results

def fit_functional_model_context_greedy_group_merges(
        X, pa, target,
    fun_gp,       # callable: (Xtr, ytr, params_score_type) -> (model, score_bits)
    alpha=0.05,              # significance thresh for no-hypercompression
    vb=0
):
    """
    Returns:
      {
        'partition': dict context->cluster_id,
        'groups': list of lists of contexts,
        'models_per_context': dict c -> model,
        'scores_per_context': dict c -> L_c (bits),
        'pairwise_delta': 2D np.array (C x C),
        'group_models': list of (model, L_group_bits) aligned with 'groups',
        'total_score_bits': float
      }
    """
    import numpy as np
    X_parents = {c: np.random.normal(size=X[c][:, [target]].shape) for c in range(len(X))} if len(pa) == 0 else {
        c: X[c][:, pa] for c in range(len(X))}
    X_target = {c: X[c][:, target] for c in range(len(X))}

    assert callable(fun_gp), "fit_score_gp or _rff"

    contexts = sorted(X_parents.keys())
    C = len(contexts)

    # per-context fits
    models_c, Lc = {}, {}
    for c in contexts:
        Xtr = X_parents[c]
        ytr = X_target[c]
        models_c[c], Lc[c] = fun_gp(Xtr, ytr, dict()) #, score_type=GPType.EXACT) #params_score_type)

    from functools import lru_cache
    import numpy as np

    def _concat_ctx(ctx_list):
        Xs = [X_parents[c] for c in ctx_list]
        ys = [X_target[c]  for c in ctx_list]
        return np.vstack(Xs), np.concatenate(ys, axis=0)

    @lru_cache(maxsize=None)
    def _fit_group_tuple(ctx_tuple_sorted):
        Xtr, ytr = _concat_ctx(list(ctx_tuple_sorted))
        model, L = fun_gp(Xtr, ytr, dict())#score_type=GPType.EXACT) #params_score_type)
        return model, float(L)

    # pairwise joint fits
    Delta = np.zeros((C, C), dtype=float)  # Δ_{c,d} = L_cd - (L_c + L_d)
    idx_of = {c: i for i, c in enumerate(contexts)}
    for i, c in enumerate(contexts):
        for j, d in enumerate(contexts):
            if j <= i:
                continue
            _, L_cd = _fit_group_tuple(tuple(sorted((c, d))))
            Delta[i, j] = L_cd - (Lc[c] + Lc[d])
            Delta[j, i] = Delta[i, j]

    # greedy bottom-up MDL merging using Δ and k = -log2(alpha)
    k_bits = -np.log2(alpha) if alpha is not None else 0.0

    # Start with singleton groups
    groups = [[c] for c in contexts]
    # Cache group scores
    group_L = {}
    for g in range(len(groups)):
        ct = tuple(sorted(groups[g]))
        # singleton: we already have per-context L
        if len(ct) == 1:
            group_L[g] = Lc[ct[0]]
        else:
            group_L[g] = _fit_group_tuple(ct)[1]

    def total_L():
        return sum(group_L[g] for g in group_L.keys())

    changed = True
    while changed and len(groups) > 1:
        changed = False
        best_merge = None
        best_gain = 0.0  # positive means L decreases
        # consider all pairs of groups
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                A, B = groups[a], groups[b]
                L_A = group_L[a]
                L_B = group_L[b]
                AUB = tuple(sorted(A + B))
                L_AUB = _fit_group_tuple(AUB)[1]
                # MDL improvement if L_AUB <= L_A + L_B - k  (k = min gain to call it significant)
                gain = (L_A + L_B) - L_AUB - k_bits
                if gain > best_gain:
                    best_gain = gain
                    best_merge = (a, b, L_AUB)

        if best_merge is not None:
            a, b, L_ab = best_merge
            # merge b into a
            new_group = sorted(groups[a] + groups[b])
            if vb >= 2:
                print(f"[merge] {groups[a]} + {groups[b]} "
                      f"-> {new_group} | gain={best_gain:.3f} bits (k={k_bits:.3f})")
            groups[a] = new_group
            del groups[b]

            # rebuild group_L map (indexes changed)
            group_L = {}
            for g, G in enumerate(groups):
                ct = tuple(sorted(G))
                if len(ct) == 1:
                    group_L[g] = Lc[ct[0]]
                else:
                    group_L[g] = _fit_group_tuple(ct)[1]

            changed = True

    #final MDL over groups + bookkeeping
    final_total = total_L()
    group_models = []
    for G in groups:
        ct = tuple(sorted(G))
        model, Lg = _fit_group_tuple(ct)
        group_models.append((model, Lg))

    # build partition: context -> cluster id
    part = {}
    for gid, G in enumerate(groups):
        for c in G:
            part[c] = gid

    labels_pred = np.array([part[c] for c in contexts], dtype=int)

    results = {
        "partition": part,  # dict: context -> group id
        "groups": groups,  # list of lists of contexts
        "labels_pred": labels_pred,
        "contexts": contexts,
        "models_per_context": models_c,
        "scores_per_context": Lc,
        "pairwise_delta": Delta,
        "group_models": group_models,
        "total_score_bits": final_total,
        "k_bits": k_bits,
    }

    if vb >= 1:
        print(f"[partition] target {target} groups={groups} | total_score_bits={float(final_total):.2f} | k={k_bits:.2f}")
    return  float(final_total), results


def _standardize_xy(X, y):
    Xm = X.mean(axis=0, keepdims=True); Xs = X.std(axis=0, keepdims=True) + 1e-12
    ym = y.mean(); ys = y.std() + 1e-12
    Xz = (X - Xm) / Xs
    yz = (y - ym) / ys
    return Xz, yz, (Xm, Xs, ym, ys)

def _rff_features(X, W, b, sf2):
    # phi(x) = sqrt(2/D) * sqrt(sf2) * cos(W x + b)
    Z = X @ W.T + b  # (n, D)
    return np.sqrt(2.0 / W.shape[0]) * np.sqrt(sf2) * np.cos(Z)

def fit_score_rff(Xtr, ytr, params):
    """
    Random Fourier Features + ridge with MDL/BIC (bits).
    Params (with sensible defaults):
      - D: int number of features (e.g., 300)
      - restarts: int (e.g., 5)
      - ell_bounds: (low, high) for log_ell (default [-1.5, 1.5])
      - sf2_bounds: (low, high) for log_sf2 (default [-1.0, 1.0])
      - ridge_bounds: (low, high) for log_lambda (default [-6.0, 2.0])
      - seed: int or None
      - bic_penalty: bool (default True)
    Returns:
      model dict (has predict_rff) and score_bits (float)
    """
    X = np.asarray(Xtr, float)
    y = np.asarray(ytr, float).reshape(-1)
    n, p = X.shape

    D = int(params.get("D", 300))
    restarts = int(params.get("restarts", 5))
    ell_low, ell_high = params.get("ell_bounds", (-1.5, 1.5))
    sf2_low, sf2_high = params.get("sf2_bounds", (-1.0, 1.0))
    ridge_low, ridge_high = params.get("ridge_bounds", (-6.0, 2.0))
    bic_penalty = params.get("bic_penalty", True)
    rng = np.random.default_rng(params.get("seed", None))

    # standardize once (keeps search stable)
    Xz, yz, stats = _standardize_xy(X, y)

    def fit_once(log_ell, log_sf2, log_lambda):
        ell = np.exp(log_ell)
        sf2 = np.exp(log_sf2)
        lam = np.exp(log_lambda)

        # sample RFFs
        # W ~ N(0, 1/ell^2 I), b ~ U(0, 2π)
        W = rng.normal(0.0, 1.0 / (ell + 1e-12), size=(D, p))
        b = rng.uniform(0.0, 2*np.pi, size=(D,))

        Phi = _rff_features(Xz, W, b, sf2)  # (n, D)

        # ridge closed-form
        # w = (Phi^T Phi + lam I)^(-1) Phi^T y
        # Use Cholesky on D x D (D usually << n)
        G = Phi.T @ Phi
        A = G + lam * np.eye(D)
        try:
            L = np.linalg.cholesky(A)
            w = np.linalg.solve(L.T, np.linalg.solve(L, Phi.T @ yz))
        except np.linalg.LinAlgError:
            return np.inf, None

        yhat = Phi @ w
        resid = yz - yhat
        RSS = float(resid.T @ resid)
        # MLE noise variance
        sigma2 = max(RSS / n, 1e-12)

        # Negative log-likelihood at sigma2 MLE: (n/2) * (1 + log(2π σ^2))
        nll_nats = 0.5 * n * (1.0 + np.log(2.0 * np.pi * sigma2))

        # MDL/BIC: add (k/2) log n, k ~ D + 1 (weights + noise variance)
        k_params = D + 1
        if bic_penalty and n > 1:
            nll_nats += 0.5 * k_params * np.log(n)

        mdl_bits = nll_nats / np.log(2.0)

        model = {
            "W": W, "b": b, "w": w,
            "log_ell": float(log_ell), "log_sf2": float(log_sf2), "log_lambda": float(log_lambda),
            "stats": stats, "D": D, "sigma2_mle": float(sigma2),
        }
        return mdl_bits, model

    best_bits, best_model = np.inf, None
    # coarse random search
    for _ in range(restarts):
        log_ell = rng.uniform(ell_low, ell_high)
        log_sf2 = rng.uniform(sf2_low, sf2_high)
        log_lambda = rng.uniform(ridge_low, ridge_high)
        bits, model = fit_once(log_ell, log_sf2, log_lambda)
        if bits < best_bits:
            best_bits, best_model = bits, model

    # prediction closure (returns mean; optional variance under linear-Gaussian model)
    def predict(Xte, return_var=False):
        Xte = np.asarray(Xte, float)
        Xm, Xs, ym, ys = best_model["stats"]
        Xtez = (Xte - Xm) / Xs
        Phi_te = _rff_features(Xtez, best_model["W"], best_model["b"], np.exp(best_model["log_sf2"]))
        yhat_z = Phi_te @ best_model["w"]
        yhat = yhat_z * ys + ym
        if not return_var:
            return yhat
        # Predictive variance under fixed w is σ²; under Bayesian linear model you’d add Phi Σw Phi^T
        return yhat, np.full(Xte.shape[0], best_model["sigma2_mle"] * (ys**2))

    best_model["predict"] = predict
    return best_model, float(best_bits)


def fit_functional_model(
        X, pa, target, score_type, **scoring_params):
    r""" fitting and scoring functional models

    :param X: parents
    :param y: target
    :param scoring_params: hyperparameters

    :Keyword Arguments:
    * *score_type* (``ScoreType``) -- regressor and associated information-theoretic score
    """
    params_score_type = score_type
    params_gam_scale = scoring_params.get("gam_scale", False)

    X_pa = np.random.normal(size=X[:, [target]].shape) if len(pa) == 0 else  X[:, pa]
    X_target = X[:, target]

    if params_score_type in GPType._member_map_.values():
        Xtr, ytr = (data_scale(X_pa), data_scale(X_target.reshape(-1, 1))) if params_gam_scale else (X_pa, X_target)
        model, score = fit_score_gp(Xtr, ytr, **scoring_params)

    elif params_score_type == ScoreType.GAM:
        Xtr, ytr = (data_scale(X_pa), data_scale(X_target.reshape(-1, 1))) if params_gam_scale else (X_pa, X_target)
        model, score = fit_score_gam(Xtr, ytr)

    elif params_score_type == ScoreType.SPLINE:
        raise DeprecationWarning()
    else:
        raise ValueError(f"Invalid score {params_score_type}")

    return score, dict(model=model)

# ---------- tiny GP toolbox (RBF + noise, pure NumPy) ----------


import numpy as np

def _standardize(X, y, eps=1e-12):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    Xmu = X.mean(axis=0, keepdims=True)
    Xsd = X.std(axis=0, keepdims=True)
    Xsd = np.where(Xsd < eps, 1.0, Xsd)
    Xn = (X - Xmu) / Xsd
    ymu = y.mean()
    ysd = y.std()
    if ysd < eps:  # degenerate target: keep ysd=1 to avoid blowups
        ysd = 1.0
    yn = (y - ymu) / ysd
    return Xn, yn, (Xmu, Xsd, ymu, ysd)

def _rbf_kernel(X, Z, log_ell, log_sf2):
    ell = np.exp(log_ell)
    sf2 = np.exp(log_sf2)
    X2 = np.sum(X**2, axis=1, keepdims=True)
    Z2 = np.sum(Z**2, axis=1, keepdims=True).T
    d2 = X2 + Z2 - 2.0 * X @ Z.T
    return sf2 * np.exp(-0.5 * d2 / (ell**2 + 1e-12))

def _chol_solve(L, b):
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)

def _neg_log_marginal_lik(y, K):
    n = y.shape[0]
    L = np.linalg.cholesky(K)
    alpha = _chol_solve(L, y)
    nll = 0.5 * (y @ alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2.0 * np.pi)
    if not np.isfinite(nll):
        nll = np.inf
    return float(nll), L, alpha

def _build_K_adaptive(X, log_ell, log_sf2, log_sn2, base_jitter=1e-6, max_tries=6):
    K = _rbf_kernel(X, X, log_ell, log_sf2)
    sn2 = np.exp(log_sn2)
    n = X.shape[0]
    jitter = base_jitter
    for _ in range(max_tries):
        K_try = K.copy()
        K_try.flat[:: n + 1] += (sn2 + jitter)  # add to diagonal
        try:
            # probe Cholesky just to test conditioning
            _ = np.linalg.cholesky(K_try)
            return K_try, jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # last attempt: return with largest jitter; let caller handle failure in nll
    K.flat[:: n + 1] += (sn2 + jitter)
    return K, jitter

def _random_restarts_bounds(d, low=-5.0, high=5.0, rng=None, n=12):
    rng = np.random.default_rng() if rng is None else rng
    return [rng.uniform(low, high, size=d) for _ in range(n)]

def _grid_around(best, width=0.75, steps=3):
    grids = []
    for i in range(len(best)):
        vs = best[i] + np.linspace(-width, width, steps)
        grids.append(vs)
    mesh = np.meshgrid(*grids, indexing="ij")
    cand = np.stack([m.reshape(-1) for m in mesh], axis=1)
    return [c for c in cand]

def _mdl_bits_from_nll(nll_nats, k_params, n):
    # BIC/MDL penalty
    penalty = 0.5 * k_params * np.log(max(n, 2))
    return (nll_nats + penalty) / np.log(2.0)

def _null_gaussian_mdl_bits(y):
    """Fallback: y ~ N(mu, sigma^2) i.i.d., MLE parameters; MDL in bits with BIC."""
    n = len(y)
    if n == 0:
        return 1e9  # degenerate
    mu = np.mean(y)
    var = np.var(y) + 1e-12
    nll = 0.5 * n * (np.log(2 * np.pi * var) + 1.0)  # exact MLE -loglik in nats
    k = 2  # mu, var
    return _mdl_bits_from_nll(nll, k, n)

def fit_score_gp(Xtr, ytr, **params):
    """
    Robust GP (RBF + noise) with safe fallbacks.
    Returns:
      model dict, score_bits (finite)
    """
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float).reshape(-1)
    # guard: finite data
    if not np.all(np.isfinite(Xtr)) or not np.all(np.isfinite(ytr)):
        # replace non-finite with zeros (and continue)
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        ytr = np.nan_to_num(ytr, nan=0.0, posinf=0.0, neginf=0.0)

    # standardize (improves conditioning); store scalers
    Xn, yn, scalers = _standardize(Xtr, ytr)
    n = Xn.shape[0]

    # hyperparam search config
    restarts = params.get("restarts", 10)
    low = params.get("bounds", {}).get("low", -5.0)
    high = params.get("bounds", {}).get("high", 5.0)
    refine = params.get("refine", True)
    rng = np.random.default_rng(params.get("seed", None))
    base_jitter = params.get("base_jitter", 1e-6)
    k_params = params.get("k_params", 3)
    use_bic = params.get("bic_penalty", True)

    # candidates
    cands = _random_restarts_bounds(3, low=low, high=high, rng=rng, n=restarts)
    # Add anchor candidates tuned to standardized y (var ~ 1)
    cands += [
        np.array([0.0, 0.0, -2.0]),   # ell=1, sf2=1, sn2≈0.135
        np.array([1.0, 0.0, -1.0]),   # ell≈2.7, sf2=1, sn2≈0.368
        np.array([-1.0, 0.0, 0.0]),   # ell≈0.37, sf2=1, sn2=1
        np.array([0.0, -2.0, 0.0]),   # low sf2 (underfit)
        np.array([0.0,  2.0,  2.0]),  # high sf2 & sn2 (noise-dominated)
    ]

    best = None
    best_nll = np.inf
    best_cache = None
    best_jitter = base_jitter

    def eval_params(theta):
        log_ell, log_sf2, log_sn2 = theta
        # clamp noise variance to a minimum to avoid pathological sn2->0
        log_sn2 = max(log_sn2, np.log(1e-6))
        K, used_jitter = _build_K_adaptive(Xn, log_ell, log_sf2, log_sn2, base_jitter=base_jitter)
        try:
            nll, L, alpha = _neg_log_marginal_lik(yn, K)
        except np.linalg.LinAlgError:
            return np.inf, None, used_jitter
        if not np.isfinite(nll):
            return np.inf, None, used_jitter
        return nll, (theta, K, L, alpha), used_jitter

    # coarse
    for th in cands:
        nll, cache, used_jit = eval_params(th)
        if nll < best_nll:
            best_nll, best, best_cache, best_jitter = nll, th, cache, used_jit

    # refine
    if refine and best is not None and np.isfinite(best_nll):
        for th in _grid_around(best, width=0.75, steps=3):
            nll, cache, used_jit = eval_params(th)
            if nll < best_nll:
                best_nll, best, best_cache, best_jitter = nll, th, cache, used_jit

    # Fallback if nothing worked
    if (best is None) or (not np.isfinite(best_nll)):
        score_bits = _null_gaussian_mdl_bits(yn) if use_bic else (_null_gaussian_mdl_bits(yn))
        # return a trivial model (predict mean in original scale)
        Xmu, Xsd, ymu, ysd = scalers
        def predict(Xte, return_var=False):
            m = np.full((np.asarray(Xte).shape[0],), ymu)
            if return_var:
                return m, np.full_like(m, ysd**2)
            return m
        model = dict(
            kind="fallback_null",
            mdl_bits=float(score_bits),
            predict=predict,
            scalers=dict(Xmu=Xmu, Xsd=Xsd, ymu=ymu, ysd=ysd),
        )
        return model, float(score_bits)

    # unpack best cache
    (log_ell, log_sf2, log_sn2), K, L, alpha = best_cache

    # MDL score (in bits)
    penalty = (0.5 * k_params * np.log(max(n, 2))) if use_bic else 0.0
    score_bits = (best_nll + penalty) / np.log(2.0)

    # prediction in ORIGINAL scale
    Xmu, Xsd, ymu, ysd = scalers
    def predict(Xte, return_var=False):
        Xte = np.asarray(Xte, float)
        Xte_n = (Xte - Xmu) / Xsd
        Kxs = _rbf_kernel(Xn, Xte_n, log_ell, log_sf2)
        mean_n = Kxs.T @ alpha
        if not return_var:
            return ymu + ysd * mean_n
        Kxx = _rbf_kernel(Xte_n, Xte_n, log_ell, log_sf2)
        v = np.linalg.solve(L, Kxs)
        var_n = np.maximum(0.0, np.diag(Kxx) - np.sum(v**2, axis=0)) + np.exp(log_sn2)
        return ymu + ysd * mean_n, (ysd**2) * var_n

    model = {
        "kind": "gp_rbf",
        "log_ell": float(log_ell),
        "log_sf2": float(log_sf2),
        "log_sn2": float(max(log_sn2, np.log(1e-6))),
        "Xtr_std": Xn,  # standardized copy (for debugging)
        "ytr_std": yn,
        "L": L,
        "alpha": alpha,
        "predict": predict,
        "nll_nats": float(best_nll),
        "mdl_bits": float(score_bits),
        "used_jitter": float(best_jitter),
        "scalers": dict(Xmu=Xmu, Xsd=Xsd, ymu=ymu, ysd=ysd),
    }
    return model, float(score_bits)

# other

def fit_score_gp_old(Xtr, ytr, score_type):
    gp = fit_gaussian_process(
        Xtr, ytr,
        scoring_function=  TimeseriesScoringFunction.GP if score_type==GPType.EXACT else TimeseriesScoringFunction.GP_QFF,
        check_fit=False)
    score, lik, model, pen = gp.mdl_score_ytrain()

    return gp, score


def fit_score_gam(Xtr, ytr):
    gam = GAM()
    gam.fit(Xtr, ytr)
    n_splines, order = 20, 3
    mse = np.mean((gam.predict(Xtr) - ytr) ** 2)
    n = Xtr.shape[0]
    p = Xtr.shape[1] * n_splines * order
    gam.mdl_lik_train = n * np.log(mse)
    gam.mdl_model_train = 2 * p
    gam.mdl_pen_train = 0
    gam.mdl_train = gam.mdl_lik_train + gam.mdl_model_train + gam.mdl_pen_train
    return gam, gam.mdl_train


class RegressorType(Enum):
    LN = 0
    SPLN = 1
    GAM = 2
    GP = 3
    NN = 4


def fit_fun(causes, effect, reg_type, seed=42, test_indep=False):
    if causes.shape[1] == 0:
        causes = np.random.normal(size=effect.reshape(-1, 1).shape)
    if reg_type == RegressorType.LN:
        return fit_ln(causes, effect, seed, test_indep)
    elif reg_type == RegressorType.SPLN:
        return fit_spln(causes, effect, seed, test_indep)
    elif reg_type == RegressorType.GAM:
        return fit_gam(causes, effect, seed, test_indep)
    else:
        raise ValueError("Unknown regression type.")


def fit_ln(causes, effect, seed, test_indep=False):
    m = LinearRegression()
    m.fit(causes, effect)
    preds = m.predict(causes)
    resids = (effect - preds).reshape(-1, 1)
    loglik_strength = -0.5 * mean_squared_error(effect, preds) * len(effect)
    return resids, loglik_strength


def fit_spln(causes, effect, seed, test_indep=False):
    if causes.shape[1] > 1:
        raise ValueError("Polynomial spline fitting only supports single feature currently.")

    causes_flat = causes.flatten()
    knots = np.linspace(min(causes_flat), max(causes_flat), 4)  # Adjust number of knots as needed
    m = make_lsq_spline(causes_flat, effect, t=knots[1:-1])
    predictions = m(causes_flat)
    resids = (effect - predictions).reshape(-1, 1)
    loglik_strength = -0.5 * mean_squared_error(effect, predictions) * len(effect)

    return resids, loglik_strength


def fit_gam(causes, effect, seed, test_indep=False):
    m = LinearGAM().fit(causes, effect)
    preds = m.predict(causes)
    resids = (effect - preds).reshape(-1, 1)
    loglik_strength = -0.5 * mean_squared_error(effect, preds) * len(effect)

    return resids, loglik_strength

