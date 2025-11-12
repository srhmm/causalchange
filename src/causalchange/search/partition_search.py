import numpy as np
import networkx as nx
from functools import lru_cache

import numpy as np
import networkx as nx
from functools import lru_cache

from causallearn.utils.cit import KCI

import numpy as np

import numpy as np
import networkx as nx
import numpy as np
import networkx as nx

import numpy as np

# expects you already have:
# - flat_from_context_dict, krr_residuals, discrepancy_mmd
# - gaussian_k, median_heuristic, etc. (from earlier messages)

def hsic_unbiased(X, Y, sigma_x=None, sigma_y=None):
    X = np.asarray(X); Y = np.asarray(Y)
    if X.ndim == 1: X = X.reshape(-1,1)
    if Y.ndim == 1: Y = Y.reshape(-1,1)
    n = X.shape[0]
    if n < 4: return 0.0
    def _med(Z):
        if Z.ndim == 1: Z = Z.reshape(-1,1)
        if Z.shape[0] < 2: return 1.0
        D2 = ((Z[:,None,:]-Z[None,:,:])**2).sum(-1)
        d2 = D2[np.triu_indices_from(D2, 1)]
        d2 = d2[d2 > 0]
        return float(np.sqrt(np.median(d2))) if d2.size else 1.0
    sx = _med(X) if sigma_x is None else float(sigma_x)
    sy = _med(Y) if sigma_y is None else float(sigma_y)
    from numpy import eye, ones
    def _gauss(A,B,s):
        if A.ndim==1: A=A.reshape(-1,1)
        if B.ndim==1: B=B.reshape(-1,1)
        A2=(A**2).sum(1,keepdims=True); B2=(B**2).sum(1,keepdims=True).T
        D2=A2-2*A@B.T+B2
        return np.exp(-D2/(2*s**2))
    K = _gauss(X,X,sx); L = _gauss(Y,Y,sy)
    np.fill_diagonal(K, 0.0); np.fill_diagonal(L, 0.0)
    H = eye(n) - ones((n,n))/n
    KH = K @ H; LH = L @ H
    hsic = np.trace(KH @ LH) / (n-1)**2
    return float(max(0.0, hsic))

def prune_incoming_combo(target, G, X_all, C_idx,
                         lam_mix=0.7,         # weight on invariance vs independence
                         keep_min=0.05,       # required combined relative contribution to KEEP
                         krr_lam=1e-2, krr_sigma=None, mmd_sigma=None, eps=1e-12):
    """
    Backward elimination for incoming edges u->target.
    For each parent u, compute:
      - Invariance contribution (relative): rel_inv = (D_minus - D_full)/max(D_minus,eps)
      - Independence contribution (relative): rel_dep = (HSIC_minus - HSIC_full)/max(HSIC_minus,eps),
        where HSIC is between residuals and X_u (should drop when u is included).
    Keep u iff lam_mix*rel_inv + (1-lam_mix)*rel_dep >= keep_min.
    Iterate until stable. Returns list of removed (u, target).
    """
    removed = []
    P = list(G.predecessors(target))
    if not P: return removed

    # Full invariance with all current parents
    from math import isfinite
    D_full = discrepancy_mmd(target, P, X_all, C_idx, krr_lam=krr_lam, krr_sigma=krr_sigma, mmd_sigma=mmd_sigma)
    # Residuals with all parents (reuse across candidates for HSIC_full)
    y = X_all[:, target]
    XP = X_all[:, P] if P else np.zeros((X_all.shape[0], 0))
    R_full = krr_residuals(y, XP, lam=krr_lam, sigma=krr_sigma)

    changed = True
    while changed and P:
        changed = False
        to_drop = []
        for u in list(P):
            P_minus = [v for v in P if v != u]
            # Invariance contribution (MMD)
            D_minus = discrepancy_mmd(target, P_minus, X_all, C_idx, krr_lam=krr_lam, krr_sigma=krr_sigma, mmd_sigma=mmd_sigma)
            rel_inv = (D_minus - D_full) / (max(D_minus, eps))

            # Independence contribution (HSIC residual ⟂ X_u)
            Xu = X_all[:, u]
            # HSIC with u included (should be small)
            hs_full = hsic_unbiased(R_full, Xu)
            # HSIC when u is removed (should increase if u matters)
            XPm = X_all[:, P_minus] if P_minus else np.zeros((X_all.shape[0], 0))
            R_minus = krr_residuals(y, XPm, lam=krr_lam, sigma=krr_sigma)
            hs_minus = hsic_unbiased(R_minus, Xu)
            rel_dep = (hs_minus - hs_full) / (max(hs_minus, eps))

            combo = lam_mix * rel_inv + (1.0 - lam_mix) * rel_dep
            if not isfinite(combo): combo = -np.inf
            if combo < keep_min:
                to_drop.append((u, D_minus, R_minus))
        if to_drop:
            # remove one-by-one (largest “not helping” first, i.e., smallest combo)
            for u, Dm, Rm in to_drop:
                G.remove_edge(u, target)
                removed.append((u, target))
                P.remove(u)
                D_full = Dm
                R_full = Rm
                changed = True
    return removed

def prune_incoming_rel(target, G, X_all, C_idx, rel_keep=0.02,  # min relative contribution to KEEP an edge
                       krr_lam=1e-2, krr_sigma=None, mmd_sigma=None, eps=1e-12):
    """
    Backward elimination for incoming edges to `target` using relative-contribution criterion.
    Keep parent u only if removing it worsens invariance by at least `rel_keep` fraction.

    rel contribution of u := (D_without_u - D_full) / max(D_without_u, eps)

    If rel < rel_keep, drop u. Iterate until stable.
    Returns: list of removed edges (u, target)
    """
    removed = []
    P = list(G.predecessors(target))
    if not P:
        return removed

    # current full discrepancy with all parents
    D_full = discrepancy_mmd(target, P, X_all, C_idx, krr_lam=krr_lam, krr_sigma=krr_sigma, mmd_sigma=mmd_sigma)

    changed = True
    while changed and P:
        changed = False
        for u in list(P):
            P_minus = [v for v in P if v != u]
            D_minus = discrepancy_mmd(target, P_minus, X_all, C_idx, krr_lam=krr_lam, krr_sigma=krr_sigma, mmd_sigma=mmd_sigma)
            rel = (D_minus - D_full) / (max(D_minus, eps))
            if rel < rel_keep:
                # removing u doesn't hurt invariance enough -> drop it
                G.remove_edge(u, target)
                removed.append((u, target))
                P = P_minus
                D_full = D_minus
                changed = True
    return removed

def add_edges_rel_reduction(order, X_all, C_idx, rel_min=0.1, lambda_pa=0.0,
                            max_parents=None, krr_lam=1e-2, krr_sigma=None, mmd_sigma=None):
    import networkx as nx, numpy as np
    G = nx.DiGraph(); G.add_nodes_from(range(X_all.shape[1]))
    pos = {v:i for i,v in enumerate(order)}
    edges = []
    for t in order:
        P = []
        D_base = discrepancy_mmd(t, P, X_all, C_idx, krr_lam, krr_sigma, mmd_sigma)
        while True:
            cand = [u for u in order if pos[u] < pos[t] and u not in P]
            if not cand: break
            best_u, best_D, best_rel = None, D_base, 0.0
            for u in cand:
                D = discrepancy_mmd(t, P+[u], X_all, C_idx, krr_lam, krr_sigma, mmd_sigma)
                rel = (D_base - D) / (D_base + 1e-12)
                # simple penalized score
                score = -rel + lambda_pa * (len(P)+1)
                if rel > best_rel and score < (-best_rel + lambda_pa * (len(P)+1)):
                    best_u, best_D, best_rel = u, D, rel
            if best_u is None or best_rel < rel_min: break
            if max_parents is not None and len(P) >= max_parents: break
            P.append(best_u); G.add_edge(best_u, t); edges.append((best_u, t)); D_base = best_D
    return G, edges

def discrepancy_mmd(effect, parents, X_all, C_idx, krr_lam=1e-2, krr_sigma=None, mmd_sigma=None):
    Xi = X_all[:, effect]
    Xp = X_all[:, parents] if parents else np.zeros((X_all.shape[0], 0))
    R = krr_residuals(Xi, Xp, lam=krr_lam, sigma=krr_sigma)
    return mmd_across_contexts(R, C_idx, sigma=mmd_sigma)

def pick_source_mmd_single(candidates, X_all, C_idx, krr_lam=1e-2, krr_sigma=None, mmd_sigma=None, eps=1e-9):
    stats = []
    for i in candidates:
        D0 = discrepancy_mmd(i, [], X_all, C_idx, krr_lam, krr_sigma, mmd_sigma)
        Dmin = D0
        for j in candidates:
            if j == i: continue
            D1 = discrepancy_mmd(i, [j], X_all, C_idx, krr_lam, krr_sigma, mmd_sigma)
            if D1 < Dmin: Dmin = D1
        R = D0 - Dmin
        Rn = R / (D0 + eps)
        stats.append((i, D0, Dmin, R, Rn))
    best = min(range(len(stats)), key=lambda k: (stats[k][4], -stats[k][1]))
    return stats[best][0], stats  # node, full stats

def perm_test_improve(effect, parents_old, parent_new, X_all, C_idx, delta_obs, n_perm=200, krr_lam=1e-2, krr_sigma=None, mmd_sigma=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    Xi = X_all[:, effect]
    Xp0 = X_all[:, parents_old] if parents_old else np.zeros((X_all.shape[0], 0))
    Xp1 = X_all[:, parents_old + [parent_new]]
    R0 = krr_residuals(Xi, Xp0, lam=krr_lam, sigma=krr_sigma)
    R1 = krr_residuals(Xi, Xp1, lam=krr_lam, sigma=krr_sigma)
    greater = 0
    for _ in range(n_perm):
        C_perm = rng.permutation(C_idx)
        D0b = mmd_across_contexts(R0, C_perm, sigma=mmd_sigma)
        D1b = mmd_across_contexts(R1, C_perm, sigma=mmd_sigma)
        if (D0b - D1b) >= delta_obs:
            greater += 1
    return (greater + 1.0) / (n_perm + 1.0)

def add_edges_change_mmd_given_order(order, X_all, C_idx, alpha=0.05, n_perm=200, max_parents=None, min_improve=0.0, krr_lam=1e-2, krr_sigma=None, mmd_sigma=None, rng=None):
    G = nx.DiGraph(); G.add_nodes_from(range(X_all.shape[1]))
    pos = {v:i for i,v in enumerate(order)}
    edges = []
    for t in order:
        P = []
        D_base = discrepancy_mmd(t, P, X_all, C_idx, krr_lam, krr_sigma, mmd_sigma)
        while True:
            cand = [u for u in order if pos[u] < pos[t] and u not in P]
            if not cand: break
            best_u, best_D = None, D_base
            for u in cand:
                D = discrepancy_mmd(t, P+[u], X_all, C_idx, krr_lam, krr_sigma, mmd_sigma)
                if D < best_D:
                    best_u, best_D = u, D
            if best_u is None: break
            delta = D_base - best_D
            if delta <= min_improve: break
            pval = perm_test_improve(t, P, best_u, X_all, C_idx, delta, n_perm, krr_lam, krr_sigma, mmd_sigma, rng)
            if pval <= alpha and (max_parents is None or len(P) < max_parents):
                P.append(best_u); edges.append((best_u, t)); D_base = best_D
            else:
                break
        for u in P: G.add_edge(u, t)
    return G, edges


# ------------ data ------------
def flat_from_context_dict(X_dict, context_labels=None):
    if not isinstance(X_dict, dict):
        X_all = np.asarray(X_dict)
        C_idx = np.asarray(context_labels) if context_labels is not None else np.zeros(X_all.shape[0], dtype=int)
        return X_all, C_idx
    ctxs = sorted(X_dict.keys())
    X_list = [np.asarray(X_dict[c]) for c in ctxs]
    X_all = np.vstack(X_list)
    C_idx = np.concatenate([np.full(x.shape[0], ctxs[i], dtype=int) for i, x in enumerate(X_list)])
    return X_all, C_idx

# ------------ kernels & residuals ------------
def median_heuristic(Z):
    Z = np.asarray(Z)
    if Z.ndim == 1: Z = Z.reshape(-1,1)
    if Z.shape[0] < 2: return 1.0
    D2 = ((Z[:,None,:]-Z[None,:,:])**2).sum(-1)
    d2 = D2[np.triu_indices_from(D2, 1)]
    d2 = d2[d2 > 0]
    return float(np.sqrt(np.median(d2))) if d2.size else 1.0

def gaussian_k(A, B, sigma, eps=1e-9):
    A = np.asarray(A); B = np.asarray(B)
    if A.ndim == 1: A = A.reshape(-1,1)
    if B.ndim == 1: B = B.reshape(-1,1)
    A2 = (A**2).sum(1, keepdims=True); B2 = (B**2).sum(1, keepdims=True).T
    D2 = A2 - 2*A@B.T + B2
    return np.exp(-D2/(2*(sigma**2 + eps)))

def krr_residuals(y, Xp, lam=1e-2, sigma=None):
    y = np.asarray(y).ravel()
    if Xp is None or np.size(Xp) == 0:
        return y.copy()
    Xp = np.asarray(Xp)
    s = median_heuristic(Xp) if sigma is None else float(sigma)
    K = gaussian_k(Xp, Xp, s)
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam*np.eye(n), y)
    yhat = K @ alpha
    return y - yhat

# ------------ MMD across contexts (invariance) ------------
def mmd2_gaussian(u, v, sigma):
    u = np.asarray(u).reshape(-1,1); v = np.asarray(v).reshape(-1,1)
    Ku = gaussian_k(u,u,sigma); Kv = gaussian_k(v,v,sigma); Kuv = gaussian_k(u,v,sigma)
    np.fill_diagonal(Ku, 0.0); np.fill_diagonal(Kv, 0.0)
    nu, nv = len(u), len(v)
    term1 = Ku.sum()/(max(1, nu*(nu-1)))
    term2 = Kv.sum()/(max(1, nv*(nv-1)))
    term3 = 2.0*Kuv.sum()/(max(1, nu*nv))
    return float(term1 + term2 - term3)

def mmd_across_contexts(residuals, C_idx, sigma=None):
    C_idx = np.asarray(C_idx)
    ctxs = np.unique(C_idx)
    Rs = [residuals[C_idx==c] for c in ctxs]
    r_all = np.concatenate(Rs) if len(Rs) else residuals
    s = median_heuristic(r_all) if sigma is None else float(sigma)
    vals = []
    for a in range(len(ctxs)):
        for b in range(a+1, len(ctxs)):
            if len(Rs[a]) >= 2 and len(Rs[b]) >= 2:
                vals.append(mmd2_gaussian(Rs[a], Rs[b], s))
    return float(np.mean(vals)) if vals else 0.0

def discrepancy_mmd(effect, parents, X_all, C_idx, krr_lam=1e-2, krr_sigma=None, mmd_sigma=None):
    Xi = X_all[:, effect]
    Xp = X_all[:, parents] if parents else np.zeros((X_all.shape[0], 0))
    R = krr_residuals(Xi, Xp, lam=krr_lam, sigma=krr_sigma)
    return mmd_across_contexts(R, C_idx, sigma=mmd_sigma)

# ------------ HSIC (residual independence) ------------
def hsic_unbiased(X, Y, sigma_x=None, sigma_y=None):
    """
    Unbiased HSIC estimator (Gretton et al. 2005). Returns a nonnegative statistic.
    X: (n,d_x), Y: (n,d_y)
    """
    X = np.asarray(X); Y = np.asarray(Y)
    if X.ndim == 1: X = X.reshape(-1,1)
    if Y.ndim == 1: Y = Y.reshape(-1,1)
    n = X.shape[0]
    if n < 4: return 0.0
    sx = median_heuristic(X) if sigma_x is None else float(sigma_x)
    sy = median_heuristic(Y) if sigma_y is None else float(sigma_y)
    K = gaussian_k(X, X, sx); L = gaussian_k(Y, Y, sy)
    np.fill_diagonal(K, 0.0); np.fill_diagonal(L, 0.0)
    H = np.eye(n) - np.ones((n, n))/n
    KH = K @ H; LH = L @ H
    hsic = np.trace(KH @ LH) / (n-1)**2
    return float(max(0.0, hsic))

def residual_dependence_all(effect, X_all, regressors_idx, krr_lam=1e-2, krr_sigma=None, hsic_sig_x=None, hsic_sig_y=None):
    """
    RESIT-style: regress Xi on X_reg and test dependence between residuals and X_reg via HSIC.
    Higher value => stronger dependence (worse sink, better source).
    """
    Xi = X_all[:, effect]
    Xreg = X_all[:, regressors_idx] if regressors_idx else np.zeros((X_all.shape[0], 0))
    R = krr_residuals(Xi, Xreg, lam=krr_lam, sigma=krr_sigma)
    if Xreg.size == 0:
        return hsic_unbiased(R, Xi)  # degenerate; just something tiny
    return hsic_unbiased(R, Xreg, sigma_x=hsic_sig_y, sigma_y=hsic_sig_x)

###############################################################################
def _entropy_bits_dist(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=-1, keepdims=True)
    return -(p * np.log2(p)).sum(axis=-1)

def _js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 1e-12, 1.0); p /= p.sum(axis=-1, keepdims=True)
    q = np.clip(q, 1e-12, 1.0); q /= q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log2(p) - np.log2(m))).sum(axis=-1)
    kl_qm = (q * (np.log2(q) - np.log2(m))).sum(axis=-1)
    return 0.5 * (kl_pm + kl_qm)

import numpy as np
from functools import lru_cache
from sklearn.neighbors import KernelDensity

def _discrete_entropy_bits(p, dy=1.0):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    p = p / (p.sum() * dy)
    return float(-(p * np.log2(p)).sum() * dy)

def _js_divergence_discrete(p, q, dy=1.0):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0)
    p /= (p.sum() * dy)
    q /= (q.sum() * dy)
    m = 0.5 * (p + q)
    def _kl(a, b): return float(((a * (np.log2(a) - np.log2(b))).sum()) * dy)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def _scotts_bandwidth(X):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    sigma = np.std(X, axis=0, ddof=1)
    return float(np.power(n, -1.0 / (d + 4)) * (np.mean(sigma) + 1e-12))

def _make_y_grid(Y, bins=256, pad=0.05):
    y_min, y_max = np.min(Y), np.max(Y)
    span = y_max - y_min + 1e-9
    y_min -= pad * span
    y_max += pad * span
    grid = np.linspace(y_min, y_max, bins)
    dy = grid[1] - grid[0]
    return grid, dy

def partition_search(
        X, pa, target,
        bandwidth_joint=None,
        bandwidth_x=None,
        y_bins=256,
        vb=0
):
    X_parents = {c: (np.random.normal(size=X[c][:, [target]].shape) if len(pa) == 0 else X[c][:, pa])
                 for c in range(len(X))}
    Y_target = {c: X[c][:, target].reshape(-1, 1) for c in range(len(X))}
    contexts = sorted(X_parents.keys())
    C = len(contexts)

    Y_all = np.vstack([Y_target[c] for c in contexts]).ravel()
    y_grid, dy = _make_y_grid(Y_all, bins=y_bins)

    cond_density = {}
    entropies_bits = []

    for c in contexts:
        Xc = np.atleast_2d(X_parents[c])
        Yc = np.atleast_2d(Y_target[c])
        Zc = np.hstack([Yc, Xc])

        bw_joint = _scotts_bandwidth(Zc) if bandwidth_joint is None else float(bandwidth_joint)
        bw_x = _scotts_bandwidth(Xc) if bandwidth_x is None else float(bandwidth_x)

        kde_joint = KernelDensity(kernel='gaussian', bandwidth=bw_joint).fit(Zc)
        kde_x = KernelDensity(kernel='gaussian', bandwidth=bw_x).fit(Xc)

        # For each sample x_i in context c, compute p(y|x_i) over the grid and average
        py_given_x_avg = np.zeros_like(y_grid, dtype=float)

        for i in range(Xc.shape[0]):
            xi = Xc[i:i+1, :]
            yi_x_grid = np.hstack([y_grid.reshape(-1, 1), np.repeat(xi, repeats=y_grid.size, axis=0)])
            log_p_yx = kde_joint.score_samples(yi_x_grid)
            log_p_x = kde_x.score_samples(xi)[0]
            log_p_y_given_x = log_p_yx - log_p_x
            p_y_given_x = np.exp(log_p_y_given_x)
            p_y_given_x = p_y_given_x / (p_y_given_x.sum() * dy)
            py_given_x_avg += p_y_given_x

        py_given_x_avg /= max(1, Xc.shape[0])
        py_given_x_avg = np.clip(py_given_x_avg, 1e-12, np.inf)
        py_given_x_avg /= (py_given_x_avg.sum() * dy)

        cond_density[c] = py_given_x_avg
        entropies_bits.append(_discrete_entropy_bits(py_given_x_avg, dy=dy))

    score_bits = float(np.mean(entropies_bits))

    results = {
        "contexts": contexts,
        "y_grid": y_grid,
        "dy": dy,
        "cond_density": cond_density,
        "entropy_bits_per_context": np.array(entropies_bits, dtype=float),
        # compatibility fields
        "labels_pred": np.arange(C, dtype=int),
        "groups": [[c] for c in contexts]
    }

    if vb >= 1:
        print(f"[density] contexts={C} | y_bins={y_bins} | mean H(Y|pa)={score_bits:.4f} bits")

    return score_bits, results

def partition_search_constraintbased(
        X, pa, target,
        test_fun,
        alpha=0.05,
        n_perm=200,
        lam_krr=1e-3,
        vb=0
):
    X_parents = {c: np.random.normal(size=X[c][:, [target]].shape) for c in range(len(X))} if len(pa) == 0 else {
        c: X[c][:, pa] for c in range(len(X))}
    X_target = {c: X[c][:, target] for c in range(len(X))}
    contexts = sorted(X_parents.keys())
    C = len(contexts)
    idx_of = {c: i for i, c in enumerate(contexts)}

    @lru_cache(maxsize=None)
    def _pval_pair_sorted(c_small, c_big):
        Xp = np.vstack([X_parents[c_small], X_parents[c_big]])
        Yp = np.concatenate([X_target[c_small], X_target[c_big]], axis=0).reshape(-1, 1)
        Sp = np.concatenate([
            np.zeros((X_parents[c_small].shape[0], 1)),
            np.ones((X_parents[c_big].shape[0], 1))
        ], axis=0)
        return float(test_fun(Yp, Sp, Xp))

    P = np.ones((C, C), dtype=float)
    for i, c in enumerate(contexts):
        for j, d in enumerate(contexts):
            if j <= i:
                continue
            pval = _pval_pair_sorted(*sorted((c, d)))
            P[i, j] = P[j, i] = pval

    edges = []
    for i in range(C):
        for j in range(i + 1, C):
            if P[i, j] >= alpha:
                edges.append((contexts[i], contexts[j]))

    G = nx.Graph()
    G.add_nodes_from(contexts)
    G.add_edges_from(edges)
    comps = list(nx.connected_components(G))
    groups = [sorted(list(comp)) for comp in comps]
    part = {c: gid for gid, grp in enumerate(groups) for c in grp}
    labels_pred = np.array([part[c] for c in contexts], dtype=int)

    S = P.copy()
    np.fill_diagonal(S, 1.0)
    S = np.clip(S, 1e-12, 1.0)
    S = S / S.sum(axis=1, keepdims=True)
    probs_pred = {contexts[i]: S[i] for i in range(C)}

    total_score_bits = float(_entropy_bits(labels_pred))

    results = {
        "partition": part,
        "groups": groups,
        "labels_pred": labels_pred,
        "contexts": contexts,
        "pairwise_p": P,
        "edges": edges,
        "alpha": float(alpha),
        "probs_pred": probs_pred
    }
    if vb >= 1:
        print(f"[cc] alpha={alpha} | edges={len(edges)} | groups={groups}")
        print(f"[final] target {target} groups={groups}")
    return total_score_bits, results

def partition_search_constraintbased_old(
        X, pa, target,
        test_fun,        # callable: (Y, S, Z) -> pval in [0,1]; if None uses residual-HSIC
        alpha=0.05,
        n_perm=200,           # only for fallback
        lam_krr=1e-3,         # only for fallback
        vb=0
):
    """
    Constraint-based version of partition search.
    Tests, for each pair of contexts (c,d), whether Y ⟂ S | X_pa within the pooled data of {c,d},
    using a kernel conditional test (KCI via `test_fun`) or a residual-HSIC fallback.
    Adds an undirected edge between c and d iff p >= alpha (fail to reject discrepancy),
    then returns connected components as groups.
    """

    # assemble per-context parents & target, exactly like your score-based version
    X_parents = {c: np.random.normal(size=X[c][:, [target]].shape) for c in range(len(X))} if len(pa) == 0 else {
        c: X[c][:, pa] for c in range(len(X))}
    X_target = {c: X[c][:, target] for c in range(len(X))}

    contexts = sorted(X_parents.keys())
    C = len(contexts)
    idx_of = {c: i for i, c in enumerate(contexts)}

    # (optional) quick sizes log
    if vb >= 2:
        msg = ", ".join([f"{c}: n={X_parents[c].shape[0]}" for c in contexts])
        print(f"[contexts] {msg}")

    # cache pairwise tests
    @lru_cache(maxsize=None)
    def _pval_pair_sorted(c_small, c_big):
        # pool the two contexts
        Xp = np.vstack([X_parents[c_small], X_parents[c_big]])
        Yp = np.concatenate([X_target[c_small], X_target[c_big]], axis=0).reshape(-1, 1)
        Sp = np.concatenate([
            np.zeros((X_parents[c_small].shape[0], 1)),
            np.ones((X_parents[c_big].shape[0], 1))
        ], axis=0)
        p = float(test_fun(Yp, Sp, Xp))
        return p

    # pairwise p-values
    P = np.ones((C, C), dtype=float)
    for i, c in enumerate(contexts):
        for j, d in enumerate(contexts):
            if j <= i:
                continue
            pval = _pval_pair_sorted(*sorted((c, d)))
            P[i, j] = P[j, i] = pval
    if vb >= 2:
        print(f"[pairwise] p-value stats: min={P[P<1].min():.3f}, max={P.max():.3f}, mean={P[np.triu_indices(C,1)].mean():.3f}")

    # edges: connect if NOT significantly different (p >= alpha)
    edges = []
    for i in range(C):
        for j in range(i + 1, C):
            if P[i, j] >= alpha:
                edges.append((contexts[i], contexts[j]))

    G = nx.Graph()
    G.add_nodes_from(contexts)
    G.add_edges_from(edges)
    comps = list(nx.connected_components(G))
    groups = [sorted(list(comp)) for comp in comps]
    part = {c: gid for gid, grp in enumerate(groups) for c in grp}

    labels_pred = np.array([part[c] for c in contexts], dtype=int)

    total_score_bits =  float(_entropy_bits(labels_pred))

    results = {
        "partition": part,
        "groups": groups,
        "labels_pred": labels_pred,
        "contexts": contexts,
        "pairwise_p": P,
        "edges": edges,
        "alpha": float(alpha),
    }

    if vb >= 1:
        print(f"[cc] alpha={alpha} | edges={len(edges)} | groups={groups}")
        print(f"[final] target {target} groups={groups}")

    return total_score_bits, results

def _entropy_bits(labels):
    labels = np.asarray(labels, int).ravel()
    if labels.size == 0:
        return 0.0
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def partition_search_scorebased(
        X, pa, target,
        score_fun,  # callable: (Xtr, ytr, params_score_type_dict) -> (model, score_bits)
        alpha=0.05,  # significance thresh for no-hypercompression
        vb=0
):
    """
    """

    X_parents = {c: np.random.normal(size=X[c][:, [target]].shape) for c in range(len(X))} if len(pa) == 0 else {
        c: X[c][:, pa] for c in range(len(X))}
    X_target = {c: X[c][:, target] for c in range(len(X))}

    contexts = sorted(X_parents.keys())
    C = len(contexts)
    idx_of = {c: i for i, c in enumerate(contexts)}

    models_c, Lc = {}, {}
    for c in contexts:
        Xtr = X_parents[c]
        ytr = X_target[c]
        models_c[c], Lc[c] = score_fun(Xtr, ytr, **dict())
    if vb >= 1:
        msg = ", ".join([f"{c}: {Lc[c]:.3f}" for c in contexts])
        print(f"[per-context] L_c bits: {msg}")

    def _concat_ctx(ctx_list):
        Xs = [X_parents[c] for c in ctx_list]
        ys = [X_target[c] for c in ctx_list]
        return np.vstack(Xs), np.concatenate(ys, axis=0)

    @lru_cache(maxsize=None)
    def _fit_pair_sorted(c_small, c_big):
        Xtr, ytr = _concat_ctx([c_small, c_big])
        model, L = score_fun(Xtr, ytr, **dict())
        return model, float(L)

    @lru_cache(maxsize=None)
    def _fit_group_tuple(ctx_tuple_sorted):
        if len(ctx_tuple_sorted) == 2:
            c_small, c_big = ctx_tuple_sorted
            return _fit_pair_sorted(c_small, c_big)
        Xtr, ytr = _concat_ctx(list(ctx_tuple_sorted))
        model, L = score_fun(Xtr, ytr, **dict())
        return model, float(L)

    # pairw joint fits and delta
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
        for j in range(i + 1, C):
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
        "partition": part,  # dict: context  group id
        "groups": groups,  # list of lists of contexts
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


def partition_search_bottom_up(
        X, pa, target,
        fun_gp,  # callable: (Xtr, ytr, params_score_type) -> (model, score_bits)
        alpha=0.05,  # significance thresh for no-hypercompression
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
        models_c[c], Lc[c] = fun_gp(Xtr, ytr, dict())  # , score_type=GPType.EXACT) #params_score_type)

    from functools import lru_cache
    import numpy as np

    def _concat_ctx(ctx_list):
        Xs = [X_parents[c] for c in ctx_list]
        ys = [X_target[c] for c in ctx_list]
        return np.vstack(Xs), np.concatenate(ys, axis=0)

    @lru_cache(maxsize=None)
    def _fit_group_tuple(ctx_tuple_sorted):
        Xtr, ytr = _concat_ctx(list(ctx_tuple_sorted))
        model, L = fun_gp(Xtr, ytr, dict())  # score_type=GPType.EXACT) #params_score_type)
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

    # final MDL over groups + bookkeeping
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
        print(
            f"[partition] target {target} groups={groups} | total_score_bits={float(final_total):.2f} | k={k_bits:.2f}")
    return float(final_total), results

