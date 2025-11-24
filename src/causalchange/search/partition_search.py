
import networkx as nx
import numpy as np
from scipy.stats import gamma
from typing import List

from src.causalchange.scoring.fit_cond_mixture import conditional_mixture_known_assgn, fit_conditional_mixture
from src.causalchange.util.utils_idl import get_true_idl, get_true_idl_Z


def edge_gain_combo_mmd_resit(
    child,
    parent,
    parents_base,
    X_all,
    C_idx,
    lam_edge=0.5,
    krr_lam=1e-2,
    krr_sigma=None,
    mmd_sigma=None,
    eps=1e-9,
    agg_parents="max",
    agg_contexts="mean",
    min_n_ctx=10,
):
    """
    Compute combined gain for adding edge parent -> child,
    given current parent set parents_base.

    Returns:
        gain_combo, r_mmd, r_dep, D_base, D_with, dep_base, dep_with
    """
    # ensure no duplicates
    parents_base = list(dict.fromkeys(parents_base))
    if parent in parents_base:
        return 0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan

    parents_with = parents_base + [parent]

    # --- MMD part ---
    # baseline discrepancy
    D_base = discrepancy_mmd(child, parents_base, X_all, C_idx,
                             krr_lam=krr_lam, krr_sigma=krr_sigma, mmd_sigma=mmd_sigma)
    if (not np.isfinite(D_base)) or D_base <= 0:
        D_base = 0.0
        r_mmd = 0.0
        D_with = D_base
    else:
        D_with = discrepancy_mmd(child, parents_with, X_all, C_idx,
                                 krr_lam=krr_lam, krr_sigma=krr_sigma, mmd_sigma=mmd_sigma)
        if (not np.isfinite(D_with)) or D_with < 0:
            D_with = D_base  # ignore weird values
        delta_mmd = max(0.0, D_base - D_with)
        r_mmd = delta_mmd / (D_base + eps)

    # --- RESIT-style HSIC dep part ---
    dep_base = resit_dep_score_joint_pairwise(
        effect=child,
        candidates=parents_base,
        X_all=X_all,
        C_idx=C_idx,
        krr_lam=krr_lam,
        krr_sigma=krr_sigma,
        agg_parents=agg_parents,
        agg_contexts=agg_contexts,
        min_n_ctx=min_n_ctx,
    ) if len(parents_base) > 0 else 0.0

    if (not np.isfinite(dep_base)) or dep_base < 0:
        dep_base = 0.0

    if len(parents_with) > 0:
        dep_with = resit_dep_score_joint_pairwise(
            effect=child,
            candidates=parents_with,
            X_all=X_all,
            C_idx=C_idx,
            krr_lam=krr_lam,
            krr_sigma=krr_sigma,
            agg_parents=agg_parents,
            agg_contexts=agg_contexts,
            min_n_ctx=min_n_ctx,
        )
        if (not np.isfinite(dep_with)) or dep_with < 0:
            dep_with = dep_base
    else:
        dep_with = dep_base

    if dep_base > 0:
        delta_dep = max(0.0, dep_base - dep_with)
        r_dep = delta_dep / (dep_base + eps)
    else:
        r_dep = 0.0

    # --- Combine ---
    if not np.isfinite(r_mmd):
        r_mmd = 0.0
    if not np.isfinite(r_dep):
        r_dep = 0.0

    gain_combo = lam_edge * r_mmd + (1.0 - lam_edge) * r_dep

    if not np.isfinite(gain_combo):
        gain_combo = 0.0

    return gain_combo, r_mmd, r_dep, D_base, D_with, dep_base, dep_with

def resit_dep_score_joint(
    effect,
    parents,
    X_all,
    C_idx,
    dep_fun,
    krr_lam=1e-2,
    krr_sigma=None,
    min_n_ctx=10,
):
    """
    RESIT-style dependence score for node `effect` with joint parent set `parents`.

    For each context c:
      - regress X_effect(c) on X_parents(c)
      - compute dep_fun(residuals, X_parents(c))  (e.g. HSIC)
    Aggregate across contexts (mean).

    Returns: scalar dependence score (larger = more dependence).
    """
    X_all = np.asarray(X_all, float)
    y = X_all[:, effect]
    S = y.shape[0]

    parents = list(parents)
    if len(parents) == 0:
        # No parents: residuals = raw y
        # (you could also use krr_residuals with empty design)
        R_full = y.reshape(-1, 1)
        X_par_full = np.zeros((S, 0))
    else:
        X_par_full = X_all[:, parents]
        # One global fit, then we reuse per context
        R_full = krr_residuals(y, X_par_full, lam=krr_lam, sigma=krr_sigma)

    C_idx = np.asarray(C_idx, int)
    contexts = np.unique(C_idx)

    dep_vals = []
    for c in contexts:
        mask = (C_idx == c)
        n_c = int(mask.sum())
        if n_c < min_n_ctx:
            continue

        R_c = R_full[mask].reshape(-1, 1)
        if len(parents) == 0:
            Xp_c = np.zeros((n_c, 0))
        else:
            Xp_c = X_par_full[mask, :]

        # dep_fun is something like hsic_unbiased(R_c, Xp_c)
        d_c = float(dep_fun(R_c, Xp_c)) if Xp_c.shape[1] > 0 else 0.0
        if np.isfinite(d_c):
            dep_vals.append(max(0.0, d_c))

    if len(dep_vals) == 0:
        return 0.0

    return float(np.mean(dep_vals))

def edge_dep_resit_per_context(
    effect,
    parent,
    parents_all,
    X_all,
    C_idx,
    krr_lam=1e-2,
    krr_sigma=None,
    min_n_ctx=10,
    agg_contexts="mean",   # "mean" or "max"
):
    """
    RESIT-style dependence score for edge parent -> effect over contexts.

    For each context c:
      - regress X_effect(c) on all parents_all \ {parent}
      - get residuals R_i^c
      - compute HSIC(R_i^c, X_parent(c))
    Aggregate HSICs over contexts -> dep_score (larger = more evidence that parent is needed).
    """
    X_all = np.asarray(X_all, float)
    C_idx = np.asarray(C_idx, int)

    parents_minus = [q for q in parents_all if q != parent]
    dep_ctx = []

    for c in np.unique(C_idx):
        mask_c = (C_idx == c)
        n_c = int(mask_c.sum())
        if n_c < min_n_ctx:
            continue

        Xc = X_all[mask_c, :]
        y_c = Xc[:, effect]

        if parents_minus:
            X_par_minus_c = Xc[:, parents_minus]
        else:
            # no other parents in the regression
            X_par_minus_c = np.zeros((n_c, 0))

        # residuals of effect regressed on all parents except "parent"
        R_c = krr_residuals(y_c, X_par_minus_c, lam=krr_lam, sigma=krr_sigma)

        Xp_c = Xc[:, parent].reshape(-1, 1)
        h = hsic_unbiased(R_c, Xp_c)

        if np.isfinite(h) and h >= 0.0:
            dep_ctx.append(h)

    if len(dep_ctx) == 0:
        return 0.0

    if agg_contexts == "mean":
        dep_score = float(np.mean(dep_ctx))
    elif agg_contexts == "max":
        dep_score = float(np.max(dep_ctx))
    else:
        raise ValueError(f"Unknown agg_contexts='{agg_contexts}' (use 'mean' or 'max').")

    return max(0.0, dep_score)

def resit_dep_score_joint_pairwise(
    effect,
    candidates,
    X_all,
    C_idx,
    krr_lam=1e-2,
    krr_sigma=None,
    agg_parents="max",    # "max" or "mean"
    agg_contexts="mean",  # "mean" or "max"
    min_n_ctx=10,
):
    """
    RESIT-style dependence score across contexts for node `effect`:

    For each context c:
      - regress X_effect on ALL other candidates jointly
      - compute HSIC(residuals, X_j) for each candidate parent j
      - aggregate HSICs over parents (max/mean) -> d_i^c
    Finally, aggregate d_i^c over contexts -> d_i.

    Returns:
        dep_score : float >= 0   (larger = more dependent = less source-like)
    """
    X_all = np.asarray(X_all, float)
    C_idx = np.asarray(C_idx, int)
    S_mask = np.isin(C_idx, np.unique(C_idx))  # basically all, but keeps shape explicit

    # parent candidates for effect
    parents = [j for j in candidates if j != effect]
    if len(parents) == 0:
        return 0.0

    dep_ctx = []  # d_i^c per context

    for c in np.unique(C_idx):
        mask_c = (C_idx == c) & S_mask
        n_c = int(mask_c.sum())
        if n_c < min_n_ctx:
            continue

        Xc = X_all[mask_c, :]
        y_c = Xc[:, effect]

        # regress on ALL parents jointly
        X_parents_c = Xc[:, parents]
        # safeguard: if numeric issues or no variance
        if X_parents_c.shape[0] < 2 or X_parents_c.shape[1] == 0:
            continue

        R_c = krr_residuals(y_c, X_parents_c, lam=krr_lam, sigma=krr_sigma)

        # pairwise HSIC(R_c, X_j_c) for each parent j
        hsics = []
        for j in parents:
            Xj_c = Xc[:, j].reshape(-1, 1)
            h = hsic_unbiased(R_c, Xj_c)
            if np.isfinite(h) and h >= 0.0:
                hsics.append(h)

        if len(hsics) == 0:
            continue

        if agg_parents == "max":
            d_c = float(np.max(hsics))
        elif agg_parents == "mean":
            d_c = float(np.mean(hsics))
        else:
            raise ValueError(f"Unknown agg_parents='{agg_parents}' (use 'max' or 'mean').")

        dep_ctx.append(d_c)

    if len(dep_ctx) == 0:
        return 0.0

    if agg_contexts == "mean":
        dep_score = float(np.mean(dep_ctx))
    elif agg_contexts == "max":
        dep_score = float(np.max(dep_ctx))
    else:
        raise ValueError(f"Unknown agg_contexts='{agg_contexts}' (use 'mean' or 'max').")

    return max(0.0, dep_score)

def _rbf_kernel_rows(X, sigma=None):
    """
    X: (n, d)
    Returns K: (n, n) with RBF kernel.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    # pairwise squared distances
    XX = np.sum(X * X, axis=1, keepdims=True)
    D2 = XX + XX.T - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)

    if sigma is None:
        # median heuristic
        vals = D2[np.triu_indices(n, k=1)]
        med2 = np.median(vals) if vals.size > 0 else 1.0
        if med2 <= 0:
            med2 = 1.0
        sigma = np.sqrt(0.5 * med2)

    K = np.exp(-D2 / (2.0 * sigma ** 2))
    return K

def hsic_gamma_pvalue(x, y, sigma_x=None, sigma_y=None, eps=1e-12):
    """
    HSIC test with γ-approximation.
    x: (n, dx) or (n,)
    y: (n, dy) or (n,)
    Returns a p-value in [0,1].
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n = x.shape[0]
    if n < 5:
        return 1.0

    K = _rbf_kernel_rows(x, sigma_x)
    L = _rbf_kernel_rows(y, sigma_y)

    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    # biased HSIC estimate (fast, OK for ranking)
    hsic = np.sum(Kc * Lc) / (n - 1) ** 2

    # moments under H0 (Gretton thesis approximation)
    # use off-diagonal entries only
    K_off = K - np.diag(np.diag(K))
    L_off = L - np.diag(np.diag(L))
    KL = K_off * L_off

    m = n * (n - 1)
    mu = np.sum(KL) / m
    var = 2.0 * np.sum((KL - mu) ** 2) / (m ** 2)

    if var <= eps or mu <= eps or not np.isfinite(hsic):
        return 1.0

    alpha = mu ** 2 / (var + eps)
    beta = (var + eps) / (mu + eps)

    # p-value = 1 - F_gamma(hsic; alpha, beta)
    p = 1.0 - gamma.cdf(hsic, alpha, scale=beta)
    print( gamma.cdf(hsic, alpha, scale=beta))
    if not np.isfinite(p):
        p = 1.0
    p = max(0.0, min(1.0, float(p)))
    return p
def resit_dep_pvalue_per_context(
    effect,
    candidates,
    X_all,
    C_idx,
    indep_test_fun=hsic_gamma_pvalue,
    krr_lam=1e-2,
    krr_sigma=None,
    min_n_ctx=20,
    alpha_corr="bonferroni",
    eps=1e-9,
):
    """
    RESIT-style dependence score across contexts, joint over all parents.

    For node `effect`:
      - For each context c with >= min_n_ctx samples:
          * regress X_effect^(c) on X_candidates^(c)
          * compute residuals R_c
          * run independence test indep_test_fun(R_c, X_parents_c) -> p_c
      - Combine {p_c} across contexts by a multiple-testing correction.

    Returns:
      p_resit in [0,1], larger = weaker dependence of residuals on parents.
    """
    # if no parents to regress on, treat as (trivially) independent
    if len(candidates) == 0:
        return 1.0

    y_all = X_all[:, effect]
    contexts = np.unique(C_idx)

    p_list = []

    for c in contexts:
        idx_c = np.where(C_idx == c)[0]
        n_c = idx_c.size
        if n_c < min_n_ctx:
            continue

        y_c = y_all[idx_c]
        X_par_c = X_all[idx_c][:, candidates]

        # regress on *all* parents in S\{i} in this context
        R_c = krr_residuals(y_c, X_par_c, lam=krr_lam, sigma=krr_sigma)

        # independence test between residuals and full parent vector
        p_c = float(indep_test_fun(R_c.reshape(-1, 1), X_par_c))
        #print(p_c, R_c.shape)
        if np.isfinite(p_c):
            p_c = max(0.0, min(1.0, p_c))
            p_list.append(p_c)

    if not p_list:
        # no usable contexts -> no evidence of dependence
        return 1.0

    p_list = np.asarray(p_list, float)

    # multiple-testing correction across contexts
    m = p_list.size
    if alpha_corr == "bonferroni":
        p_comb = float(np.min(p_list) * m)
    elif alpha_corr == "holm":
        # Holm step-down: simple version using sorted p's
        p_sorted = np.sort(p_list)
        p_holm = np.min((m - np.arange(m)) * p_sorted)
        p_comb = float(p_holm)
    else:
        # no correction: use min p
        p_comb = float(np.min(p_list))
    p_comb = max(0.0, min(1.0, p_comb))
    return p_comb

def hsic_resit_dependence(effect, candidates, X_all,
                          krr_lam=1e-2, krr_sigma=None, eps=1e-9):
    """
    RESIT-style dependence term for node `effect` given candidate parents `candidates` (S \ {effect}).

    Steps:
      1) Regress X_effect on X_candidates (all-at-once).
      2) Compute HSIC between residuals and each X_j, j in candidates.
      3) Return a scalar dependence score, here max_j HSIC(residuals, X_j).

    Smaller value => more source-like (weaker dependence on the remaining variables).
    """
    if len(candidates) == 0:
        return 0.0

    y = X_all[:, effect]
    n = y.shape[0]
    X_par = X_all[:, candidates] if len(candidates) > 0 else np.zeros((n, 0))

    # residuals when regressing on all other candidates
    R = krr_residuals(y, X_par, lam=krr_lam, sigma=krr_sigma)

    deps = []
    for j in candidates:
        Xj = X_all[:, j].reshape(-1, 1)
        hs = hsic_unbiased(R, Xj)
        if np.isfinite(hs):
            deps.append(max(0.0, float(hs)))

    if not deps:
        return 0.0
    return float(max(deps))

def edge_improvement_combo(source, target, G, X_all, C_idx,
                           lam_edge=0.9,
                           krr_lam=1e-2,
                           krr_sigma=None,
                           mmd_sigma=None,
                           eps=1e-9):
    """
    Combo 'gain' of adding edge source -> target to graph G.

    Uses:
      - MMD of residuals across contexts (invariance),
      - HSIC of residuals vs source (independence).

    Returns:
      gain, r_mmd, r_hsic, D_base, D_with
    """
    # current parents of target (without source)
    parents = list(G.predecessors(target))
    if source in parents:
        parents.remove(source)

    # --- MMD part: residual invariance across contexts ---
    D_base = discrepancy_mmd(target, parents, X_all, C_idx,
                             krr_lam=krr_lam, krr_sigma=krr_sigma, mmd_sigma=mmd_sigma)
    if (not np.isfinite(D_base)) or D_base <= 0.0:
        D_with = D_base
        r_mmd = 0.0
    else:
        D_with = discrepancy_mmd(target, parents + [source], X_all, C_idx,
                                 krr_lam=krr_lam, krr_sigma=krr_sigma, mmd_sigma=mmd_sigma)
        if not np.isfinite(D_with):
            D_with = D_base
        delta = max(0.0, D_base - D_with)
        r_mmd = delta / (D_base + eps)

    # --- HSIC part: residual independence of source given parents ---
    y = X_all[:, target]
    n = X_all.shape[0]

    # residuals without source
    Xp_base = X_all[:, parents] if parents else np.zeros((n, 0))
    R0 = krr_residuals(y, Xp_base, lam=krr_lam, sigma=krr_sigma)
    Xu = X_all[:, source].reshape(-1, 1)
    hs0 = hsic_unbiased(R0, Xu)

    # residuals with source
    Xp_with = X_all[:, parents + [source]]
    R1 = krr_residuals(y, Xp_with, lam=krr_lam, sigma=krr_sigma)
    hs1 = hsic_unbiased(R1, Xu)

    if (not np.isfinite(hs0)) or hs0 <= 0.0 or (not np.isfinite(hs1)):
        r_hsic = 0.0
    else:
        delta_h = max(0.0, hs0 - hs1)
        r_hsic = delta_h / (hs0 + eps)

    gain = lam_edge * r_mmd + (1.0 - lam_edge) * (1-r_hsic)
    return float(gain), float(r_mmd), float(r_hsic), float(D_base), float(D_with)

def gain_min_from_sample_size(C_idx, c_edge=1.0, mode="min_ctx"):
    """
    C_idx: array of context labels per sample.
    mode: 'min_ctx' (default) or 'avg_ctx'
    """
    labels, counts = np.unique(C_idx, return_counts=True)
    if mode == "min_ctx":
        n_eff = counts.min()
    elif mode == "avg_ctx":
        n_eff = counts.mean()
    else:
        raise ValueError("mode must be 'min_ctx' or 'avg_ctx'")
    if n_eff <= 0:
        return 0.0
    return float(c_edge / np.sqrt(n_eff))

def hsic_parent_reduction(effect, candidates, X_all, krr_lam=1e-2, krr_sigma=None, eps=1e-9):
    """
    For node `effect`, compute HSIC-based parent explanation over candidate parents.
    Returns:
        H0_max: max_j hsic(R0, X_j)   (baseline dependence with no parents)
        dmax:   max_j (hs0_j - hs1_j) (best HSIC reduction by any single parent)
    """
    y = X_all[:, effect]
    n = y.shape[0]
    if len(candidates) == 0:
        return 0.0, 0.0

    # residuals with no parents (same for all j)
    R0 = krr_residuals(y, np.zeros((n, 0)), lam=krr_lam, sigma=krr_sigma)

    hs0_list = []
    red_list = []
    for j in candidates:
        Xj = X_all[:, j].reshape(-1, 1)
        hs0 = hsic_unbiased(R0, Xj)
        # residuals with parent {j}
        R1 = krr_residuals(y, Xj, lam=krr_lam, sigma=krr_sigma)
        hs1 = hsic_unbiased(R1, Xj)
        hs0_list.append(hs0)
        red_list.append(max(0.0, hs0 - hs1))  # reduction cannot be negative in our scoring

    H0_max = max(hs0_list) if hs0_list else 0.0
    dmax = max(red_list) if red_list else 0.0
    return float(H0_max), float(dmax)

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




def add_edges_combo_given_order(order, X_all, C_idx,
                                lam_edge=0.7, gain_min=0.05,
                                max_parents=None,
                                krr_lam=1e-2, krr_sigma=None, mmd_sigma=None, eps=1e-9):
    """
    Given a topological order, build a DiGraph by greedy per-target parent selection.
    For each target t in order:
      - parents can only be nodes before t in `order`
      - iteratively add the parent with max combo gain until gain < gain_min or max_parents reached
    Returns: (G, added_edges)
    """
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(order)
    pos = {v: i for i, v in enumerate(order)}
    added_edges = []

    for t in order:
        P_t = []
        while True:
            # candidate parents: all nodes before t in order and not already a parent
            cand = [u for u in order if pos[u] < pos[t] and u not in P_t]
            if not cand:
                break

            best_u = None
            best_gain = 0.0
            best_stats = None

            for u in cand:
                r_mmd, r_hsic, D_base, D_with = edge_improvement_combo(
                    u, t, G, X_all, C_idx,
                    krr_lam=krr_lam, krr_sigma=krr_sigma, mmd_sigma=mmd_sigma, eps=eps
                )
                gain = lam_edge * r_mmd + (1.0 - lam_edge) * r_hsic
                if gain > best_gain:
                    best_gain = gain
                    best_u = u
                    best_stats = (r_mmd, r_hsic, D_base, D_with)

            if best_u is None or best_gain < gain_min:
                break
            if max_parents is not None and len(P_t) >= max_parents:
                break

            G.add_edge(best_u, t)
            P_t.append(best_u)
            r_mmd, r_hsic, D_base, D_with = best_stats
            added_edges.append((best_u, t, best_gain, r_mmd, r_hsic, D_base, D_with))

    return G, added_edges


def add_outgoing_from_source_combo(source, order, G, X_all, C_idx,
                                   lam_edge=0.7, gain_min=0.05,
                                   max_parents=None,
                                   krr_lam=1e-2, krr_sigma=None, mmd_sigma=None, eps=1e-9):
    """
    For a given `source`, examine all later nodes in `order` as potential targets.
    Add edge source->target if combo gain >= gain_min and target indegree <= max_parents.
    Returns list of added edges [(source, target), ...].
    """
    pos = {v: i for i, v in enumerate(order)}
    if source not in pos:
        raise ValueError(f"source {source} not in order")

    added = []
    for target in order:
        if target == source:
            continue
        # enforce acyclicity: only allow edges forward in order
        if pos[target] <= pos[source]:
            continue

        parents = list(G.predecessors(target))
        if max_parents is not None and len(parents) >= max_parents:
            continue

        r_mmd, r_hsic, D_base, D_with = edge_improvement_combo(
            source, target, G, X_all, C_idx,
            krr_lam=krr_lam, krr_sigma=krr_sigma, mmd_sigma=mmd_sigma, eps=eps
        )
        gain = lam_edge * r_mmd + (1.0 - lam_edge) * r_hsic

        if gain >= gain_min:
            G.add_edge(source, target)
            added.append((source, target, gain, r_mmd, r_hsic, D_base, D_with))

    return added



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

def fit_citest_CONTEXTS(
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
            Z = p_y_given_x.sum() * dy
            if Z <= 0:
                # e.g. treat as uniform or return 0 score
                p_y_given_x[:] = 1.0 / (len(p_y_given_x) * dy)
            else:
                p_y_given_x /= Z

            #p_y_given_x = p_y_given_x / (p_y_given_x.sum() * dy)
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

def fit_fun_MIXED(
        mixing_type,
        X,
        covariates: list,
        target: int,
        resid=None,
        **params) -> [List, List, dict]:

    """ fits regression models using EM
    """
    if params.get("true_idls") is not None:
        true_idl = get_true_idl(params["true_idls"], covariates, target, params["t_A"])
    elif params.get("t_Z") is not None:
        true_idl = get_true_idl_Z(
            covariates, target, params["t_A"], params["t_Z"], params["t_n_Z"], X.shape[0])
    else: true_idl = None

    if params["oracle_Z"]:
        assert true_idl is not None
        true_idl, true_pproba, true_dict = conditional_mixture_known_assgn(
            X=X, node_i=target, pa_i=covariates, true_idl=true_idl, **params)
        return true_idl, true_pproba, true_dict

    range_k = range(1, params["k_max"] + 1) if not params["oracle_K"] else [len(np.unique(true_idl))]
    res_dict = fit_conditional_mixture(
        mty=mixing_type, X=X, node_i=target, pa_i=covariates, range_k=range_k, resid=resid, true_idl=true_idl,
        lg=params.get("lg", None))
    score = res_dict["bic"] #idl_dict.get("bic", 0)
    return score, res_dict



def fit_fun_CONTEXTS(
        X, pa, target,
        score_fun,  # callable: (Xtr, ytr, params_score_type_dict) -> (model, score_bits)
        alpha=0.05,  # significance thresh for no-hypercompression
        vb=0
):
    """ fits regression models per contexts and discovers a partition
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

