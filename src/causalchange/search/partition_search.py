import numpy as np
import networkx as nx
from functools import lru_cache

import numpy as np
import networkx as nx
from functools import lru_cache

from causallearn.utils.cit import KCI

def partition_search_constraintbased(
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

