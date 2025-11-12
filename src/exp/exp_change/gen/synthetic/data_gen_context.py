import numpy as np
import networkx as nx

def _sinc(x):  # plain sinc: sin(x)/x with safe handling
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(x[nz]) / x[nz]
    return out

_NONLIN_LIB = [
    ("sq",   lambda x: x**2),
    ("cub",  lambda x: x**3),
    ("tanh", np.tanh),
    ("sinc", _sinc),
]

class DataGenContext(object):
    """
    Multi-context data generator:
      - fixed DAG (graph)
      - C discrete contexts
      - nodes optionally change mechanisms across contexts by partitioning contexts into K groups
    """
    def __init__(self, params, graph, seed=0, vb=0):
        self.N = params["S"]
        self.M = params["N"]
        self.C = params["C"]
        self.noise_type = params["NS"].value
        self._graph = graph
        self.vb = vb
        self.rng = np.random.default_rng(seed)

        self.W_MIN, self.W_MAX = 0.5, 2.5
        self.SIG_MIN, self.SIG_MAX = 1.0, 3.0
        self.SIG2_MIN, self.SIG2_MAX = 2.0, 5.0
        self.NOISE_SC_BASE = 1.0
        self.HARD_IV_PROB = 0.2
        self.NOISE_SCALE_PROB = 0.5

        # per-node "is variant?" coin using P_C
        self.P_C = float(params.get("PC", 0.0))
        self.variant_flags = self.rng.random(self.M) < self.P_C

        # number of mechanism groups per variant node
        # ensure 1..C-1; default min(3, C-1)
        self.K_default = min(3, max(1, self.C - 1)) - 1
        self.K_map = {}
        for i in range(self.M):
            if not self.variant_flags[i]:
                self.K_map[i] = 1  # invariant -> one group
            else:
                K_param = int(params.get("K", self.K_default)) + 1
                K_param = max(1, min(self.C - 1, K_param))
                self.K_map[i] = K_param

        # build topo order
        self.topo = list(nx.topological_sort(self._graph))

        # mechanism bank per node: for each node i
        #  - part_i: array length C mapping context -> mechanism group id in {0..K_i-1}
        #  - mech[i][g]: dict with 'W_col', 'f_funcs', 'f_names', 'sigma', flags
        self.partitions = {}
        self.mech = {}
        self._sample_mechanisms()

    def _random_partition_labels(self, C, K):
        """
        Randomly assign each of the C contexts into K non-empty groups (labels 0..K-1).
        """
        perm = self.rng.permutation(C)
        labels = np.empty(C, dtype=int)
        for k in range(K):
            labels[perm[k]] = k
        for idx in perm[K:]:
            labels[idx] = int(self.rng.integers(0, K))
        return labels

    def _sample_mechanisms(self):
        self.mech.clear()
        self.partitions.clear()

        for i in range(self.M):
            K_i = self.K_map[i]
            # partition contexts into K_i groups
            part_i = np.zeros(self.C, dtype=int) if K_i == 1 else self._random_partition_labels(self.C, K_i)
            self.partitions[i] = part_i

            # parents of i
            parents = list(self._graph.predecessors(i))
            p_idx = np.array(parents, dtype=int)

            self.mech[i] = {}
            # decide uniform vs normal per *context*, 50/50
            noise_is_normal = self.rng.random(self.C) < 0.5  # per context

            # pick one group for (optional) hard intervention and noise scaling
            hard_iv_group = None
            if self.variant_flags[i] and self.rng.random() < self.HARD_IV_PROB:
                hard_iv_group = int(self.rng.integers(0, K_i))
            noise_scale_group = None
            if self.variant_flags[i] and self.rng.random() < self.NOISE_SCALE_PROB:
                noise_scale_group = int(self.rng.integers(0, K_i))

            for g in range(K_i):
                # per (i,g) choose nonlinearity for each parent->i edge
                if len(parents) > 0:
                    chosen = self.rng.choice(_NONLIN_LIB, size=len(parents), replace=True)
                    f_names, f_funcs = zip(*chosen)
                    f_names, f_funcs = list(f_names), list(f_funcs)
                else:
                    f_names, f_funcs = [], []

                # weights only for existing edges j->i (positive per spec)
                W_col = np.zeros(self.M)
                if len(parents):
                    W_col[p_idx] = self.rng.uniform(self.W_MIN, self.W_MAX, size=len(parents))

                # hard intervention: zero inbound weights in this mechanism group
                hard_iv = False
                if hard_iv_group is not None and g == hard_iv_group:
                    hard_iv = True
                    if len(parents):
                        W_col[p_idx] = 0.0

                # sigma per context in this group (defaults in [1,3])
                ctx_in_group = np.where(self.partitions[i] == g)[0]
                sigma = self.rng.uniform(self.SIG_MIN, self.SIG_MAX, size=len(ctx_in_group))
                if noise_scale_group is not None and g == noise_scale_group:
                    sigma = self.rng.uniform(self.SIG2_MIN, self.SIG2_MAX, size=len(ctx_in_group))

                self.mech[i][g] = {
                    "parents": parents,
                    "W_col": W_col,             # length M (only parents nonzero)
                    "f_funcs": f_funcs,         # list aligned with 'parents'
                    "f_names": f_names,         # names aligned with 'parents'
                    "ctx_idx": ctx_in_group,    # contexts belonging to this group
                    "sigma": sigma,             # per-context sigma aligned with ctx_idx
                    "noise_is_normal": noise_is_normal,  # full C-length boolean
                    "hard_iv": hard_iv,
                }

            if self.vb >= 1:
                print(f"[setup] Node {i}: parents={parents}, variant={bool(self.variant_flags[i])}, "
                      f"K={K_i}, partition={self.partitions[i].tolist()}")
            if self.vb >= 2:
                for g, bundle in self.mech[i].items():
                    par_str = ", ".join([f"{p}({bundle['f_names'][k]})" for k, p in enumerate(bundle["parents"])]) \
                              if bundle["parents"] else "∅"
                    ctx_list = bundle["ctx_idx"].tolist()
                    w_show = [float(bundle["W_col"][p]) for p in bundle["parents"]]
                    print(f"  [setup]  Group g={g} | hard_iv={bundle['hard_iv']} | parents={par_str} | "
                          f"weights={w_show} | contexts={ctx_list}")
            if self.vb >= 3:
                for g, bundle in self.mech[i].items():
                    # per-context noise details
                    ctx = bundle["ctx_idx"]
                    # build readable tuples (c, sigma_c, dist)
                    details = []
                    for c, s in zip(ctx, bundle["sigma"]):
                        dist = "Normal" if bundle["noise_is_normal"][int(c)] else "Uniform"
                        details.append((int(c), float(s), dist))
                    print(f"    [setup]   g={g} noise per context: {details}")

    def _draw_noise(self, is_normal, scale, size=None):
        """
        Vectorized noise drawer.
        - is_normal: bool or boolean array (length = #samples)
        - scale: float or array matched to is_normal
        - size: ignored if arrays are provided; only used for scalar case
        """
        if np.isscalar(is_normal) and np.isscalar(scale):
            if is_normal:
                return self.rng.normal(0.0, self.NOISE_SC_BASE * scale, size=size)
            a = np.sqrt(3) * self.NOISE_SC_BASE * scale
            return self.rng.uniform(-a, a, size=size)

        is_normal = np.asarray(is_normal, dtype=bool)
        scale = np.asarray(scale, dtype=float)
        if is_normal.shape != scale.shape:
            raise ValueError(f"is_normal.shape {is_normal.shape} != scale.shape {scale.shape}")

        out = np.empty_like(scale, dtype=float)
        if is_normal.any():
            sc_n = self.NOISE_SC_BASE * scale[is_normal]
            out[is_normal] = self.rng.normal(0.0, sc_n, size=sc_n.shape[0])
        if (~is_normal).any():
            sc_u = np.sqrt(3) * self.NOISE_SC_BASE * scale[~is_normal]
            out[~is_normal] = self.rng.uniform(-sc_u, sc_u, size=sc_u.shape[0])
        return out

    def gen_X(self):
        """
        Returns:
          X : (S, M) array of samples
          C_idx : (S,) integer array of context labels in {0..C-1}
        """
        S = self.N
        M = self.M
        C = self.C
        X = np.zeros((S, M), dtype=float)

        # context label for each sample: roughly balanced
        reps = int(np.ceil(S / C))
        C_idx = np.tile(np.arange(C), reps)[:S]
        self.rng.shuffle(C_idx)
        if self.vb >= 1:
            counts = {int(c): int((C_idx == c).sum()) for c in range(C)}
            print(f"[gen] Sampled context counts: {counts}")

        # generate in topological order
        for i in self.topo:
            parents = self.mech[i][0]["parents"]  # same parent set across groups (DAG-fixed)

            if self.vb >= 1:
                print(f"[gen] Node {i} start | parents={parents}")

            if len(parents) == 0:
                # pure noise source per context
                sig_per_sample = np.zeros(S)
                is_norm_per_sample = np.zeros(S, dtype=bool)
                for g, bundle in self.mech[i].items():
                    ctx = bundle["ctx_idx"]
                    sigma_map = {int(c): float(s) for c, s in zip(ctx, bundle["sigma"])}
                    for c in ctx:
                        mask = (C_idx == c)
                        if not np.any(mask):
                            continue
                        sig_per_sample[mask] = sigma_map[int(c)]
                        is_norm_per_sample[mask] = bundle["noise_is_normal"][int(c)]
                X[:, i] = self._draw_noise(is_norm_per_sample, sig_per_sample, size=S)

                if self.vb >= 2:
                    print(f"  [gen] Node {i} is source; noise-only with per-context sigma applied.")
                continue

            # compute contribution from parents sample-wise
            x_i = np.zeros(S)

            # For each mechanism group g, apply its weights+nonlin to the samples whose context maps to g
            for g, bundle in self.mech[i].items():
                ctx = bundle["ctx_idx"]
                if len(ctx) == 0:
                    continue
                mask_ctx = np.isin(C_idx, ctx)
                if not np.any(mask_ctx):
                    continue

                W_col = bundle["W_col"]
                p_idx = np.array(parents, dtype=int)

                if self.vb >= 2:
                    par_eq = " + ".join(
                        [f"{W_col[p]:.3f}*{bundle['f_names'][k]}(X[{p}])" for k, p in enumerate(p_idx)]
                    ) if len(p_idx) else "0"
                    print(f"  [gen] Node {i} | group g={g} | hard_iv={bundle['hard_iv']} | "
                          f"contexts={ctx.tolist()} | eq: X[{i}] = {par_eq} + ε")

                # sum_j w_{ij} f_{ij}(X_j)
                if len(p_idx):
                    f_funcs = bundle["f_funcs"]
                    T = np.zeros((mask_ctx.sum(), len(p_idx)))
                    for k, j in enumerate(p_idx):
                        T[:, k] = f_funcs[k](X[mask_ctx, j])
                    contrib = T @ W_col[p_idx]
                else:
                    contrib = 0.0

                # noise per context for selected samples
                sig_vec = np.zeros(mask_ctx.sum())
                is_norm_vec = np.zeros(mask_ctx.sum(), dtype=bool)
                sigma_map = {int(c): float(s) for c, s in zip(ctx, bundle["sigma"])}
                # fill vectors per context
                idx_in_mask = np.where(mask_ctx)[0]
                for c in ctx:
                    m_local = (C_idx[mask_ctx] == c)
                    if not np.any(m_local):
                        continue
                    sig_vec[m_local] = sigma_map[int(c)]
                    is_norm_vec[m_local] = bundle["noise_is_normal"][int(c)]

                noise = self._draw_noise(is_norm_vec, sig_vec, size=mask_ctx.sum())
                x_i[mask_ctx] = contrib + noise

                if self.vb >= 3:
                    # show a tiny preview to avoid flooding
                    nprev = min(3, mask_ctx.sum())
                    prev_idx = np.where(mask_ctx)[0][:nprev]
                    print(f"    [gen] preview X[{i}] samples idx={prev_idx.tolist()} -> "
                          f"{x_i[prev_idx].round(3).tolist()}")

            X[:, i] = x_i

        return X, C_idx
