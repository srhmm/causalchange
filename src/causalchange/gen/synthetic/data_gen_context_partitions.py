import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from src.causalchange.gen.generate import IvType, FunType, NoiseType


def _id(x): return x
def _quad(x): return x**2
def _cub(x): return x**3
def _exp(x): return np.exp(x)
def _log(x): return np.log(np.clip(x, 1e-6, None))
def _sin(x): return np.sin(x)

_FUN_LIB_ALL = {
    'lin':  [('lin',  _id)],
    'quad': [('quad', _quad)],
    'cub':  [('cub',  _cub)],
    'exp':  [('exp',  _exp)],
    'log':  [('log',  _log)],
    'sin':  [('sin',  _sin)],
    'mix':  [('lin', _id), ('quad', _quad), ('cub', _cub), ('exp', _exp), ('log', _log), ('sin', _sin)],
}

class DataGenContextPartitions(object):
    def __init__(self, params, graph, seed=0, vb=0):
        self.N = params["S"]
        self.M = params["N"]
        self.C = params["C"]
        self.fun_type = params["F"].value if isinstance(params["F"], FunType) else str(params["F"])
        self.noise_type = params["NS"].value if isinstance(params["NS"], NoiseType) else str(params["NS"])
        self.iv_type = params.get("IVT", IvType.COEF)
        self.iv_type = self.iv_type.value if isinstance(self.iv_type, IvType) else str(self.iv_type)
        self._graph = graph
        self.vb = vb
        self.rng = np.random.default_rng(seed)

        self.W_MIN, self.W_MAX = 0.5, 2.5
        self.B_MIN, self.B_MAX = -2.0, 2.0
        self.SIG_MIN, self.SIG_MAX = 1.0, 3.0
        self.SIG2_MIN, self.SIG2_MAX = 2.0, 5.0

        self.P_C = float(params.get("PC", 0.0))
        self.variant_flags = self.rng.random(self.M) < self.P_C

        self.K_default = min(3, max(1, self.C - 1)) - 1
        self.K_map = {}
        for i in range(self.M):
            if not self.variant_flags[i]:
                self.K_map[i] = 1
            else:
                K_param = int(params.get("K", self.K_default)) + 1
                K_param = max(1, min(self.C - 1, K_param))
                self.K_map[i] = K_param

        self.topo = list(nx.topological_sort(self._graph))

        self.partitions = {}
        self.mech = {}
        self._sample_mechanisms()

    def _random_partition_labels(self, C, K):
        perm = self.rng.permutation(C)
        labels = np.empty(C, dtype=int)
        for k in range(K):
            labels[perm[k]] = k
        for idx in perm[K:]:
            labels[idx] = int(self.rng.integers(0, K))
        return labels

    def _choose_fun_lib(self, fun_type):
        key = getattr(fun_type, "value", str(fun_type)).lower()
        return list(_FUN_LIB_ALL[key])

    def _draw_noise_vec(self, kind_vec, scale_vec):
        kind_vec = np.asarray(kind_vec)
        scale_vec = np.asarray(scale_vec, dtype=float)
        out = np.empty_like(scale_vec, dtype=float)

        m = (kind_vec == 'normal')
        if m.any():
            out[m] = self.rng.normal(0.0, scale_vec[m], size=m.sum())

        m = (kind_vec == 'unif')
        if m.any():
            a = np.sqrt(3) * scale_vec[m]
            out[m] = self.rng.uniform(-a, a, size=m.sum())

        m = (kind_vec == 'exp')
        if m.any():
            s = scale_vec[m]
            out[m] = self.rng.exponential(s, size=m.sum()) - s

        m = (kind_vec == 'gumbel')
        if m.any():
            s = scale_vec[m]
            # mean = loc + gamma*s with loc=0; subtract Euler–Mascheroni*scale to zero-center
            out[m] = self.rng.gumbel(loc=0.0, scale=s, size=m.sum()) - 0.5772156649015329 * s

        return out

    def _sample_mechanisms(self):
        self.mech.clear()
        self.partitions.clear()

        for i in range(self.M):
            K_i = self.K_map[i]
            part_i = np.zeros(self.C, dtype=int) if K_i == 1 else self._random_partition_labels(self.C, K_i)
            self.partitions[i] = part_i

            parents = list(self._graph.predecessors(i))
            p_idx = np.array(parents, dtype=int)
            fun_lib = self._choose_fun_lib(self.fun_type)

            iv_mode = self.iv_type
            if iv_mode == 'mix' and self.variant_flags[i]:
                iv_mode = self.rng.choice(['coef', 'shift', 'hard'])

            hard_iv_group = None
            if self.variant_flags[i] and iv_mode == 'hard' and K_i > 1:
                hard_iv_group = int(self.rng.integers(0, K_i))

            base_W = np.zeros(self.M)
            if len(parents):
                base_W[p_idx] = self.rng.uniform(self.W_MIN, self.W_MAX, size=len(parents))

            self.mech[i] = {}
            for g in range(K_i):
                if len(parents) > 0:
                    if self.fun_type == 'mix':
                        chosen = self.rng.choice(fun_lib, size=len(parents), replace=True)
                        f_names, f_funcs = zip(*chosen)
                    else:
                        f_names, f_funcs = zip(*([fun_lib[0]] * len(parents)))
                    f_names, f_funcs = list(f_names), list(f_funcs)
                else:
                    f_names, f_funcs = [], []

                W_col = np.zeros(self.M)
                if len(parents):
                    if (not self.variant_flags[i]) or K_i == 1:
                        W_col[p_idx] = base_W[p_idx]
                    else:
                        if iv_mode == 'coef':
                            W_col[p_idx] = self.rng.uniform(self.W_MIN, self.W_MAX, size=len(parents))
                        elif iv_mode in ('shift', 'hard'):
                            W_col[p_idx] = base_W[p_idx]

                bias = 0.0
                if self.variant_flags[i] and iv_mode == 'shift':
                    bias = float(self.rng.uniform(self.B_MIN, self.B_MAX))

                hard_iv = False
                if hard_iv_group is not None and g == hard_iv_group:
                    hard_iv = True
                    if len(parents):
                        W_col[p_idx] = 0.0

                ctx_in_group = np.where(self.partitions[i] == g)[0]

                sigma = self.rng.uniform(self.SIG_MIN, self.SIG_MAX, size=len(ctx_in_group))
                if self.variant_flags[i] and K_i > 1 and self.rng.random() < 0.5:
                    sigma = self.rng.uniform(self.SIG2_MIN, self.SIG2_MAX, size=len(ctx_in_group))

                if self.noise_type == 'mix':
                    all_types = ['normal', 'unif', 'exp', 'gumbel']
                    noise_kind_per_context = self.rng.choice(all_types, size=self.C, replace=True)
                else:
                    noise_kind_per_context = np.array([self.noise_type] * self.C, dtype=object)

                self.mech[i][g] = {
                    "parents": parents,
                    "W_col": W_col,
                    "bias": bias,
                    "f_funcs": f_funcs,
                    "f_names": f_names,
                    "ctx_idx": ctx_in_group,
                    "sigma": sigma,
                    "noise_kind": noise_kind_per_context,
                    "hard_iv": hard_iv,
                }

            if self.vb >= 1:
                print(f"[setup] Node {i}: parents={parents}, variant={bool(self.variant_flags[i])}, "
                      f"K={K_i}, IV={iv_mode}, partition={self.partitions[i].tolist()}")

    def gen_X(self):
        S = self.N
        M = self.M
        C = self.C
        X = np.zeros((S, M), dtype=float)

        reps = int(np.ceil(S / C))
        C_idx = np.tile(np.arange(C), reps)[:S]
        self.rng.shuffle(C_idx)
        if self.vb >= 1:
            counts = {int(c): int((C_idx == c).sum()) for c in range(C)}
            print(f"[gen] Sampled context counts: {counts}")

        for i in self.topo:
            parents = self.mech[i][0]["parents"]

            if len(parents) == 0:
                sig_per_sample = np.zeros(S)
                kind_per_sample = np.empty(S, dtype=object)
                for g, bundle in self.mech[i].items():
                    ctx = bundle["ctx_idx"]
                    sigma_map = {int(c): float(s) for c, s in zip(ctx, bundle["sigma"])}
                    for c in ctx:
                        m = (C_idx == c)
                        if not np.any(m):
                            continue
                        sig_per_sample[m] = sigma_map[int(c)]
                        kind_per_sample[m] = bundle["noise_kind"][int(c)]
                X[:, i] = self._draw_noise_vec(kind_per_sample, sig_per_sample)
                continue

            x_i = np.zeros(S)

            for g, bundle in self.mech[i].items():
                ctx = bundle["ctx_idx"]
                if len(ctx) == 0:
                    continue
                mask_ctx = np.isin(C_idx, ctx)
                if not np.any(mask_ctx):
                    continue

                W_col = bundle["W_col"]
                bias = bundle["bias"]
                p_idx = np.array(parents, dtype=int)

                if len(p_idx):
                    f_funcs = bundle["f_funcs"]
                    T = np.zeros((mask_ctx.sum(), len(p_idx)))
                    for k, j in enumerate(p_idx):
                        T[:, k] = f_funcs[k](X[mask_ctx, j])
                    contrib = T @ W_col[p_idx] + bias
                else:
                    contrib = bias

                sig_vec = np.zeros(mask_ctx.sum())
                kind_vec = np.empty(mask_ctx.sum(), dtype=object)
                sigma_map = {int(c): float(s) for c, s in zip(ctx, bundle["sigma"])}
                idx_local = np.where(mask_ctx)[0]
                for c in ctx:
                    m_local = (C_idx[mask_ctx] == c)
                    if not np.any(m_local):
                        continue
                    sig_vec[m_local] = sigma_map[int(c)]
                    kind_vec[m_local] = bundle["noise_kind"][int(c)]

                noise = self._draw_noise_vec(kind_vec, sig_vec)
                x_i[mask_ctx] = contrib + noise

            X[:, i] = x_i

        return X, C_idx

    def plot_X(
            self,
            X=None,
            C_idx=None,
            s=4,
            alpha=0.5,
            figsize_cell=(2.0, 1.6),
            cmap='tab10',
            hist_bins=30,
            include_sources=True
    ):
        """
        For each node i:
          - If i has parents, plot a grid with rows = parents and columns = contexts.
            Each cell is a scatter: X[parent] (x-axis) vs X[i] (y-axis) for that context.
          - If i has no parents and include_sources=True, plot a single-row grid of per-context
            marginal distributions (histograms) of X[i] across contexts.

        Colors are assigned by the node's true mechanism groups (same color for all contexts in a group).
        The title shows the partition as sets of contexts, e.g. {0,1}, {2,3}.
        """
        if X is None or C_idx is None:
            X, C_idx = self.gen_X()

        figs = []
        C = self.C

        for i in self.topo:
            parents = self.mech[i][0]["parents"]

            part_i = np.asarray(self.partitions.get(i, np.zeros(C, dtype=int)), dtype=int)
            groups_sorted = sorted(np.unique(part_i).tolist())
            group_to_ctx = {g: sorted([c for c in range(C) if part_i[c] == g]) for g in groups_sorted}
            part_sets_str = ", ".join("{" + ",".join(map(str, ctxs)) + "}" for g, ctxs in group_to_ctx.items())

            cmap_obj = plt.get_cmap(cmap)
            group_colors = {g: cmap_obj(k / max(1, len(groups_sorted) - 1)) for k, g in enumerate(groups_sorted)}

            if len(parents) == 0:
                if not include_sources:
                    continue
                nrows, ncols = 1, C
                fig, axes = plt.subplots(
                    nrows, ncols,
                    figsize=(figsize_cell[0] * ncols, figsize_cell[1] * nrows),
                    squeeze=False, sharex=True
                )
                x_min, x_max = np.min(X[:, i]), np.max(X[:, i])
                pad = 0.05 * max(1e-9, x_max - x_min)
                bins = np.linspace(x_min - pad, x_max + pad, hist_bins + 1)

                for c in range(C):
                    ax = axes[0, c]
                    mask = (C_idx == c)
                    g_c = int(part_i[c])
                    color_c = group_colors[g_c]
                    if np.any(mask):
                        ax.hist(X[mask, i], bins=bins, density=True, alpha=0.8, color=color_c)
                    if c == 0:
                        ax.set_ylabel(f"dens(X[{i}])")
                    ax.set_title(f"ctx {c}", fontsize=9)
                    ax.set_xlabel(f"X[{i}]")
                handles = [
                    plt.Line2D([0], [0], marker='s', linestyle='', markersize=6, color=group_colors[g], alpha=alpha)
                    for g in groups_sorted]
                labels = ["{" + ",".join(map(str, group_to_ctx[g])) + "}" for g in groups_sorted]
                fig.legend(handles, labels, title="Groups (contexts)", loc="upper right", bbox_to_anchor=(1.0, 1.02))
                fig.suptitle(f"Source {i} | Partition: {part_sets_str}")
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                figs.append(fig)
                continue

            nrows = len(parents)
            ncols = C
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(figsize_cell[0] * ncols, figsize_cell[1] * nrows),
                squeeze=False, sharex=False, sharey='row'
            )

            for r, j in enumerate(parents):
                for c in range(C):
                    ax = axes[r, c]
                    mask = (C_idx == c)
                    g_c = int(part_i[c])
                    color_c = group_colors[g_c]
                    if np.any(mask):
                        ax.scatter(X[mask, j], X[mask, i], s=s, alpha=alpha, color=color_c)
                    if r == 0:
                        ax.set_title(f"ctx {c}", fontsize=9)
                    if c == 0:
                        ax.set_ylabel(f"X[{i}]  (pa {j})")
                    if r == nrows - 1:
                        ax.set_xlabel(f"X[{j}]")

            handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=6, color=group_colors[g], alpha=alpha)
                       for g in groups_sorted]
            labels = ["{" + ",".join(map(str, group_to_ctx[g])) + "}" for g in groups_sorted]
            fig.legend(handles, labels, title="Groups (contexts)", loc="upper right", bbox_to_anchor=(1.0, 1.02))

            fig.suptitle(f"Effect {i} | Partition: {part_sets_str}")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            figs.append(fig)

        return figs
