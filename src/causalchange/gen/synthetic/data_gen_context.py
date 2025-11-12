import numpy as np
import networkx as nx
from enum import Enum

import matplotlib.pyplot as plt

from src.causalchange.gen.generate import IvType, FunType, NoiseType

from sklearn.neighbors import KernelDensity

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

class DataGenContext(object):
    def __init__(self, params, graph, seed=0, vb=0):
        self.params = params  #f reference
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

    def _make_intervention_partition(self, C, m):
        # returns labels in {0..m}, where 0 is baseline group (non-intervened contexts)
        # and groups 1..m are singleton groups for the chosen contexts
        labels = np.zeros(C, dtype=int)
        if m <= 0:
            return labels, 1
        m = int(min(max(m, 0), C - 1))
        singles = self.rng.choice(np.arange(C), size=m, replace=False)
        # keep a stable order for display consistency
        singles = np.sort(singles)
        for k, c in enumerate(singles, start=1):
            labels[c] = k
        return labels, 1 + m

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

        use_interventions = ("Kmn" in self.__dict__.get("params", {}) or "Kmx" in self.__dict__.get("params", {}))
        # If params not stored, pull from constructor input
        if not use_interventions:
            # try to read from known place
            try:
                Kmn = int(self.params.get("Kmn", None)) if hasattr(self, "params") else None
                Kmx = int(self.params.get("Kmx", None)) if hasattr(self, "params") else None
                use_interventions = (Kmn is not None) or (Kmx is not None)
            except Exception:
                use_interventions = False

        # robustly fetch Kmn/Kmx (when present); clamp to [0, C-1]
        Kmn = int(self.params.get("Kmn", 0)) if use_interventions else None
        Kmx = int(self.params.get("Kmx", self.C - 1)) if use_interventions else None
        if use_interventions:
            Kmn = max(0, min(self.C - 1, Kmn))
            Kmx = max(Kmn, min(self.C - 1, Kmx))

        for i in range(self.M):
            parents = list(self._graph.predecessors(i))
            p_idx = np.array(parents, dtype=int)

            # Partition: interventions-as-singletons if requested; else fallback
            if not self.variant_flags[i]:
                part_i = np.zeros(self.C, dtype=int)
                K_i = 1
            else:
                if use_interventions:
                    m = int(self.rng.integers(Kmn, Kmx + 1))
                    part_i, K_i = self._make_intervention_partition(self.C, m)
                else:
                    # fallback to old K-map logic
                    K_i = self.K_map[i]
                    part_i = np.zeros(self.C, dtype=int) if K_i == 1 else self._random_partition_labels(self.C, K_i)

            self.partitions[i] = part_i

            # function library (per edge)
            fun_lib = self._choose_fun_lib(self.fun_type)
            if len(parents) > 0:
                if self.fun_type == 'mix':
                    chosen = self.rng.choice(fun_lib, size=len(parents), replace=True)
                    f_names, f_funcs = zip(*chosen)
                else:
                    f_names, f_funcs = zip(*([fun_lib[0]] * len(parents)))
                f_names, f_funcs = list(f_names), list(f_funcs)
            else:
                f_names, f_funcs = [], []

            # intervention type for this node
            iv_mode = self.iv_type
            if iv_mode == 'mix' and self.variant_flags[i]:
                iv_mode = self.rng.choice(['coef', 'shift', 'hard'])

            # base weights shared by baseline group (and SHIFT / HARD)
            base_W = np.zeros(self.M)
            if len(parents):
                base_W[p_idx] = self.rng.uniform(self.W_MIN, self.W_MAX, size=len(parents))

            self.mech[i] = {}

            # create groups 0..K_i-1
            for g in range(K_i):
                ctx_in_group = np.where(part_i == g)[0]

                # weights & bias per group (by iv_mode)
                W_col = np.zeros(self.M)
                bias = 0.0
                if len(parents):
                    if not self.variant_flags[i] or K_i == 1:
                        W_col[p_idx] = base_W[p_idx]
                    else:
                        if iv_mode == 'coef':
                            # singleton groups get fresh weights; baseline keeps base_W
                            if g == 0:
                                W_col[p_idx] = base_W[p_idx]
                            else:
                                W_col[p_idx] = self.rng.uniform(self.W_MIN, self.W_MAX, size=len(parents))
                        elif iv_mode == 'shift':
                            W_col[p_idx] = base_W[p_idx]
                            if g != 0:
                                bias = float(self.rng.uniform(self.B_MIN, self.B_MAX))
                        elif iv_mode == 'hard':
                            if g == 0:
                                W_col[p_idx] = base_W[p_idx]
                            else:
                                W_col[p_idx] = 0.0

                # noise per group/context (kept compatible with previous behavior)
                # scale
                sigma = self.rng.uniform(self.SIG_MIN, self.SIG_MAX, size=len(ctx_in_group))
                # optional scaling diversity
                if self.variant_flags[i] and K_i > 1 and self.rng.random() < 0.5:
                    sigma = self.rng.uniform(self.SIG2_MIN, self.SIG2_MAX, size=len(ctx_in_group))

                # kind
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
                    "hard_iv": (iv_mode == 'hard' and g != 0),
                }

            if self.vb >= 1:
                # partition pretty-print e.g. {0,1,2},{3},{4}
                groups_sorted = sorted(np.unique(part_i).tolist())
                group_to_ctx = {g: sorted([c for c in range(self.C) if part_i[c] == g]) for g in groups_sorted}
                part_sets_str = ", ".join("{" + ",".join(map(str, v)) + "}" for _, v in group_to_ctx.items())
                print(f"[setup] Node {i}: parents={parents}, variant={bool(self.variant_flags[i])}, "
                      f"groups={K_i}, IV={iv_mode}, partition={part_sets_str}")

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

    def plot_conditionals_under(self, G_alt, X=None, C_idx=None, measure='js', y_bins=256, figsize_row=(6.0, 2.2), cmap='viridis'):
        """
        Visual probe of 'changes-only' logic under an alternative graph G_alt.
        For each node i, we estimate per-context conditional densities p(X_i | parents) and
        compute a per-context discrepancy (default: JS divergence in bits) to a pooled reference.
        We show side-by-side bars: no parents vs parents from G_alt. Colors reflect discrepancy magnitude.

        Parameters
        ----------
        G_alt : networkx.DiGraph
            Graph whose parent sets define the 'with parents' conditioning.
        X, C_idx : optional
            If omitted, data are generated via self.gen_X().
        measure : {'js'}
            Discrepancy measure. (JS only here, easy to add others.)
        y_bins : int
            Number of grid points for 1D KDE over the target variable.
        figsize_row : (w, h)
            Figure size per node row.
        cmap : str
            Matplotlib colormap for bars.

        Returns
        -------
        figs : list of matplotlib.figure.Figure
        """
        if X is None or C_idx is None:
            X, C_idx = self.gen_X()

        def _scotts_bandwidth(A):
            A = np.asarray(A, dtype=float)
            n, d = A.shape
            if n <= 1:
                return 1.0
            sigma = np.std(A, axis=0, ddof=1) + 1e-12
            return float(np.power(n, -1.0 / (d + 4)) * np.mean(sigma))

        def _make_y_grid(y, bins=256, pad=0.05):
            y = np.asarray(y).ravel()
            y_min, y_max = float(np.min(y)), float(np.max(y))
            span = max(1e-9, y_max - y_min)
            y_min -= pad * span
            y_max += pad * span
            grid = np.linspace(y_min, y_max, bins)
            dy = grid[1] - grid[0]
            return grid, dy

        def _kde_conditional_1d(y_vec, X_par, y_grid):
            """
            Estimate p(y | x) by KDE ratio and average over samples in the context:
              p(y|x) ~= mean_i [ p(y,x_i) / p(x_i) ] on grid.
            Returns density on y_grid normalized to integrate to 1.
            """
            y_vec = np.asarray(y_vec).reshape(-1, 1)
            X_par = np.asarray(X_par) if X_par is not None else np.zeros((y_vec.shape[0], 0))
            Z = np.hstack([y_vec, X_par])  # joint samples [y, x]
            if len(y_vec) == 0:
                return np.ones_like(y_grid) / len(y_grid)

            bw_joint = _scotts_bandwidth(Z)
            kde_joint = KernelDensity(kernel='gaussian', bandwidth=bw_joint).fit(Z)

            if X_par.shape[1] > 0:
                bw_x = _scotts_bandwidth(X_par)
                kde_x = KernelDensity(kernel='gaussian', bandwidth=bw_x).fit(X_par)
            else:
                kde_x = None

            out = np.zeros_like(y_grid, dtype=float)
            for i in range(X_par.shape[0]):
                xi = X_par[i:i+1, :]
                if xi.shape[1] == 0:
                    yi_x_grid = np.stack([y_grid], axis=1)
                    log_p_yx = kde_joint.score_samples(yi_x_grid)
                    p = np.exp(log_p_yx)
                else:
                    yi_x_grid = np.hstack([y_grid.reshape(-1, 1), np.repeat(xi, repeats=y_grid.size, axis=0)])
                    log_p_yx = kde_joint.score_samples(yi_x_grid)
                    log_p_x = kde_x.score_samples(xi)[0]
                    p = np.exp(log_p_yx - log_p_x)
                out += p
            out = out / max(1, X_par.shape[0])
            dy = y_grid[1] - y_grid[0]
            out = np.clip(out, 1e-12, np.inf)
            out /= (out.sum() * dy)
            return out

        def _js_discrete(p, q, dy=1.0):
            p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
            p = np.clip(p, 1e-12, np.inf); q = np.clip(q, 1e-12, np.inf)
            p /= (p.sum() * dy); q /= (q.sum() * dy)
            m = 0.5 * (p + q)
            def _kl(a, b): return float(((a * (np.log2(a) - np.log2(b))).sum()) * dy)
            return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

        def _per_context_discrepancies(i, parents):
            y = X[:, i]
            if parents:
                X_par = X[:, parents]
            else:
                X_par = np.zeros((X.shape[0], 0))

            y_grid, dy = _make_y_grid(y, bins=y_bins)
            C = self.C
            dens = []
            weights = []
            for c in range(C):
                mask = (C_idx == c)
                yc = y[mask]
                Xc = X_par[mask]
                d = _kde_conditional_1d(yc, Xc, y_grid)
                dens.append(d)
                weights.append(int(mask.sum()))
            dens = np.stack(dens, axis=0)
            weights = np.asarray(weights, dtype=float)
            w = weights / max(1, weights.sum())
            ref = (dens.T @ w).T  # weighted average density
            js = np.array([_js_discrete(dens[c], ref, dy=dy) for c in range(self.C)], dtype=float)
            return js, dens, ref, y_grid, dy

        figs = []
        C = self.C
        for i in self.topo:
            pa_alt = list(G_alt.predecessors(i)) if G_alt is not None else []
            js_none, dens_none, ref_none, _, _ = _per_context_discrepancies(i, parents=[])
            js_alt,  dens_alt,  ref_alt,  _, _ = _per_context_discrepancies(i, parents=pa_alt)

            # figure: left = no parents, right = with parents; bars colored by value
            fig, axes = plt.subplots(1, 2, figsize=(figsize_row[0], figsize_row[1]), squeeze=False)
            axes = axes[0]

            vmax = max(js_none.max(), js_alt.max(), 1e-9)
            norm = plt.Normalize(vmin=0.0, vmax=vmax)
            cmap_obj = plt.get_cmap(cmap)

            # left panel
            colors_left = cmap_obj(norm(js_none))
            axes[0].bar(np.arange(C), js_none, color=colors_left, edgecolor='black', linewidth=0.3)
            axes[0].set_title(f"Node {i}: no parents", fontsize=10)
            axes[0].set_xlabel("context")
            axes[0].set_ylabel("JS divergence (bits)")
            axes[0].set_xticks(np.arange(C))

            # right panel
            colors_right = cmap_obj(norm(js_alt))
            axes[1].bar(np.arange(C), js_alt, color=colors_right, edgecolor='black', linewidth=0.3)
            axes[1].set_title(f"Node {i}: parents={pa_alt}", fontsize=10)
            axes[1].set_xlabel("context")
            axes[1].set_xticks(np.arange(C))

            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
            cbar.set_label("discrepancy (JS bits)")

            # annotate expected behavior quickly:
            # show partition-as-sets to correlate with bars
            part_i = self.partitions.get(i, np.zeros(C, dtype=int))
            groups_sorted = sorted(np.unique(part_i).tolist())
            group_to_ctx = {g: sorted([c for c in range(C) if part_i[c] == g]) for g in groups_sorted}
            part_sets_str = ", ".join("{" + ",".join(map(str, v)) + "}" for _, v in group_to_ctx.items())
            fig.suptitle(f"Conditionals for node {i} | true partition: {part_sets_str}", y=1.02, fontsize=11)

            fig.tight_layout()
            figs.append(fig)

        return figs
