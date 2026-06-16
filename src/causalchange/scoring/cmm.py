from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigBase
from causalchange.core.results import CMMMixtureResult, CMMTargetMixtureResult
from causalchange.core.types import MixedSCMType
from causalchange.scoring.regression import fit_conditional_mixture
from causalchange.scoring.tabular import SCMScoreTabular


@dataclass(frozen=True)
class _CMMFit:
    score: float
    labels: np.ndarray
    responsibilities: np.ndarray
    component_weights: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SCMScoreCMM(SCMScoreTabular):
    def __init__(self, cfg: CausalChangeConfigBase):
        super().__init__(cfg)

        self.mix_type: MixedSCMType = cfg.mix_type
        self.k_max: int = int(self.score_params.get("k_max", 5))
        self.lambda_mix: float = float(self.score_params.get("lambda_mix", 1.0))
        self.hybrid_mixing: bool = bool(self.score_params.get("hybrid_mixing", True))

    def local_score(
        self,
        df: pd.DataFrame,
        effect: str,
        parents: tuple[str, ...],
    ) -> float:
        fit = self._fit_mixture_family(
            X=df,
            effect=effect,
            parents=parents,
        )
        return float(fit.score)

    def fit_final_mixture_components(
        self,
        X: pd.DataFrame,
        graph: nx.DiGraph,
    ) -> CMMMixtureResult:
        X = self._stringify_columns(X)

        target_components: dict[Any, CMMTargetMixtureResult] = {}
        diagnostics: dict[str, Any] = {
            "mode": "cmm_final_graph_mixtures",
            "failures": [],
        }

        for target in graph.nodes():
            target = str(target)
            parents = tuple(sorted((str(p) for p in graph.predecessors(target)), key=repr))

            try:
                target_components[target] = self._fit_final_target_mixture(
                    X=X,
                    target=target,
                    parents=parents,
                )
            except Exception as exc:
                diagnostics["failures"].append(
                    {
                        "target": target,
                        "parents": parents,
                        "error": repr(exc),
                    }
                )

        return CMMMixtureResult(
            target_components=target_components,
            global_labels=None,
            global_responsibilities=None,
            diagnostics=diagnostics,
        )

    def _fit_final_target_mixture(
        self,
        *,
        X: pd.DataFrame,
        target: Any,
        parents: tuple[Any, ...],
    ) -> CMMTargetMixtureResult:
        fit = self._fit_mixture_family(
            X=X,
            effect=str(target),
            parents=tuple(str(p) for p in parents),
        )

        labels = np.asarray(fit.labels, dtype=int)
        responsibilities = np.asarray(fit.responsibilities, dtype=float)

        component_weights = fit.component_weights
        if component_weights is None:
            component_weights = responsibilities.mean(axis=0)

        return CMMTargetMixtureResult(
            target=target,
            parents=parents,
            labels=labels.astype(int).tolist(),
            responsibilities=responsibilities.astype(float).tolist(),
            component_weights=np.asarray(component_weights, dtype=float).tolist(),
            score=float(fit.score),
            n_components=int(responsibilities.shape[1]) if responsibilities.ndim == 2 else None,
            diagnostics={
                "n_samples": int(len(labels)),
                "n_parents": int(len(parents)),
                **dict(fit.diagnostics),
            },
        )

    def _fit_mixture_family(
        self,
        *,
        X: pd.DataFrame,
        effect: str,
        parents: tuple[str, ...],
    ) -> _CMMFit:
        X = self._stringify_columns(X)
        self._ensure_bound(X)

        if self._edges is None:
            raise RuntimeError("CMM scorer is not bound. Call fit(...) first.")

        effect = str(effect)
        parents = tuple(str(p) for p in parents)

        j = self._col_index[effect]
        pa = [self._col_index[p] for p in parents]

        res = fit_conditional_mixture(
            mty=self.mix_type,
            X=self._edges.X,
            node_i=j,
            pa_i=pa,
            range_k=range(1, int(self.k_max) + 1),
            resid=None,
            true_idl=None,
            lg=None,
            vb=0,
        )

        score = float(res.get("bic", res.get("score", res.get("loss"))))

        labels = res.get("idl", res.get("labels", res.get("assignments")))
        responsibilities = res.get("pproba", res.get("responsibilities", res.get("posterior")))

        if labels is None and responsibilities is None:
            raise ValueError(
                "CMM backend did not return labels/responsibilities. "
                "Expected keys 'idl' and/or 'pproba' from fit_conditional_mixture()."
            )

        fit = self._make_cmm_fit(
            score=score,
            labels=labels,
            responsibilities=responsibilities,
            component_weights=None,
            diagnostics={
                key: value
                for key, value in res.items()
                if key not in {"idl", "labels", "assignments", "pproba", "responsibilities", "posterior"}
            },
            n_samples=int(X.shape[0]),
        )

        return fit

    def _make_cmm_fit(
        self,
        *,
        score: Any,
        labels: Any,
        responsibilities: Any,
        component_weights: Any,
        diagnostics: Any,
        n_samples: int,
    ) -> _CMMFit:
        if score is None:
            raise ValueError("CMM full result does not contain a score.")

        if labels is None and responsibilities is None:
            raise ValueError("CMM fit did not return labels or responsibilities.")

        if responsibilities is None:
            labels_arr = np.asarray(labels, dtype=int)
            n_components = int(labels_arr.max()) + 1 if labels_arr.size else 1
            responsibilities_arr = np.zeros((len(labels_arr), n_components), dtype=float)
            responsibilities_arr[np.arange(len(labels_arr)), labels_arr] = 1.0
        else:
            responsibilities_arr = np.asarray(responsibilities, dtype=float)
            if responsibilities_arr.ndim == 1:
                responsibilities_arr = responsibilities_arr.reshape(-1, 1)

            if labels is None:
                labels_arr = np.argmax(responsibilities_arr, axis=1).astype(int)
            else:
                labels_arr = np.asarray(labels, dtype=int)

        if len(labels_arr) != n_samples:
            raise ValueError(f"CMM labels length mismatch: got {len(labels_arr)}, expected {n_samples}.")

        if responsibilities_arr.shape[0] != n_samples:
            raise ValueError(
                f"CMM responsibilities row mismatch: got {responsibilities_arr.shape[0]}, expected {n_samples}."
            )

        if component_weights is None:
            component_weights_arr = responsibilities_arr.mean(axis=0)
        else:
            component_weights_arr = np.asarray(component_weights, dtype=float)

        if not isinstance(diagnostics, dict):
            diagnostics = {"raw_diagnostics": diagnostics}

        return _CMMFit(
            score=float(score),
            labels=labels_arr,
            responsibilities=responsibilities_arr,
            component_weights=component_weights_arr,
            diagnostics=dict(diagnostics),
        )
