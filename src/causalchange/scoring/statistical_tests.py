from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MechanismTestResult:
    same: bool
    pvalue: float
    method: str


class SCMEqualityTestKCI:
    """
    Test whether two subsamples share the same target mechanism.

    With parents:
        H0: E ⟂ Y | parents

    Without parents:
        H0: Y_A and Y_B have the same marginal distribution.

    High p-value means no evidence for a mechanism difference.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        min_samples: int = 5,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive.")

        self.alpha = float(alpha)
        self.min_samples = int(min_samples)

    def same_mechanism(
        self,
        *,
        sample_a: pd.DataFrame,
        sample_b: pd.DataFrame,
        target_col: str,
        parent_cols: list[str],
    ) -> MechanismTestResult:
        sample_a = self._clean_sample(sample_a, target_col, parent_cols)
        sample_b = self._clean_sample(sample_b, target_col, parent_cols)

        if len(sample_a) < self.min_samples or len(sample_b) < self.min_samples:
            return MechanismTestResult(
                same=False,
                pvalue=0.0,
                method="too_few_samples",
            )

        if not parent_cols:
            pvalue = self._mmd_or_ks_pvalue(
                sample_a[target_col].to_numpy(dtype=float),
                sample_b[target_col].to_numpy(dtype=float),
            )
            return MechanismTestResult(
                same=bool(pvalue > self.alpha),
                pvalue=float(pvalue),
                method="mmd" if self._has_hyppo() else "ks",
            )

        pvalue = self._kci_pvalue(
            sample_a=sample_a,
            sample_b=sample_b,
            target_col=target_col,
            parent_cols=parent_cols,
        )

        return MechanismTestResult(
            same=bool(pvalue > self.alpha),
            pvalue=float(pvalue),
            method="kci",
        )

    def _clean_sample(
        self,
        sample: pd.DataFrame,
        target_col: str,
        parent_cols: list[str],
    ) -> pd.DataFrame:
        cols = [*parent_cols, target_col]
        missing = [c for c in cols if c not in sample.columns]
        if missing:
            raise ValueError(f"Missing columns in mechanism sample: {missing}")

        out = sample[cols].copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(axis=0)
        return out

    def _kci_pvalue(
        self,
        *,
        sample_a: pd.DataFrame,
        sample_b: pd.DataFrame,
        target_col: str,
        parent_cols: list[str],
    ) -> float:
        try:
            from causallearn.utils.cit import CIT
        except ImportError as exc:
            raise ImportError(
                "KCI mechanism testing requires the optional dependency "
                "`causal-learn`. Install it with `pip install causal-learn`."
            ) from exc

        Xa = sample_a[parent_cols].to_numpy(dtype=float)
        Xb = sample_b[parent_cols].to_numpy(dtype=float)

        ya = sample_a[target_col].to_numpy(dtype=float).reshape(-1, 1)
        yb = sample_b[target_col].to_numpy(dtype=float).reshape(-1, 1)

        ea = np.zeros((len(sample_a), 1), dtype=float)
        eb = np.ones((len(sample_b), 1), dtype=float)

        # columns: E, parents..., target
        data = np.vstack(
            [
                np.hstack([ea, Xa, ya]),
                np.hstack([eb, Xb, yb]),
            ]
        )

        env_idx = 0
        target_idx = data.shape[1] - 1
        condition_set = tuple(range(1, 1 + len(parent_cols)))

        kci = CIT(data, "kci")
        pvalue = kci(env_idx, target_idx, condition_set)
        return float(pvalue)

    def _mmd_or_ks_pvalue(self, ya: np.ndarray, yb: np.ndarray) -> float:
        if self._has_hyppo():
            from hyppo.ksample import MMD

            _, pvalue = MMD().test(
                ya.reshape(-1, 1),
                yb.reshape(-1, 1),
            )
            return float(pvalue)

        from scipy.stats import ks_2samp

        _, pvalue = ks_2samp(ya, yb)
        return float(pvalue)

    def _has_hyppo(self) -> bool:
        try:
            import hyppo  # noqa: F401
        except ImportError:
            return False
        return True
