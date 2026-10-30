"""
methylcycle.models.iz_features
==============================
Contrast features between replication-timing strata of initiation zones (IZs).

For every cell, IZs are grouped into ``early``, ``mid`` and ``late`` classes and
eight statistics are contrasted between the three pairs (late-early, mid-early,
late-mid), giving 24 features. A difference or ratio between two strata of the same
cell cancels its global methylation level, its depth and the coverage unit, which
keeps the features comparable across protocols.

All computations use numpy. In particular the fraction below the reference
(``fb``) is computed in numpy: computing it with ``DataFrame.lt(Series, axis=1)``
sends a zero-stride broadcast to numexpr, and numexpr <= 2.10.2 with more than one
thread returns wrong comparisons for tables larger than 1e6 elements.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CLASES: List[str] = ["early", "mid", "late"]
PARES = [("late", "early"), ("mid", "early"), ("late", "mid")]
MIN_COV: int = 3

#: The nine IZ features used by the stage 2 of the cascade.
CIEGO9: List[str] = [
    "dmed_late_early", "dfb_late_early", "dmed_mid_early", "dfb_mid_early",
    "rsd_mid_early", "riqr_late_mid", "ddet_mid_early", "rdens_mid_early",
    "dldet_late_mid",
]


def _mascara(m: pd.DataFrame, u: pd.DataFrame, min_cov: int):
    """Methylation fraction, NaN where coverage < min_cov; and the coverage."""
    M = np.asarray(m, dtype=np.float64)
    C = M + np.asarray(u, dtype=np.float64)
    P = np.full(C.shape, np.nan)
    ok = C >= min_cov
    P[ok] = M[ok] / C[ok]
    return P, C


def _ranks_in_cell(P: np.ndarray) -> np.ndarray:
    """Rank of each IZ inside its own cell (column), scaled to (0, 1)."""
    R = np.full(P.shape, np.nan)
    for j in range(P.shape[1]):
        col = P[:, j]
        ok = np.isfinite(col)
        n = int(ok.sum())
        if n >= 2:
            R[ok, j] = (rankdata(col[ok]) - 0.5) / n
    return R


def _logit(p, n):
    p = np.clip((np.asarray(p, dtype=np.float64) * n + 0.5) / (n + 1.0), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def iz_contrast_features(
    m: pd.DataFrame,
    u: pd.DataFrame,
    iz_classes: pd.Series,
    *,
    min_cov: int = MIN_COV,
    features: List[str] | None = None,
) -> pd.DataFrame:
    """IZ contrast features, one row per cell.

    Parameters
    ----------
    m, u:
        ``(n_IZs × n_cells)`` methylated / unmethylated counts (index = IZ region id).
    iz_classes:
        Series mapping IZ region id -> ``early`` / ``mid`` / ``late``.
    min_cov:
        Minimum coverage of an IZ in a cell for its methylation to be used.
    features:
        Columns to return (default: all 24).

    Returns
    -------
    pd.DataFrame
        ``(n_cells × n_features)``.
    """
    cls = iz_classes.reindex(m.index).to_numpy()
    rows: Dict[str, np.ndarray] = {c: np.flatnonzero(cls == c) for c in CLASES}
    for c in CLASES:
        if len(rows[c]) == 0:
            raise ValueError(f"no IZ of class '{c}' among the regions of the input")

    P, C = _mascara(m, u, min_cov)
    R = _ranks_in_cell(P)
    with np.errstate(all="ignore"):
        med = {c: np.nanmedian(P[rows[c]], axis=0) for c in CLASES}
        q25 = {c: np.nanpercentile(P[rows[c]], 25, axis=0) for c in CLASES}
        q75 = {c: np.nanpercentile(P[rows[c]], 75, axis=0) for c in CLASES}
        sd = {c: np.nanstd(P[rows[c]], axis=0, ddof=1) for c in CLASES}
        rk = {c: np.nanmean(R[rows[c]], axis=0) for c in CLASES}
    det = {c: (C[rows[c]] >= min_cov).mean(axis=0) for c in CLASES}
    dens = {c: C[rows[c]].sum(axis=0) / len(rows[c]) for c in CLASES}
    iqr = {c: q75[c] - q25[c] for c in CLASES}

    # equal-weight reference: mean of the three class medians (not the pooled median,
    # which the early class, ~81 % of IZs, would dominate)
    med_eq = np.mean([med[c] for c in CLASES], axis=0)
    fb = {c: (P[rows[c]] < med_eq[None, :]).mean(axis=0) for c in CLASES}   # NaN counts as False

    def ratio(a, b):
        b = np.where(b == 0, np.nan, b)
        return a / b

    F = {}
    for a, b in PARES:
        F[f"dmed_{a}_{b}"] = med[a] - med[b]
        F[f"drk_{a}_{b}"] = rk[a] - rk[b]
        F[f"dfb_{a}_{b}"] = fb[a] - fb[b]
        F[f"riqr_{a}_{b}"] = ratio(iqr[a], iqr[b])
        F[f"rsd_{a}_{b}"] = ratio(sd[a], sd[b])
        F[f"ddet_{a}_{b}"] = det[a] - det[b]
        F[f"dldet_{a}_{b}"] = _logit(det[a], len(rows[a])) - _logit(det[b], len(rows[b]))
        F[f"rdens_{a}_{b}"] = ratio(dens[a], dens[b])

    out = pd.DataFrame(F, index=m.columns).replace([np.inf, -np.inf], np.nan)
    return out[features] if features is not None else out
