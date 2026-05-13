"""
methylcycle.utils.entropy
=========================
Compute per-region methylation entropy from unique-CpG count matrices.

The entropy metric is a measure of methylation heterogeneity within a
genomic region.  It is derived from binary (methylated / unmethylated)
CpG calls:

    p = (n_meth_cpgs + prior) / (n_total_cpgs + 2*prior)
    H = -p*log2(p) - (1-p)*log2(1-p)

Regions with fewer than ``min_abs`` unique CpGs are masked to NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_entropy_matrix(
    df_meth: pd.DataFrame,
    df_unmeth: pd.DataFrame,
    *,
    min_abs: int = 5,
    prior: float = 0.5,
) -> pd.DataFrame:
    """Compute Shannon entropy per region per sample.

    Parameters
    ----------
    df_meth:
        ``(n_regions × n_samples)`` counts of *unique* methylated CpG
        positions per region per sample.
    df_unmeth:
        Same shape, for unmethylated CpG positions.
    min_abs:
        Minimum total unique CpGs required; regions below this threshold
        are set to ``NaN``.
    prior:
        Laplace / add-constant smoothing (default 0.5 — Jeffreys prior).

    Returns
    -------
    pd.DataFrame
        Shannon entropy values, same shape as *df_meth*.
        Values lie in ``[0, 1]`` (bits, base-2).
    """
    n_cpgs = df_meth + df_unmeth                             # total unique CpGs

    p = (df_meth + prior) / (n_cpgs + 2 * prior)
    q = 1.0 - p

    with np.errstate(divide="ignore", invalid="ignore"):
        H = -p * np.log2(p) - q * np.log2(q)

    # Mask low-coverage regions
    H[n_cpgs < min_abs] = np.nan

    return H
