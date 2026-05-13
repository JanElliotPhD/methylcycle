"""
methylcycle.utils.normalization
===============================
Per-sample normalisation routines used before model inference.

``iqr_normalize_per_sample``
    Used for the RT mean-methylation model.
    Formula: (x - median(sample)) / IQR(sample)

``zscore_normalize_per_sample``
    Used for the ChromHMM mean-methylation model.
    Formula: (x - mean(sample)) / std(sample)

Both functions operate on a **(n_regions × n_samples)** DataFrame and
return a DataFrame of the same shape.  NaN values are propagated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def iqr_normalize_per_sample(df: pd.DataFrame) -> pd.DataFrame:
    """IQR normalisation applied column-wise (per sample).

    Parameters
    ----------
    df:
        Shape ``(n_regions, n_samples)``.  Columns = samples, rows = regions.

    Returns
    -------
    pd.DataFrame
        Same shape and index / columns as *df*.
        Samples whose IQR is zero are returned as all-NaN (to avoid
        division by zero).
    """
    arr = df.values.astype(np.float64)
    medians = np.nanmedian(arr, axis=0)            # (n_samples,)
    q75 = np.nanpercentile(arr, 75, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    iqr = q75 - q25

    # Avoid division by zero: set IQR=0 columns to NaN
    zero_iqr = iqr == 0
    iqr = np.where(zero_iqr, np.nan, iqr)

    norm = (arr - medians) / iqr
    return pd.DataFrame(norm, index=df.index, columns=df.columns)


def zscore_normalize_per_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalisation applied column-wise (per sample).

    Parameters
    ----------
    df:
        Shape ``(n_regions, n_samples)``.

    Returns
    -------
    pd.DataFrame
        Same shape and index / columns as *df*.
        Samples with zero standard deviation are returned as all-NaN.
    """
    arr = df.values.astype(np.float64)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0, ddof=0)

    zero_std = stds == 0
    stds = np.where(zero_std, np.nan, stds)

    norm = (arr - means) / stds
    return pd.DataFrame(norm, index=df.index, columns=df.columns)
