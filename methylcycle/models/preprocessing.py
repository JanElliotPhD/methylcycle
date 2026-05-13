"""
methylcycle.models.preprocessing
=================================
Pre-processing step applied to the feature matrix before model inference.

Rules (fixed, not configurable):

+-------------+--------+---------------------------------------------+
| Map         | Metric | Preprocessing                               |
+=============+========+=============================================+
| RT          | MEAN   | IQR per-sample  →  PCA transform            |
| RT          | ENTROPY| None                                        |
| ChromHMM    | MEAN   | Z-score per-sample                          |
| ChromHMM    | ENTROPY| None                                        |
| Custom      | *      | None (caller handles)                       |
+-------------+--------+---------------------------------------------+

The PCA object for the RT-mean pipeline and the ordered region list /
feature filter for ChromHMM are loaded from the model artefact directory
configured via :data:`methylcycle.models.paths`.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from methylcycle.models.maps import MapType
from methylcycle.models.feature_extraction import Metric
from methylcycle.utils.normalization import (
    iqr_normalize_per_sample,
    zscore_normalize_per_sample,
)


def preprocess(
    df_metric: pd.DataFrame,
    map_type: MapType,
    metric: Metric,
    *,
    pca_model=None,
    chromhmm_region_order: List[str] | None = None,
) -> pd.DataFrame:
    """Apply the appropriate pre-processing for a given map/metric combination.

    Parameters
    ----------
    df_metric:
        ``(n_regions × n_samples)`` feature DataFrame produced by
        :func:`~methylcycle.models.feature_extraction.extract_features`.
    map_type:
        The map used during feature extraction.
    metric:
        The metric used during feature extraction.
    pca_model:
        A fitted ``sklearn.decomposition.PCA`` (or compatible) object.
        Required for ``RT + MEAN``.  All PCA components are used.
    chromhmm_region_order:
        Ordered list of region IDs to subset/reorder the ChromHMM-mean
        feature matrix before inference.  Required for ``ChromHMM + MEAN``.

    Returns
    -------
    pd.DataFrame
        Pre-processed feature matrix ready for model inference.
        Shape may differ from input (PCA reduces dimensionality).

    Raises
    ------
    ValueError
        If required artefacts are not provided.
    """
    if map_type == MapType.RT and metric == Metric.MEAN:
        return _preprocess_rt_mean(df_metric, pca_model=pca_model)

    if map_type == MapType.CHROMHMM and metric == Metric.MEAN:
        return _preprocess_chromhmm_mean(
            df_metric, region_order=chromhmm_region_order
        )

    # RT-entropy, ChromHMM-entropy, Custom → pass through
    return df_metric.copy()


# ---------------------------------------------------------------------------
# Pipeline implementations
# ---------------------------------------------------------------------------

def _preprocess_rt_mean(
    df: pd.DataFrame,
    *,
    pca_model,
) -> pd.DataFrame:
    """IQR normalise per sample, then apply PCA.

    The input DataFrame has shape ``(n_regions × n_samples)``.
    PCA is fitted on samples → we transpose, transform, then transpose back
    so the output is ``(n_pcs × n_samples)``.

    Actually, sklearn PCA works on ``(n_samples × n_features)``, so:
        X = df.T  →  shape (n_samples, n_regions)
        X_norm = IQR per sample (i.e. per row of X → same as per column of df)
        X_pca  = pca_model.transform(X_norm)  →  (n_samples, n_pcs)
    Output DataFrame: ``(n_pcs × n_samples)`` with PC labels as index.
    """
    if pca_model is None:
        raise ValueError(
            "A fitted PCA model must be supplied for the RT + MEAN pipeline. "
            "Pass it via the 'pca_model' parameter."
        )

    # 1. IQR normalise — operates on (n_regions × n_samples) df
    df_norm = iqr_normalize_per_sample(df)

    # 2. Transpose to (n_samples × n_regions) for sklearn
    X = df_norm.T.values  # (n_samples, n_regions)

    # 3. Replace NaN with 0 before PCA (sklearn does not handle NaN)
    X = np.nan_to_num(X, nan=0.0)

    # 4. PCA transform — all components are used
    X_pca = pca_model.transform(X)  # (n_samples, n_pcs)

    n_pcs = X_pca.shape[1]
    pc_labels = [f"PC{i + 1}" for i in range(n_pcs)]

    # 5. Return as (n_pcs × n_samples) to keep regions-on-rows convention
    return pd.DataFrame(
        X_pca.T,
        index=pc_labels,
        columns=df.columns,
    )


def _preprocess_chromhmm_mean(
    df: pd.DataFrame,
    *,
    region_order: List[str] | None,
) -> pd.DataFrame:
    """Z-score normalise per sample and reorder features.

    Parameters
    ----------
    df:
        ``(n_regions × n_samples)``.
    region_order:
        Ordered list of region IDs that the model expects.
        If ``None``, no reordering is applied.
    """
    # 1. Optionally filter / reorder regions
    if region_order is not None:
        missing = set(region_order) - set(df.index)
        if missing:
            raise ValueError(
                f"{len(missing)} regions required by the ChromHMM model are "
                "absent from the feature matrix.  First missing: "
                f"{next(iter(missing))!r}"
            )
        df = df.loc[region_order]

    # 2. Z-score per sample
    return zscore_normalize_per_sample(df)
