"""
methylcycle.models.cascade
==========================
Two-stage RT + IZ cascade (SVM/SVM by default) with an optional domain-shift
correction.

    stage 1  G2M vs {G1, S}   on per-domain RT methylation (1,435 RT domains),
                               NaN -> training median, per-cell normalisation
    stage 2  G1 / S / G2M     on the 9 IZ contrast features (CIEGO9)

Decision: G2M if P1(G2M) >= t1; otherwise stage 2 (G2M if it is its argmax,
else S if wS * P2(S) > P2(G1), else G1). Combined probabilities:
P(G1) = (1 - p1) Q(G1), P(S) = (1 - p1) Q(S), P(G2M) = p1 + (1 - p1) Q(G2M).

Domain-shift correction (``correct=True``) — dataset-level location
alignment of stage 1, no labels used:

    x'_cd = x_cd - mean_{c in dataset}(x_cd) + mean_{training}(x_cd)

It replaces the dataset's own baseline profile (cell type, protocol) by the
training one and keeps every difference between cells of the dataset. It assumes
the phase mix of the dataset is not very different from the training one
(~50 % G1, 13 % S, 36 % G2M) and needs a reasonable number of cells (tens);
the prediction of a cell depends on the other cells of the batch.

Usage
-----
Build the artefact once (in the analysis notebook, with the training matrices)::

    from methylcycle.models.cascade import build_cascade, save_cascade
    art = build_cascade(X1, X2, yv, rt_regions=train_keys)          # SVM/SVM
    save_cascade(art)                                               # -> paths.CASCADE_SVM_MODEL

Predict new samples::

    from methylcycle.models.cascade import predict_cascade_files
    res = predict_cascade_files(files, correct=True, run_qc=False)
"""

from __future__ import annotations

import pathlib
from typing import List, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import methylcycle.models.paths as _paths
from methylcycle.models.iz_features import CIEGO9, iz_contrast_features

CLS: List[str] = ["G1", "S", "G2M"]


# ---------------------------------------------------------------------------
# Stage-1 preprocessing
# ---------------------------------------------------------------------------

class CellCentre(BaseEstimator, TransformerMixin):
    """NaN -> per-domain median of the training cells, then a per-cell normalisation.

    how: "median" | "mean" | "zscore" | "robust" | "rank" | None
    """
    HOWS = ("median", "mean", "zscore", "robust", "rank", None)

    def __init__(self, how="mean"):
        self.how = how

    def fit(self, X, y=None):
        if self.how not in self.HOWS:
            raise ValueError(f"how must be one of {self.HOWS}, not {self.how!r}")
        fill = np.nanmedian(np.asarray(X, float), axis=0)
        self.fill_ = np.where(np.isfinite(fill), fill, 0.0)
        return self

    def transform(self, X):
        X = np.asarray(X, float).copy()
        bad = ~np.isfinite(X)
        X[bad] = np.take(self.fill_, np.nonzero(bad)[1])
        if self.how is None:
            return X
        if self.how == "median":
            return X - np.median(X, axis=1, keepdims=True)
        if self.how == "mean":
            return X - X.mean(axis=1, keepdims=True)
        if self.how == "zscore":
            sd = X.std(axis=1, keepdims=True)
            return (X - X.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
        if self.how == "robust":
            q25, q50, q75 = np.percentile(X, [25, 50, 75], axis=1, keepdims=True)
            iqr = q75 - q25
            return (X - q50) / np.where(iqr > 0, iqr, 1.0)
        return (rankdata(X, axis=1) - 0.5) / X.shape[1]


def _base_model(name: str, seed: int):
    if name == "SVM":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, class_weight="balanced",
                                                   probability=True, random_state=seed))
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, max_features="sqrt",
                                      class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    raise ValueError(f"model must be 'SVM', 'LogReg' or 'RF', not {name!r}")


# ---------------------------------------------------------------------------
# Building / saving / loading the artefact
# ---------------------------------------------------------------------------

def build_cascade(
    X1: np.ndarray,
    X2: np.ndarray,
    y: Sequence[str],
    *,
    rt_regions: Sequence[str],
    model: str = "SVM",
    center: str | None = "mean",
    stage2_mode: str = "3class",
    t1: float = 0.5,
    wS: float = 1.0,
    seed: int = 0,
    iz_features: Sequence[str] = tuple(CIEGO9),
    iz_min_cov: int = 3,
    min_calls: int = 1,
) -> dict:
    """Train the cascade on all training cells and return the artefact dictionary.

    Parameters
    ----------
    X1:
        ``(n_cells × n_domains)`` per-domain methylation rate, columns in ``rt_regions`` order.
    X2:
        ``(n_cells × n_iz_features)`` IZ features, columns in ``iz_features`` order.
    y:
        Phase labels (``G1`` / ``S`` / ``G2M``).
    rt_regions:
        RT domain ids (``chr<N>-<start>-<end>``) in the column order of ``X1``.
    """
    y = np.asarray(y).astype(str)
    if not set(y) <= set(CLS):
        raise ValueError(f"unexpected labels: {set(y) - set(CLS)}")
    X1, X2 = np.asarray(X1, float), np.asarray(X2, float)
    if X1.shape[1] != len(rt_regions):
        raise ValueError(f"X1 has {X1.shape[1]} columns but {len(rt_regions)} rt_regions were given")

    s1 = Pipeline([("centre", CellCentre(center)), ("clf", _base_model(model, seed))])
    s1.fit(X1, (y == "G2M").astype(int))
    s2 = Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", _base_model(model, seed))])
    if stage2_mode == "binary":
        keep = y != "G2M"
        s2.fit(X2[keep], (y[keep] == "S").astype(int))
    elif stage2_mode == "3class":
        s2.fit(X2, y)
    else:
        raise ValueError("stage2_mode must be 'binary' or '3class'")

    return {
        "s1": s1, "s2": s2, "t1": float(t1), "wS": float(wS),
        "model": model, "center": center, "stage2_mode": stage2_mode,
        "rt_regions": [str(r) for r in rt_regions],
        "iz_features": list(iz_features), "iz_min_cov": int(iz_min_cov), "min_calls": int(min_calls),
        "train_mean_rt": np.nanmean(X1, axis=0),           # reference profile of the domain-shift correction
        "n_train": int(len(y)), "train_phase_mix": pd.Series(y).value_counts(normalize=True).to_dict(),
    }


def save_cascade(artefact: dict, path: Union[str, pathlib.Path, None] = None) -> pathlib.Path:
    path = pathlib.Path(path or _paths.CASCADE_SVM_MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, path)
    return path


def load_cascade(path: Union[str, pathlib.Path, None] = None) -> dict:
    path = pathlib.Path(path or _paths.CASCADE_SVM_MODEL)
    if not path.exists():
        raise FileNotFoundError(
            f"Cascade artefact not found: {path}. Build it with build_cascade() + save_cascade(), "
            "or pass the path explicitly."
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _stage_outputs(art: dict, Xa: np.ndarray, Xb: np.ndarray):
    p1 = art["s1"].predict_proba(Xa)[:, 1]
    if art["stage2_mode"] == "binary":
        ps = art["s2"].predict_proba(Xb)[:, 1]
        Q = np.column_stack([1 - ps, ps, np.zeros_like(ps)])
    else:
        Q = pd.DataFrame(art["s2"].predict_proba(Xb), columns=art["s2"].classes_)[CLS].to_numpy()
    return p1, Q


def predict_from_matrices(
    art: dict,
    Xa: np.ndarray,
    Xb: np.ndarray,
    *,
    correct: bool = True,
    index: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Cascade prediction from stage-1 rates (cells × RT domains) and IZ features (cells × features).

    ``correct=True`` applies the domain-shift correction to stage 1 using all rows of ``Xa``
    as the dataset.
    """
    Xa = np.asarray(Xa, float)
    if correct:
        with np.errstate(all="ignore"):
            Xa = Xa - np.nanmean(Xa, axis=0)[None, :] + art["train_mean_rt"][None, :]
    p1, Q = _stage_outputs(art, Xa, np.asarray(Xb, float))
    g2m2 = Q.argmax(1) == 2
    s_or_g1 = np.where(art["wS"] * Q[:, 1] > Q[:, 0], "S", "G1")
    pred = np.where(p1 >= art["t1"], "G2M", np.where(g2m2, "G2M", s_or_g1))
    P = np.column_stack([(1 - p1) * Q[:, 0], (1 - p1) * Q[:, 1], p1 + (1 - p1) * Q[:, 2]])
    out = pd.DataFrame(P, index=index, columns=[f"prob_{c}" for c in CLS])
    out["stage1_prob_G2M"] = p1
    out["prediction"] = pred
    out.attrs["corrected"] = bool(correct)
    return out


def _counts_from_extraction(df_a: pd.DataFrame, df_b: pd.DataFrame, has_counts: bool):
    """(methylated, unmethylated) per region from extract_features(..., Metric.MEAN) outputs.

    Count files: the two matrices are already methylated / unmethylated reads.
    Percentage files: df_a = sum of per-CpG fractions, df_b = number of CpGs, so
    methylated = df_a and unmethylated = df_b - df_a (one fractional call per CpG).
    """
    if has_counts:
        return df_a, df_b
    return df_a, (df_b - df_a).clip(lower=0)


def predict_cascade_files(
    files: Sequence[Union[str, pathlib.Path]],
    *,
    correct: bool = True,
    model_path: Union[str, pathlib.Path, None] = None,
    sample_names: List[str] | None = None,
    run_qc: bool = True,
    min_cpgs: int = 100_000,
    iz_regions_path: Union[str, pathlib.Path, None] = None,
    iz_classes_path: Union[str, pathlib.Path, None] = None,
    show_progress: bool = True,
    sep: str = "\t",
) -> pd.DataFrame:
    """Full RT + IZ cascade on methylation files (count or percentage files).

    Parameters
    ----------
    files:
        Sample files (BED / CSV / Parquet).
    correct:
        Apply the domain-shift correction (dataset = all files given). Use it for datasets
        from a different cell type or protocol than the training data, with a phase mix
        that is not extreme (not all G1, not all G2M).
    model_path:
        Cascade artefact (default: ``paths.CASCADE_SVM_MODEL``).
    run_qc, min_cpgs:
        Optional coverage QC (skip it with ``run_qc=False`` to save time).
    iz_regions_path, iz_classes_path:
        Override the built-in IZ map (``data/IZ_Regions.txt``, ``data/IZ_Classes.tsv``).

    Returns
    -------
    pd.DataFrame
        One row per sample: ``prob_G1``, ``prob_S``, ``prob_G2M``, ``stage1_prob_G2M``,
        ``prediction``.
    """
    from methylcycle.models.feature_extraction import Metric, extract_features
    from methylcycle.models.maps import MapType, get_iz_classes, get_region_list
    from methylcycle.models.predict import _run_qc
    from methylcycle.parsers.loader import load_methylation_file

    art = load_cascade(model_path)
    files = [pathlib.Path(f) for f in files]
    names = sample_names or [f.stem for f in files]
    if run_qc:
        files, names, _ = _run_qc(files, names, min_cpgs=min_cpgs, sep=sep)
        if not files:
            raise RuntimeError("All samples failed QC. No samples to process.")
    has_counts = load_methylation_file(files[0], sep=sep).attrs.get("has_counts", True)

    # stage 1: per-domain rate in the training domain order
    rt_m, rt_u, _ = extract_features(files, MapType.CUSTOM, Metric.MEAN, sample_names=names,
                                     custom_list=art["rt_regions"], show_progress=show_progress, sep=sep)
    rt_m, rt_u = _counts_from_extraction(rt_m, rt_u, has_counts)
    M, N = rt_m.to_numpy(float), (rt_m + rt_u).to_numpy(float)
    Xa = np.where(N >= art["min_calls"], M / np.maximum(N, 1e-12), np.nan).T

    # stage 2: IZ counts -> contrast features
    if iz_regions_path is not None:
        iz_list = get_region_list(MapType.CUSTOM, custom_path=iz_regions_path)
    else:
        iz_list = get_region_list(MapType.IZ)
    iz_m, iz_u, _ = extract_features(files, MapType.CUSTOM, Metric.MEAN, sample_names=names,
                                     custom_list=iz_list, show_progress=show_progress, sep=sep)
    iz_m, iz_u = _counts_from_extraction(iz_m, iz_u, has_counts)
    F = iz_contrast_features(iz_m, iz_u, get_iz_classes(iz_classes_path),
                             min_cov=art["iz_min_cov"], features=art["iz_features"])
    Xb = F.loc[names].to_numpy(float)

    return predict_from_matrices(art, Xa, Xb, correct=correct, index=names)
