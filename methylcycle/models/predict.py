"""
methylcycle.models.predict
==========================
High-level inference function that runs the complete pipeline:
 
    load → QC → feature extraction → preprocessing → model prediction
 
Usage
-----
.. code-block:: python
 
    from methylcycle import predict
    from methylcycle.models.maps import MapType
    from methylcycle.models.feature_extraction import Metric
 
    results = predict(
        files=["sample1.parquet", "sample2.parquet"],
        map_type=MapType.RT,
        metric=Metric.MEAN,
    )
    print(results)
"""
 
from __future__ import annotations
 
import pathlib
from typing import List, Sequence, Tuple, Union
 
import joblib
import numpy as np
import pandas as pd
 
from methylcycle.parsers.loader import load_methylation_file
from methylcycle.utils.qc import passes_qc, QCReport
from methylcycle.models.maps import MapType, get_region_list
from methylcycle.models.feature_extraction import Metric, extract_features
from methylcycle.models.preprocessing import preprocess
import methylcycle.models.paths as _paths
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def predict(
    files: Sequence[Union[str, pathlib.Path]],
    map_type: MapType,
    metric: Metric = Metric.MEAN,
    *,
    sample_names: List[str] | None = None,
    custom_path: str | pathlib.Path | None = None,
    custom_list: List[str] | None = None,
    min_cpgs: int = 100_000,
    # Model artefact overrides (use global paths if not set)
    pca_path: str | pathlib.Path | None = None,
    model_path: str | pathlib.Path | None = None,
    chromhmm_order_path: str | pathlib.Path | None = None,
    show_progress: bool = True,
    sep: str = "\t",
) -> pd.DataFrame:
    """Run the full MethylCycle pipeline on a list of sample files.
 
    Parameters
    ----------
    files:
        Sample methylation files (BED / CSV / Parquet) in canonical format.
        Files that fail QC are excluded with a warning.
    map_type:
        Genomic region map to use for feature extraction.
    metric:
        Feature metric: ``Metric.MEAN`` or ``Metric.ENTROPY``.
    sample_names:
        Labels for each sample.  Defaults to file stems.
    custom_path / custom_list:
        Region source when ``map_type == MapType.CUSTOM``.
    min_cpgs:
        QC threshold — minimum covered CpG positions.
    pca_path:
        Override the global PCA artefact path.
    model_path:
        Override the global model artefact path.
    chromhmm_order_path:
        Override the global ChromHMM region order file path.
    show_progress:
        Show ``tqdm`` progress bars.
    sep:
        Column separator for text files. Use ``','`` for CSV. Ignored for Parquet.
 
    Returns
    -------
    pd.DataFrame
        Index = sample names (QC-passed only).
        Columns = predicted class probabilities or regression values,
        depending on the loaded model.  Raw QC reports are attached as
        the ``.attrs["qc"]`` dictionary.
    """
    files = [pathlib.Path(f) for f in files]
    if sample_names is None:
        sample_names = [f.stem for f in files]
 
    # ── 1. QC filter ─────────────────────────────────────────────────────
    passed_files, passed_names, qc_reports = _run_qc(
        files, sample_names, min_cpgs=min_cpgs, sep=sep
    )
    if not passed_files:
        raise RuntimeError("All samples failed QC. No samples to process.")
 
    # ── 2. Feature extraction ─────────────────────────────────────────────
    _, _, df_metric = extract_features(
        passed_files,
        map_type,
        metric,
        sample_names=passed_names,
        custom_path=custom_path,
        custom_list=custom_list,
        show_progress=show_progress,
        sep=sep,
    )
 
    # ── 3. Load artefacts ─────────────────────────────────────────────────
    pca_model = None
    chromhmm_order = None
 
    if map_type == MapType.RT and metric == Metric.MEAN:
        _pca_path = pathlib.Path(pca_path or _paths.RT_MEAN_PCA)
        pca_model = _load_artefact(_pca_path, "PCA model")
        _model_path = pathlib.Path(model_path or _paths.RT_MEAN_MODEL)
 
    elif map_type == MapType.RT and metric == Metric.ENTROPY:
        _model_path = pathlib.Path(model_path or _paths.RT_ENTROPY_MODEL)
 
    elif map_type == MapType.CHROMHMM and metric == Metric.MEAN:
        _order_path = pathlib.Path(chromhmm_order_path or _paths.CHROMHMM_REGIONS_ORDER)
        chromhmm_order = _load_region_order(_order_path)
        _model_path = pathlib.Path(model_path or _paths.CHROMHMM_MEAN_MODEL)
 
    elif map_type == MapType.CHROMHMM and metric == Metric.ENTROPY:
        _model_path = pathlib.Path(model_path or _paths.CHROMHMM_ENTROPY_MODEL)
 
    else:
        # Custom map — caller must supply model_path
        if model_path is None:
            raise ValueError(
                "A 'model_path' must be supplied when using MapType.CUSTOM."
            )
        _model_path = pathlib.Path(model_path)
 
    clf = _load_artefact(_model_path, "model")
 
    # ── 4. Preprocessing ──────────────────────────────────────────────────
    X = preprocess(
        df_metric,
        map_type,
        metric,
        pca_model=pca_model,
        chromhmm_region_order=chromhmm_order,
    )  # (n_features × n_samples)
 
    # ── 5. Inference ──────────────────────────────────────────────────────
    X_T = X.T.values  # (n_samples, n_features) — sklearn convention
    X_T = np.nan_to_num(X_T, nan=0.0)
 
    if hasattr(clf, "predict_proba"):
        preds = clf.predict_proba(X_T)
        out_cols = [f"prob_class_{c}" for c in clf.classes_]
    else:
        preds = clf.predict(X_T).reshape(-1, 1)
        out_cols = ["prediction"]
 
    result = pd.DataFrame(preds, index=passed_names, columns=out_cols)
    result.attrs["qc"] = qc_reports
    return result
 
 
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _run_qc(
    files: List[pathlib.Path],
    names: List[str],
    *,
    min_cpgs: int,
    sep: str = "\t",
) -> Tuple[List[pathlib.Path], List[str], dict]:
    """Load each file, run QC, and return the subset that passes."""
    import warnings
 
    passed_files: List[pathlib.Path] = []
    passed_names: List[str] = []
    reports: dict = {}
 
    for path, name in zip(files, names):
        df = load_methylation_file(path, sep=sep)
        report = passes_qc(df, min_cpgs=min_cpgs)
        reports[name] = report
        if report.passed:
            passed_files.append(path)
            passed_names.append(name)
        else:
            warnings.warn(
                f"Sample '{name}' excluded — {report.message}",
                UserWarning,
                stacklevel=3,
            )
 
    return passed_files, passed_names, reports
 
 
def _load_artefact(path: pathlib.Path, label: str):
    """Load a joblib artefact from *path* with a helpful error message."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} artefact not found: {path}. "
            "Set the correct path via methylcycle.models.paths or "
            "pass it explicitly to predict()."
        )
    return joblib.load(path)
 
 
def _load_region_order(path: pathlib.Path) -> List[str]:
    """Read ChromHMM region order file (one region per line)."""
    from methylcycle.parsers.regions import load_region_file
    return load_region_file(path)