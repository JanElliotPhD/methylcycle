"""
methylcycle.datasets
====================
Convenience accessors for the example datasets shipped with MethylCycle.

Available datasets
------------------
``single_cell_mESC``
    10 single-cell scNMT samples from mouse embryonic stem cells (mESC),
    sourced from Argelaguet et al.  Files are in Parquet format.

``bulk_hepatocytes``
    12 bulk WGBS samples from mouse hepatocytes.  Files are in Parquet format.

Usage
-----
.. code-block:: python

    from methylcycle.datasets import load_dataset, list_datasets

    # See available datasets
    print(list_datasets())

    # Get file paths for a dataset
    files, names = load_dataset("single_cell_mESC")
    print(files)   # list of pathlib.Path objects
    print(names)   # list of sample name strings

    # Use directly with extract_features or predict
    from methylcycle.models.feature_extraction import extract_features, Metric
    from methylcycle.models.maps import MapType

    df_meth, df_unmeth, df_metric = extract_features(
        files=files,
        map_type=MapType.RT,
        metric=Metric.MEAN,
    )
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_DATA_DIR = pathlib.Path(__file__).parent / "data" / "example_datasets"

_DATASETS: Dict[str, dict] = {
    "single_cell_mESC": {
        "path": _DATA_DIR / "single_cell_mESC",
        "description": (
            "10 single-cell scNMT samples from mouse embryonic stem cells (mESC). "
            "Source: Argelaguet et al."
        ),
        "format": "parquet",
        "n_samples": 10,
        "organism": "Mus musculus",
        "cell_type": "Embryonic stem cells",
        "data_type": "Single-cell NMT",
    },
    "bulk_hepatocytes": {
        "path": _DATA_DIR / "bulk_hepatocytes",
        "description": (
            "12 bulk WGBS samples from mouse hepatocytes."
        ),
        "format": "parquet",
        "n_samples": 12,
        "organism": "Mus musculus",
        "cell_type": "Hepatocytes",
        "data_type": "Bulk WGBS",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_datasets() -> List[str]:
    """Return the names of all available example datasets.

    Returns
    -------
    list[str]

    Example
    -------
    >>> from methylcycle.datasets import list_datasets
    >>> list_datasets()
    ['single_cell_mESC', 'bulk_hepatocytes']
    """
    return list(_DATASETS.keys())


def dataset_info(name: str) -> dict:
    """Return metadata for a dataset.

    Parameters
    ----------
    name:
        Dataset name. Use :func:`list_datasets` to see available options.

    Returns
    -------
    dict
        Keys: ``description``, ``format``, ``n_samples``, ``organism``,
        ``cell_type``, ``data_type``, ``path``.

    Raises
    ------
    ValueError
        If ``name`` is not a known dataset.
    """
    _check_name(name)
    info = _DATASETS[name].copy()
    info["path"] = str(info["path"])
    return info


def load_dataset(
    name: str,
    *,
    check_files: bool = True,
) -> Tuple[List[pathlib.Path], List[str]]:
    """Return file paths and sample names for an example dataset.

    Parameters
    ----------
    name:
        Dataset name. Use :func:`list_datasets` to see available options.
    check_files:
        If ``True`` (default), raises ``FileNotFoundError`` when no files
        are found in the dataset directory (e.g. files not yet added).

    Returns
    -------
    files : list[pathlib.Path]
        Sorted list of Parquet file paths for this dataset.
    sample_names : list[str]
        Sample names derived from file stems (filename without extension).

    Raises
    ------
    ValueError
        If ``name`` is not a known dataset.
    FileNotFoundError
        If ``check_files=True`` and no files are found in the dataset folder.

    Example
    -------
    >>> from methylcycle.datasets import load_dataset
    >>> files, names = load_dataset("single_cell_mESC")
    >>> print(names)
    ['sample_01', 'sample_02', ...]
    """
    _check_name(name)
    info = _DATASETS[name]
    folder: pathlib.Path = info["path"]
    fmt: str = info["format"]

    files = sorted(folder.glob(f"*.{fmt}"))

    if check_files and not files:
        raise FileNotFoundError(
            f"No .{fmt} files found in dataset '{name}' at: {folder}\n"
            f"Please add the sample files to that directory."
        )

    sample_names = [f.stem for f in files]
    return files, sample_names


def get_dataset_path(name: str) -> pathlib.Path:
    """Return the directory path for a dataset.

    Useful if you want to explore the folder manually.

    Parameters
    ----------
    name:
        Dataset name.

    Returns
    -------
    pathlib.Path
    """
    _check_name(name)
    return _DATASETS[name]["path"]


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _check_name(name: str) -> None:
    if name not in _DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {list(_DATASETS.keys())}"
        )
