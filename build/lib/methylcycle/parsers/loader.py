"""
methylcycle.parsers.loader
==========================
Load methylation files in BED, CSV or Parquet format and return a
normalised DataFrame.

Two possible output schemas depending on source file content:

**Count-based** (methylated + unmethylated read columns present):

    Chromosome | Position | Count Methylated | Count Unmethylated

    ``df.attrs["has_counts"] = True``

**Percentage-only** (only a methylation percentage / beta value):

    Chromosome | Position | Methylation Percentage

    ``df.attrs["has_counts"] = False``

The ``has_counts`` flag drives downstream feature-extraction logic:

+------------+---------+----------------------------------------------+
| has_counts | Metric  | Computation                                  |
+============+=========+==============================================+
| True       | MEAN    | sum(meth) / sum(total) per region            |
| True       | ENTROPY | unique-CpG binary entropy (original method)  |
| False      | MEAN    | mean(Methylation Percentage) per region      |
| False      | ENTROPY | mean of per-CpG H(p, 1-p) across region     |
+------------+---------+----------------------------------------------+

Rules
-----
- Chromosome: always prefixed with 'chr' (added if missing).
- Methylation Percentage normalised to [0, 1].
- Extra columns silently dropped.
"""

from __future__ import annotations

import pathlib
from typing import Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column alias maps  (lower-cased keys → canonical name)
# ---------------------------------------------------------------------------
_CHROM_ALIASES   = {"chromosome", "chrom", "chr", "#chrom", "Chromosome"}
_POS_ALIASES     = {"position", "pos", "start", "chromstart", "Start"}
_METH_PCT_ALIASES = {
    "methylation_percentage", "methylation percentage",
    "meth_pct", "methpct", "pct_meth",
    "beta_value", "beta value", "methylation_rate",
}
_METH_CNT_ALIASES = {
    "count_methylated", "count methylated",
    "methylated_counts", "methylated counts",
    "meth_count", "c", "m_count", "Count Methylated", "Methylated Counts"
}
_UNMETH_CNT_ALIASES = {
    "count_unmethylated", "count unmethylated",
    "unmethylated_counts", "unmethylated counts",
    "unmeth_count", "u_count", "Count Unmethylated", "Unmethylated Counts"
}

# Canonical output column names
COL_CHROM = "Chromosome"
COL_POS   = "Position"
COL_METH  = "Count Methylated"
COL_UNMETH = "Count Unmethylated"
COL_PCT   = "Methylation Percentage"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_methylation_file(
    path: Union[str, pathlib.Path],
    *,
    sep: str = "\t",
    chrom_col: str | None = None,
    pos_col: str | None = None,
    meth_col: str | None = None,
    unmeth_col: str | None = None,
    pct_col: str | None = None,
) -> pd.DataFrame:
    """Load a methylation file and return a normalised DataFrame.

    Parameters
    ----------
    path:
        Path to a ``.bed``, ``.csv``, ``.tsv`` or ``.parquet`` file.
    sep:
        Column separator for text files (default ``\\t``). Ignored for Parquet.
    chrom_col, pos_col, meth_col, unmeth_col, pct_col:
        Override automatic column detection with explicit column names.

    Returns
    -------
    pd.DataFrame
        Schema depends on source file (see module docstring).
        ``df.attrs["has_counts"]`` (bool) indicates which schema is present.

    Raises
    ------
    ValueError
        If required columns cannot be found or inferred.
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Methylation file not found: {path}")

    raw = _read_raw(path, sep=sep)
    df, has_counts = _normalise_columns(
        raw,
        chrom_col=chrom_col,
        pos_col=pos_col,
        meth_col=meth_col,
        unmeth_col=unmeth_col,
        pct_col=pct_col,
    )
    df = _normalise_chromosome(df)
    df[COL_POS] = df[COL_POS].astype(np.int64)

    if has_counts:
        df[COL_METH]  = df[COL_METH].astype(np.int64)
        df[COL_UNMETH] = df[COL_UNMETH].astype(np.int64)
    else:
        df[COL_PCT] = df[COL_PCT].astype(np.float64)

    df.attrs["has_counts"] = has_counts
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_raw(path: pathlib.Path, sep: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".bed", ".tsv"}:
        return pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    if suffix == ".csv":
        return pd.read_csv(path, sep=sep, comment="#", low_memory=False)
    try:
        return pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=",", comment="#", low_memory=False)


def _find_col(
    columns: list,
    aliases: set,
    *,
    override: str | None = None,
    required: bool = True,
    label: str = "column",
) -> str | None:
    if override is not None:
        if override not in columns:
            raise ValueError(
                f"Explicit column '{override}' not found in file. "
                f"Available: {columns}"
            )
        return override
    col_lower = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in col_lower:
            return col_lower[alias]
    if required:
        raise ValueError(
            f"Cannot detect {label} column. "
            f"Tried aliases: {aliases}. "
            f"Available columns: {columns}. "
            "Use the explicit override parameter to specify it."
        )
    return None


def _normalise_columns(
    df: pd.DataFrame,
    *,
    chrom_col: str | None,
    pos_col: str | None,
    meth_col: str | None,
    unmeth_col: str | None,
    pct_col: str | None,
) -> tuple:
    """Return (normalised_df, has_counts)."""
    cols = df.columns.tolist()

    chrom  = _find_col(cols, _CHROM_ALIASES,    override=chrom_col,  label="chromosome")
    pos    = _find_col(cols, _POS_ALIASES,       override=pos_col,    label="position")
    meth   = _find_col(cols, _METH_CNT_ALIASES,  override=meth_col,   required=False, label="methylated count")
    unmeth = _find_col(cols, _UNMETH_CNT_ALIASES, override=unmeth_col, required=False, label="unmethylated count")
    pct    = _find_col(cols, _METH_PCT_ALIASES,  override=pct_col,   required=False, label="methylation percentage")

    out = pd.DataFrame()
    out[COL_CHROM] = df[chrom]
    out[COL_POS]   = df[pos]

    if meth is not None and unmeth is not None:
        out[COL_METH]  = pd.to_numeric(df[meth],   errors="coerce").fillna(0)
        out[COL_UNMETH] = pd.to_numeric(df[unmeth], errors="coerce").fillna(0)
        return out, True

    if pct is not None:
        p = pd.to_numeric(df[pct], errors="coerce").fillna(0.0)
        if p.max() > 1.0:
            p = p / 100.0
        out[COL_PCT] = p.clip(0.0, 1.0)
        return out, False

    raise ValueError(
        "File must contain either (Count Methylated + Count Unmethylated) "
        "or a Methylation Percentage column. None detected."
    )


def _normalise_chromosome(df: pd.DataFrame) -> pd.DataFrame:
    chrom = df[COL_CHROM].astype(str).str.strip()
    mask = ~chrom.str.startswith("chr")
    chrom = chrom.copy()
    chrom.loc[mask] = "chr" + chrom.loc[mask]
    df = df.copy()
    df[COL_CHROM] = chrom
    return df
