"""
methylcycle.utils.qc
====================
Quality-control check for a single methylation sample.

A sample **passes** if it has at least ``min_cpgs`` (default 100 000)
covered CpG positions:

- count files: positions where ``Count Methylated + Count Unmethylated >= 1``;
- percentage-only files: positions with a (non-missing) ``Methylation Percentage``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MIN_COVERED_CPGS: int = 100_000


@dataclass
class QCReport:
    """Result of a QC check.

    Attributes
    ----------
    passed : bool
    covered_cpgs : int
        Number of positions with coverage ≥ 1.
    min_required : int
        Threshold used.
    message : str
        Human-readable summary.
    """
    passed: bool
    covered_cpgs: int
    min_required: int
    message: str

    def __bool__(self) -> bool:   # allows ``if passes_qc(df):``
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"QCReport({status} | "
            f"covered={self.covered_cpgs:,} / required={self.min_required:,})"
        )


def passes_qc(
    df: pd.DataFrame,
    *,
    min_cpgs: int = MIN_COVERED_CPGS,
    meth_col: str = "Count Methylated",
    unmeth_col: str = "Count Unmethylated",
    pct_col: str = "Methylation Percentage",
) -> QCReport:
    """Check whether a methylation DataFrame passes coverage QC.

    Parameters
    ----------
    df:
        Output of :func:`methylcycle.parsers.loader.load_methylation_file`.
    min_cpgs:
        Minimum number of CpG positions that must have at least one read.
        Default is 100 000.
    meth_col, unmeth_col:
        Column names for methylated / unmethylated counts.
    pct_col:
        Column name of the methylation percentage, used when the count
        columns are absent (percentage-only files).

    Returns
    -------
    QCReport
        Evaluates to ``True`` (pass) or ``False`` (fail) in boolean context.

    Raises
    ------
    KeyError
        If neither the count columns nor the percentage column are present.
    """
    if meth_col in df.columns and unmeth_col in df.columns:
        coverage = df[meth_col] + df[unmeth_col]
        covered = int((coverage >= 1).sum())
    elif pct_col in df.columns:
        covered = int(df[pct_col].notna().sum())
    else:
        raise KeyError(
            f"Neither count columns ('{meth_col}', '{unmeth_col}') nor a "
            f"'{pct_col}' column found. Available columns: {df.columns.tolist()}"
        )
    passed = covered >= min_cpgs

    if passed:
        msg = (
            f"QC passed: {covered:,} CpG positions covered "
            f"(≥ {min_cpgs:,} required)."
        )
    else:
        msg = (
            f"QC failed: only {covered:,} CpG positions covered "
            f"({min_cpgs:,} required). Sample will be excluded."
        )

    return QCReport(
        passed=passed,
        covered_cpgs=covered,
        min_required=min_cpgs,
        message=msg,
    )