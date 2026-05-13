"""Tests for methylcycle.utils.qc"""
import numpy as np
import pandas as pd
import pytest
from methylcycle.utils.qc import passes_qc, MIN_COVERED_CPGS


def _make_df(n_covered: int, n_zero: int = 0) -> pd.DataFrame:
    rows = (
        [{"Chromosome": "chr1", "Position": i, "Count Methylated": 3, "Count Unmethylated": 2}
         for i in range(n_covered)]
        + [{"Chromosome": "chr1", "Position": n_covered + i, "Count Methylated": 0, "Count Unmethylated": 0}
           for i in range(n_zero)]
    )
    return pd.DataFrame(rows)


def test_passes_qc_above_threshold():
    df = _make_df(MIN_COVERED_CPGS + 1)
    report = passes_qc(df)
    assert report.passed is True
    assert bool(report) is True


def test_fails_qc_below_threshold():
    df = _make_df(MIN_COVERED_CPGS - 1)
    report = passes_qc(df)
    assert report.passed is False


def test_custom_threshold():
    df = _make_df(50)
    assert passes_qc(df, min_cpgs=50).passed is True
    assert passes_qc(df, min_cpgs=51).passed is False


def test_zero_coverage_rows_excluded():
    df = _make_df(n_covered=200_000, n_zero=50_000)
    report = passes_qc(df)
    assert report.covered_cpgs == 200_000


def test_missing_column_raises():
    df = pd.DataFrame({"Chromosome": ["chr1"], "Position": [1]})
    with pytest.raises(KeyError):
        passes_qc(df)
