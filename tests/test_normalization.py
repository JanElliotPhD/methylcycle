"""Tests for methylcycle.utils.normalization"""
import numpy as np
import pandas as pd
import pytest
from methylcycle.utils.normalization import iqr_normalize_per_sample, zscore_normalize_per_sample


def _df(data):
    return pd.DataFrame(data)


def test_iqr_shape_preserved():
    df = _df({"s1": [1, 2, 3, 4, 5], "s2": [10, 20, 30, 40, 50]})
    out = iqr_normalize_per_sample(df)
    assert out.shape == df.shape


def test_iqr_median_is_zero():
    df = _df({"s1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = iqr_normalize_per_sample(df)
    assert abs(out["s1"].median()) < 1e-10


def test_zscore_mean_is_zero():
    df = _df({"s1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = zscore_normalize_per_sample(df)
    assert abs(out["s1"].mean()) < 1e-10


def test_zscore_std_is_one():
    df = _df({"s1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = zscore_normalize_per_sample(df)
    assert abs(out["s1"].std(ddof=0) - 1.0) < 1e-10


def test_zero_iqr_returns_nan():
    df = _df({"s1": [5.0, 5.0, 5.0]})
    out = iqr_normalize_per_sample(df)
    assert out["s1"].isna().all()


def test_zero_std_returns_nan():
    df = _df({"s1": [3.0, 3.0, 3.0]})
    out = zscore_normalize_per_sample(df)
    assert out["s1"].isna().all()
