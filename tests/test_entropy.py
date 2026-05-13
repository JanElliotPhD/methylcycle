"""Tests for methylcycle.utils.entropy"""
import numpy as np
import pandas as pd
import pytest
from methylcycle.utils.entropy import compute_entropy_matrix


def _df(data):
    return pd.DataFrame(data, dtype=float)


def test_max_entropy_at_50pct():
    """50 % methylation should give entropy close to 1.0."""
    meth = _df({"s1": [50]})
    unmeth = _df({"s1": [50]})
    H = compute_entropy_matrix(meth, unmeth, min_abs=1)
    assert abs(H.iloc[0, 0] - 1.0) < 0.01


def test_zero_entropy_fully_methylated():
    """100 % methylation → low entropy (Laplace prior smooths away from 0)."""
    meth = _df({"s1": [100]})
    unmeth = _df({"s1": [0]})
    # With default prior=0.5, p ≈ 100.5/101 ≈ 0.995 → H < 0.07
    H = compute_entropy_matrix(meth, unmeth, min_abs=1, prior=0.5)
    assert H.iloc[0, 0] < 0.07


def test_nan_below_min_abs():
    meth = _df({"s1": [2]})
    unmeth = _df({"s1": [1]})
    H = compute_entropy_matrix(meth, unmeth, min_abs=5)
    assert np.isnan(H.iloc[0, 0])


def test_shape_preserved():
    meth = _df({"s1": [10, 20], "s2": [5, 15]})
    unmeth = _df({"s1": [10, 10], "s2": [5, 5]})
    H = compute_entropy_matrix(meth, unmeth, min_abs=1)
    assert H.shape == meth.shape
