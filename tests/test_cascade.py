"""Tests for the RT + IZ cascade (methylcycle.models.cascade, methylcycle.models.iz_features)."""
import numpy as np
import pandas as pd

from methylcycle.models.cascade import build_cascade, predict_from_matrices
from methylcycle.models.iz_features import CIEGO9, iz_contrast_features


def _toy(n=90, n_dom=40, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(["G1"] * (n // 2) + ["S"] * (n // 6) + ["G2M"] * (n - n // 2 - n // 6))
    X1 = rng.random((n, n_dom)) * 0.2 + 0.5
    X1[y == "G2M", : n_dom // 2] -= 0.15                # a G2M signal in half of the domains
    X2 = rng.normal(size=(n, len(CIEGO9)))
    X2[y == "S", 0] += 2.0
    return X1, X2, y


def test_cascade_predicts_and_corrects():
    X1, X2, y = _toy()
    art = build_cascade(X1, X2, y, rt_regions=[f"chr1-{i}-{i + 1}" for i in range(X1.shape[1])])
    for correct in (False, True):
        res = predict_from_matrices(art, X1, X2, correct=correct)
        assert list(res.columns[:3]) == ["prob_G1", "prob_S", "prob_G2M"]
        assert np.allclose(res[["prob_G1", "prob_S", "prob_G2M"]].sum(axis=1), 1.0)
        assert set(res["prediction"]) <= {"G1", "S", "G2M"}
    # a constant shift of a whole dataset is removed by the correction
    shifted = predict_from_matrices(art, X1 + 0.1, X2, correct=True)
    base = predict_from_matrices(art, X1, X2, correct=True)
    assert np.allclose(shifted["stage1_prob_G2M"], base["stage1_prob_G2M"])


def test_iz_features_shape_and_contrast():
    rng = np.random.default_rng(1)
    ids = [f"chr1-{i * 100}-{i * 100 + 50}" for i in range(300)]
    m = pd.DataFrame(rng.integers(0, 6, (300, 8)).astype(float), index=ids)
    u = pd.DataFrame(rng.integers(0, 6, (300, 8)).astype(float), index=ids)
    cls = pd.Series(["early"] * 200 + ["mid"] * 70 + ["late"] * 30, index=ids)
    F = iz_contrast_features(m, u, cls)
    assert F.shape == (8, 24)
    # all nine stage-2 features are defined for this input
    assert np.isfinite(F[CIEGO9].to_numpy()).all()
