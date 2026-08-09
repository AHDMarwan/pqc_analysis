import numpy as np

from qaccess.controls import anisotropic_relative_orientation_control, isotropic_controls
from qaccess.geometry import cross_validated_kyfan_recovery


def test_isotropic_moments_are_close_to_exact_values():
    df = isotropic_controls()
    assert np.max(np.abs(df.sample_mean - df.expected_mean)) < 0.02
    assert np.max(np.abs(df.purity - df.expected_purity)) < 0.004


def test_random_readout_mean_obeys_rank_law_for_anisotropic_ensemble():
    df = anisotropic_relative_orientation_control()
    assert np.max(np.abs(df["mean"] - df.expected)) < 0.015


def test_cross_validated_kyfan_is_finite_and_bounded():
    rng = np.random.default_rng(1234)
    u = rng.normal(size=(300, 31))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    out = cross_validated_kyfan_recovery(u, rank=5, seed=7)
    assert 0.0 <= out["train"] <= 1.0 + 1e-12
    assert 0.0 <= out["test"] <= 1.0 + 1e-12
    assert out["train_size"] + out["test_size"] == len(u)
