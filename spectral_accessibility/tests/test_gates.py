import numpy as np
from scipy.linalg import expm

from qaccess.gates import H_XY, haar_u2, haar_u4, xy_gate


def test_xy_gate_matches_matrix_exponential():
    for theta in (-1.3, -0.2, 0.0, 0.7, 2.1):
        ref = expm(-1j * theta * H_XY)
        assert np.allclose(xy_gate(theta), ref, atol=2e-14, rtol=2e-14)


def test_haar_generators_are_unitary():
    rng = np.random.default_rng(123)
    for u in (haar_u2(rng), haar_u4(rng)):
        assert np.allclose(u.conj().T @ u, np.eye(len(u)), atol=1e-12)
