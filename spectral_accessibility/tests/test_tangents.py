import numpy as np
import pytest

from qaccess.circuits import CORE_FAMILIES, parameter_count, sample_parameter_directions, simulate_vqc_tangent_batch
from qaccess.measurement import probability_tangent_batch
from qaccess.seeding import stable_seed


@pytest.mark.parametrize("family", CORE_FAMILIES)
def test_probability_tangent_matches_centered_finite_difference(family):
    n, depth = 4, 2
    rng = np.random.default_rng(stable_seed("test-fd|" + family))
    pcount = parameter_count(n, depth, family)
    theta = rng.uniform(-np.pi, np.pi, size=pcount)
    direction = sample_parameter_directions(rng, 1, pcount)
    arch = stable_seed("test-fd|" + family + "|arch")
    psi, phis, _ = simulate_vqc_tangent_batch(n, depth, family, theta, direction, arch)
    _, dp = probability_tangent_batch(psi, phis)
    eps = 1e-6
    zero = np.zeros_like(direction)
    psi_p, _, _ = simulate_vqc_tangent_batch(n, depth, family, theta + eps * direction[0], zero, arch)
    psi_m, _, _ = simulate_vqc_tangent_batch(n, depth, family, theta - eps * direction[0], zero, arch)
    dp_fd = (np.abs(psi_p) ** 2 - np.abs(psi_m) ** 2) / (2 * eps)
    assert np.max(np.abs(dp[0] - dp_fd)) < 2e-6
    assert abs(np.vdot(psi, psi).real - 1) < 1e-12
    assert abs(np.vdot(psi, phis[0])) < 1e-10
