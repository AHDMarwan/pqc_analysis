import numpy as np
import pytest

from qaccess.circuits import CORE_FAMILIES, parameter_count, sample_parameter_directions, simulate_vqc_tangent_batch
from qaccess.geometry import normalized_visible_scores_from_probabilities
from qaccess.measurement import apply_independent_bitflip_noise, probability_tangent_batch
from qaccess.seeding import stable_seed


@pytest.mark.parametrize("family", CORE_FAMILIES)
def test_measurement_cfi_does_not_exceed_pure_state_qfi(family):
    n, depth, m = 4, 3, 16
    rng = np.random.default_rng(stable_seed("bc|"+family))
    pc = parameter_count(n, depth, family)
    theta = rng.uniform(-np.pi, np.pi, pc)
    dirs = sample_parameter_directions(rng, m, pc)
    psi, phis, _ = simulate_vqc_tangent_batch(n, depth, family, theta, dirs, stable_seed("bc|"+family+"|arch"))
    p, dp = probability_tangent_batch(psi, phis)
    fq = 4*np.sum(np.abs(phis)**2, axis=1).real
    for noise in (0.0, 0.02, 0.1):
        pn, dpn = apply_independent_bitflip_noise(p, dp, n, noise)
        info = normalized_visible_scores_from_probabilities(pn, dpn, fq)
        assert np.all(info["Ffull"] <= fq*(1+2e-7))
