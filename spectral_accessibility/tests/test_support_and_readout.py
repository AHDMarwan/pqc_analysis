import math
import numpy as np
import pytest

from qaccess.circuits import CORE_FAMILIES, parameter_count, sample_parameter_directions, simulate_vqc_tangent_batch
from qaccess.geometry import normalized_visible_scores_from_probabilities, walsh_readout_basis
from qaccess.measurement import probability_tangent_batch
from qaccess.seeding import stable_seed


def _sample(family, n=6, depth=3, m=12):
    rng = np.random.default_rng(stable_seed("support|" + family))
    pcount = parameter_count(n, depth, family)
    theta = rng.uniform(-np.pi, np.pi, size=pcount)
    dirs = sample_parameter_directions(rng, m, pcount)
    psi, phis, _ = simulate_vqc_tangent_batch(n, depth, family, theta, dirs, stable_seed("support|"+family+"|arch"))
    p, dp = probability_tangent_batch(psi, phis)
    fq = 4*np.sum(np.abs(phis)**2, axis=1).real
    return normalized_visible_scores_from_probabilities(p, dp, fq)


def test_u1_half_filled_support_and_ranks():
    n = 6
    info = _sample("U1-RZ-XY-line", n=n)
    assert int(info["support"].sum()) == math.comb(n, n//2)
    support_idx = np.flatnonzero(info["support"])
    _, r1 = walsh_readout_basis(info["p_support"], support_idx, n, 1)
    _, r2 = walsh_readout_basis(info["p_support"], support_idx, n, 2)
    assert r1 == n - 1
    assert r2 == math.comb(n, 2) - 1


@pytest.mark.parametrize("family", CORE_FAMILIES[:-1])
def test_generic_full_support_and_expected_low_weight_ranks(family):
    n = 6
    info = _sample(family, n=n)
    assert int(info["support"].sum()) == 2**n
    support_idx = np.flatnonzero(info["support"])
    _, r1 = walsh_readout_basis(info["p_support"], support_idx, n, 1)
    _, r2 = walsh_readout_basis(info["p_support"], support_idx, n, 2)
    assert r1 == n
    assert r2 == n + math.comb(n, 2)


def test_direct_retention_equals_trace_overlap():
    from qaccess.geometry import walsh_readout_basis, direct_readout_retention
    rng = np.random.default_rng(99)
    n = 4
    p = rng.random(2**n)
    p /= p.sum()
    support = np.arange(2**n)
    raw = rng.normal(size=(80, 2**n))
    sp = np.sqrt(p)
    raw = raw - (raw @ sp)[:, None] * sp[None, :]
    u = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    q, rank = walsh_readout_basis(p, support, n, 2)
    c_hat = (u.T @ u) / len(u)
    trace_value = float(np.trace(q.T @ c_hat @ q))
    diag = direct_readout_retention(u, p, support, n, 2)
    assert diag["rank"] == rank
    assert abs(diag["retention"] - trace_value) < 1e-12
