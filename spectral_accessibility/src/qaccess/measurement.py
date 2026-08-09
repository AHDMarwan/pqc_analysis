from __future__ import annotations

import numpy as np

from .gates import HADAMARD, apply_1q_batch, haar_u2


def rotate_measurement_basis(
    psi: np.ndarray,
    phis: np.ndarray,
    n: int,
    basis: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform state/tangents so a Z measurement realizes the requested basis."""
    if basis == "computational":
        return psi, phis
    combined = np.vstack([psi[None, :], phis])
    if basis == "x":
        gates = [HADAMARD] * n
    elif basis == "local_haar":
        rng = np.random.default_rng(seed)
        # Measuring in columns of U means applying U^dagger before Z measurement.
        gates = [haar_u2(rng).conj().T for _ in range(n)]
    else:
        raise ValueError(f"unknown measurement basis: {basis}")
    for q, gate in enumerate(gates):
        combined = apply_1q_batch(combined, gate, q, n)
    return combined[0], combined[1:]


def probability_tangent_batch(
    psi: np.ndarray, phis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = np.abs(psi) ** 2
    dp = 2.0 * np.real(np.conjugate(psi)[None, :] * phis)
    return p, dp


def apply_independent_bitflip_noise(
    p: np.ndarray, dp: np.ndarray, n: int, rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply independent symmetric classical bit-flip readout noise."""
    rate = float(rate)
    if rate < 0 or rate >= 0.5:
        raise ValueError("bit-flip rate must be in [0, 0.5)")
    if rate == 0:
        return p, dp

    def channel(arr: np.ndarray) -> np.ndarray:
        lead = arr.shape[:-1]
        x = arr.reshape(lead + (2,) * n)
        for axis in range(len(lead), len(lead) + n):
            x = (1.0 - rate) * x + rate * np.flip(x, axis=axis)
        return x.reshape(arr.shape)

    return channel(p), channel(dp)
