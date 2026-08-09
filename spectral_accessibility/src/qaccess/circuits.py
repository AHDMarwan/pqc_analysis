from __future__ import annotations

import numpy as np

from .gates import (
    CNOT4,
    CZ4,
    H_XY,
    PAULI,
    apply_1q_batch,
    apply_2q_batch,
    haar_u4,
    rotation,
    xy_gate,
)

CORE_FAMILIES = (
    "RY-RZ-CZ-line",
    "SU2-CNOT-line",
    "SU2-HaarU4-brickwork",
    "U1-RZ-XY-line",
)

GENERIC_EXTENDED_FAMILIES = (
    "RY-RZ-CZ-line",
    "SU2-CNOT-line",
    "SU2-CZ-ring",
    "SU2-CZ-random-matching",
    "SU2-HaarU4-brickwork",
)

U1_EXTENDED_FAMILIES = (
    "U1-RZ-XY-line",
    "U1-RZ-XY-ring",
)

ALL_FAMILIES = GENERIC_EXTENDED_FAMILIES + U1_EXTENDED_FAMILIES


def family_group(family: str) -> str:
    return "u1" if family.startswith("U1-") else "generic"


def brickwork_pairs(n: int, layer: int) -> list[tuple[int, int]]:
    start = layer % 2
    return [(q, q + 1) for q in range(start, n - 1, 2)]


def ring_brickwork_pairs(n: int, layer: int) -> list[tuple[int, int]]:
    if n % 2:
        raise ValueError("ring brickwork currently requires even n")
    if layer % 2 == 0:
        return [(q, q + 1) for q in range(0, n, 2)]
    return [(q, q + 1) for q in range(1, n - 1, 2)] + [(n - 1, 0)]


def initial_state(n: int, family: str) -> np.ndarray:
    psi = np.zeros(2**n, dtype=np.complex128)
    if family.startswith("U1-"):
        if n % 2:
            raise ValueError("half-filling controls require even n")
        index = 0
        for q in range(0, n, 2):
            index |= 1 << (n - 1 - q)
        psi[index] = 1.0
    else:
        psi[0] = 1.0
    return psi


def _rotation_axes(family: str) -> tuple[str, ...]:
    if family == "RY-RZ-CZ-line":
        return ("Y", "Z")
    if family in GENERIC_EXTENDED_FAMILIES and family != "RY-RZ-CZ-line":
        return ("X", "Y", "Z")
    if family.startswith("U1-"):
        return ("Z",)
    raise ValueError(f"unknown family: {family}")


def parameter_count(n: int, depth: int, family: str) -> int:
    axes = _rotation_axes(family)
    total = depth * n * len(axes)
    if family == "U1-RZ-XY-line":
        total += sum(len(brickwork_pairs(n, layer)) for layer in range(depth))
    elif family == "U1-RZ-XY-ring":
        total += sum(len(ring_brickwork_pairs(n, layer)) for layer in range(depth))
    return total


def sample_parameters(
    rng: np.random.Generator, count: int, distribution: str = "uniform_pi"
) -> np.ndarray:
    if distribution == "uniform_pi":
        return rng.uniform(-np.pi, np.pi, size=count)
    if distribution == "normal_1":
        return rng.normal(0.0, 1.0, size=count)
    if distribution == "normal_0p1":
        return rng.normal(0.0, 0.1, size=count)
    raise ValueError(f"unknown parameter distribution: {distribution}")


def sample_parameter_directions(
    rng: np.random.Generator, count: int, p: int
) -> np.ndarray:
    v = rng.normal(size=(count, p))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("zero Gaussian parameter direction")
    return v / norms


def _random_matching(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    perm = rng.permutation(n)
    return [(int(perm[j]), int(perm[j + 1])) for j in range(0, n - 1, 2)]


def _u1_pairs(n: int, layer: int, family: str) -> list[tuple[int, int]]:
    if family == "U1-RZ-XY-line":
        return brickwork_pairs(n, layer)
    if family == "U1-RZ-XY-ring":
        return ring_brickwork_pairs(n, layer)
    raise ValueError(family)


def simulate_vqc_tangent_batch(
    n: int,
    depth: int,
    family: str,
    theta: np.ndarray,
    directions: np.ndarray,
    architecture_seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Propagate a state and analytic directional tangents.

    The four CORE_FAMILIES intentionally preserve the operation order of the
    confirmatory notebook so its reported table can be independently rerun.
    """
    pcount = parameter_count(n, depth, family)
    if len(theta) != pcount or directions.shape[1] != pcount:
        raise ValueError("theta/directions do not match parameter count")

    m = directions.shape[0]
    psi = initial_state(n, family)
    phis = np.zeros((m, 2**n), dtype=np.complex128)
    rng_arch = np.random.default_rng(architecture_seed)
    cursor = 0
    axes = _rotation_axes(family)

    for layer in range(depth):
        for q in range(n):
            for axis in axes:
                angle = float(theta[cursor])
                coeff = directions[:, cursor]
                gate = rotation(axis, angle)
                combined = np.vstack([psi[None, :], phis])
                combined = apply_1q_batch(combined, gate, q, n)
                psi_new, phis_new = combined[0], combined[1:]
                if np.any(coeff):
                    gpsi = apply_1q_batch(psi_new[None, :], PAULI[axis], q, n)[0]
                    phis_new += (-0.5j) * coeff[:, None] * gpsi[None, :]
                psi, phis = psi_new, phis_new
                cursor += 1

        if family.startswith("U1-"):
            for q1, q2 in _u1_pairs(n, layer, family):
                angle = float(theta[cursor])
                coeff = directions[:, cursor]
                gate = xy_gate(angle)
                combined = np.vstack([psi[None, :], phis])
                combined = apply_2q_batch(combined, gate, q1, q2, n)
                psi_new, phis_new = combined[0], combined[1:]
                if np.any(coeff):
                    gpsi = apply_2q_batch(psi_new[None, :], H_XY, q1, q2, n)[0]
                    phis_new += (-1j) * coeff[:, None] * gpsi[None, :]
                psi, phis = psi_new, phis_new
                cursor += 1

        if family == "RY-RZ-CZ-line":
            pairs = [(q, q + 1) for q in range(n - 1)]
            gates = [CZ4] * len(pairs)
        elif family == "SU2-CNOT-line":
            pairs = [(q, q + 1) for q in range(n - 1)]
            gates = [CNOT4] * len(pairs)
        elif family == "SU2-CZ-ring":
            pairs = [(q, q + 1) for q in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
            gates = [CZ4] * len(pairs)
        elif family == "SU2-CZ-random-matching":
            pairs = _random_matching(n, rng_arch)
            gates = [CZ4] * len(pairs)
        elif family == "SU2-HaarU4-brickwork":
            pairs = brickwork_pairs(n, layer)
            gates = [haar_u4(rng_arch) for _ in pairs]
        else:
            pairs, gates = [], []

        for (q1, q2), gate in zip(pairs, gates):
            combined = np.vstack([psi[None, :], phis])
            combined = apply_2q_batch(combined, gate, q1, q2, n)
            psi, phis = combined[0], combined[1:]

    if cursor != pcount:
        raise RuntimeError(f"parameter cursor mismatch {cursor} != {pcount}")

    overlaps = phis @ np.conjugate(psi)
    phis = phis - overlaps[:, None] * psi[None, :]
    return psi, phis, pcount
