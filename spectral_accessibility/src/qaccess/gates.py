from __future__ import annotations

import math
import numpy as np

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
PAULI = {"X": X, "Y": Y, "Z": Z}
H_XY = (np.kron(X, X) + np.kron(Y, Y)) / 4.0
CZ4 = np.diag([1, 1, 1, -1]).astype(np.complex128)
CNOT4 = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)


def rotation(axis: str, theta: float) -> np.ndarray:
    p = PAULI[axis]
    return math.cos(theta / 2.0) * I2 - 1j * math.sin(theta / 2.0) * p


def xy_gate(theta: float) -> np.ndarray:
    # exp[-i theta (XX+YY)/4]. 4 H_XY^2 is the projector onto span{|01>,|10>}.
    return (
        np.eye(4, dtype=np.complex128)
        + (math.cos(theta / 2.0) - 1.0) * (4.0 * H_XY @ H_XY)
        - 2j * math.sin(theta / 2.0) * H_XY
    )


def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    phases = np.where(np.abs(d) > 0, d / np.abs(d), 1.0)
    return q @ np.diag(np.conjugate(phases))


def haar_u4(rng: np.random.Generator) -> np.ndarray:
    return haar_unitary(4, rng)


def haar_u2(rng: np.random.Generator) -> np.ndarray:
    return haar_unitary(2, rng)


def apply_1q_batch(states: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    m = states.shape[0]
    tensor = states.reshape((m,) + (2,) * n)
    moved = np.moveaxis(tensor, q + 1, 1)
    out = np.einsum("ab,mb...->ma...", gate, moved, optimize=True)
    out = np.moveaxis(out, 1, q + 1)
    return out.reshape(m, -1)


def apply_2q_batch(
    states: np.ndarray, gate: np.ndarray, q1: int, q2: int, n: int
) -> np.ndarray:
    if q1 == q2:
        raise ValueError("q1 and q2 must be distinct")
    m = states.shape[0]
    tensor = states.reshape((m,) + (2,) * n)
    moved = np.moveaxis(tensor, (q1 + 1, q2 + 1), (1, 2))
    mat = moved.reshape(m, 4, -1)
    out = np.einsum("ab,mbk->mak", gate, mat, optimize=True)
    out = out.reshape((m, 2, 2) + (2,) * (n - 2))
    out = np.moveaxis(out, (1, 2), (q1 + 1, q2 + 1))
    return out.reshape(m, -1)
