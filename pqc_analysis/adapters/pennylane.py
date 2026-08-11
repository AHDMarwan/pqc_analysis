from typing import Callable, Iterable, Union

import numpy as np
import pennylane as qml

from ..trainability.barren_plateau import BarrenPlateauScanResult, barren_plateau_scan


def make_pennylane_gradient_fn(
    cost_circuit: Callable,
    n_qubits: int,
    *,
    device_name: str = "default.qubit",
    diff_method: str = "best",
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a NumPy-facing gradient function from a scalar PennyLane circuit.

    ``cost_circuit`` must accept one trainable parameter vector and return a
    scalar expectation value or another differentiable scalar measurement.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")

    device = qml.device(device_name, wires=n_qubits)
    qnode = qml.QNode(cost_circuit, device, interface="autograd", diff_method=diff_method)
    grad_fn = qml.grad(qnode)

    def gradient(theta: np.ndarray) -> np.ndarray:
        params = qml.numpy.array(theta, requires_grad=True)
        grad = grad_fn(params)
        return np.asarray(qml.math.toarray(grad), dtype=float).reshape(-1)

    return gradient


def pennylane_barren_plateau_scan(
    circuit_factory: Callable[[int], Callable],
    qubit_counts: Iterable[int],
    n_params: Union[int, Callable[[int], int]],
    *,
    samples: int = 100,
    init_strategy: str = "uniform",
    seed: int = 42,
    near_zero_tol: float = 1e-8,
    device_name: str = "default.qubit",
    diff_method: str = "best",
    min_r_squared: float = 0.8,
    min_decay_rate: float = 0.1,
) -> BarrenPlateauScanResult:
    """Run barren-plateau scaling diagnostics for PennyLane cost circuits.

    ``circuit_factory(n_qubits)`` must return a scalar-valued PennyLane quantum
    function whose single argument is the trainable parameter vector.
    """

    def gradient_factory(n_qubits: int):
        circuit = circuit_factory(n_qubits)
        return make_pennylane_gradient_fn(
            circuit,
            n_qubits,
            device_name=device_name,
            diff_method=diff_method,
        )

    return barren_plateau_scan(
        gradient_factory,
        qubit_counts,
        n_params,
        samples=samples,
        init_strategy=init_strategy,
        seed=seed,
        near_zero_tol=near_zero_tol,
        min_r_squared=min_r_squared,
        min_decay_rate=min_decay_rate,
    )
