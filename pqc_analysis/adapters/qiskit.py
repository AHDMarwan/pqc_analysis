from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from ..trainability.barren_plateau import BarrenPlateauScanResult, barren_plateau_scan


def make_qiskit_gradient_fn(
    circuit,
    observable,
    *,
    estimator=None,
    parameters: Optional[Sequence] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a NumPy-facing parameter-shift gradient for a Qiskit circuit.

    Qiskit is an optional dependency. Install the ``qiskit`` extra before
    calling this function. By default a local ``StatevectorEstimator`` is used.
    A custom BaseEstimatorV2-compatible estimator can be supplied for other
    execution targets.
    """
    try:
        from qiskit.primitives import StatevectorEstimator
        from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
    except ImportError as exc:
        raise ImportError(
            "Qiskit support requires 'qiskit' and 'qiskit-algorithms'. "
            "Install pqc_analysis with the qiskit extra."
        ) from exc

    estimator = StatevectorEstimator(seed=42) if estimator is None else estimator
    gradient_engine = ParamShiftEstimatorGradient(estimator)
    selected_parameters = list(circuit.parameters) if parameters is None else list(parameters)
    if not selected_parameters:
        raise ValueError("circuit must contain at least one selected parameter")

    def gradient(theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype=float).reshape(-1)
        if values.size != len(selected_parameters):
            raise ValueError(
                f"expected {len(selected_parameters)} parameter values, got {values.size}"
            )
        result = gradient_engine.run(
            [circuit],
            [observable],
            [values.tolist()],
            parameters=[selected_parameters],
        ).result()
        return np.asarray(result.gradients[0], dtype=float).reshape(-1)

    return gradient


def qiskit_barren_plateau_scan(
    problem_factory: Callable[[int], Tuple[object, object]],
    qubit_counts: Iterable[int],
    n_params: Union[int, Callable[[int], int]],
    *,
    estimator_factory: Optional[Callable[[int], object]] = None,
    samples: int = 100,
    init_strategy: str = "uniform",
    seed: int = 42,
    near_zero_tol: float = 1e-8,
    min_r_squared: float = 0.8,
    min_decay_rate: float = 0.1,
) -> BarrenPlateauScanResult:
    """Run scaling diagnostics for Qiskit ``(circuit, observable)`` problems."""

    def gradient_factory(n_qubits: int):
        circuit, observable = problem_factory(n_qubits)
        estimator = None if estimator_factory is None else estimator_factory(n_qubits)
        return make_qiskit_gradient_fn(circuit, observable, estimator=estimator)

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
