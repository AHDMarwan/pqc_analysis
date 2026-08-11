from typing import Callable, Optional

import numpy as np
import pennylane as qml

from ..core.result import AnalysisReport, GeometryResult
from ..core.sampling import sample_parameters
from ..geometry.metric import compute_metric_tensor
from ..geometry.redundancy import redundant_parameter_ratio
from ..geometry.spectrum import condition_score, effective_dimension, metric_rank
from ..trainability.gradients import gradient_statistics


def _geometry_summary(qnode, samples: np.ndarray, approximation: str, tol: float) -> GeometryResult:
    ranks = []
    redundancies = []
    conditions = []
    dimensions = []
    log_volumes = []

    for theta in samples:
        theta_qml = qml.numpy.array(theta, requires_grad=True)
        metric = compute_metric_tensor(qnode, theta_qml, approximation=approximation)
        ranks.append(metric_rank(metric, tol=tol))
        redundancies.append(redundant_parameter_ratio(metric, tol=tol))
        conditions.append(condition_score(metric, tol=tol))
        dimensions.append(effective_dimension(metric, tol=tol))

        sign, logdet = np.linalg.slogdet(metric)
        log_volumes.append(0.5 * logdet if sign > 0 else -np.inf)

    finite_log_volumes = np.asarray(log_volumes, dtype=float)
    finite_log_volumes = finite_log_volumes[np.isfinite(finite_log_volumes)]

    return GeometryResult(
        metric_rank=float(np.mean(ranks)),
        redundant_parameter_ratio=float(np.mean(redundancies)),
        condition_score=float(np.mean(conditions)),
        effective_dimension=float(np.mean(dimensions)),
        mean_log_volume=float(np.mean(finite_log_volumes)) if finite_log_volumes.size else float("-inf"),
        metadata={
            "metric_approximation": approximation,
            "n_samples": int(samples.shape[0]),
            "rank_std": float(np.std(ranks)),
            "condition_score_std": float(np.std(conditions)),
        },
    )


def analyze(
    circuit: Callable,
    n_qubits: int,
    n_params: int,
    samples: int = 100,
    *,
    parameter_samples: Optional[np.ndarray] = None,
    init_strategy: str = "uniform",
    seed: Optional[int] = 42,
    metric_approximation: str = "block-diag",
    gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    near_zero_tol: float = 1e-8,
    tol: float = 1e-12,
    device_name: str = "default.qubit",
) -> AnalysisReport:
    """Run unified PQC geometry and optional trainability diagnostics.

    Parameters
    ----------
    circuit:
        PennyLane-compatible quantum function accepting one parameter vector.
    n_qubits, n_params:
        Circuit size metadata.
    samples:
        Number of random parameter points when ``parameter_samples`` is omitted.
    gradient_fn:
        Optional callable returning a gradient vector for one parameter vector.
        Supplying it enables trainability diagnostics while keeping the core
        statistics independent of any autodiff backend.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if n_params <= 0:
        raise ValueError("n_params must be positive")
    if metric_approximation not in {"full", "block-diag", "diag"}:
        raise ValueError("metric_approximation must be 'full', 'block-diag', or 'diag'")

    if parameter_samples is None:
        parameter_samples = sample_parameters(n_params, samples, strategy=init_strategy, seed=seed)
    else:
        parameter_samples = np.asarray(parameter_samples, dtype=float)
        if parameter_samples.ndim != 2 or parameter_samples.shape[1] != n_params:
            raise ValueError("parameter_samples must have shape (n_samples, n_params)")

    # PennyLane's full metric tensor may use a Hadamard-test auxiliary wire.
    device_wires = n_qubits + 1 if metric_approximation == "full" else n_qubits
    device = qml.device(device_name, wires=device_wires)
    qnode = qml.QNode(circuit, device, interface="autograd", diff_method="best")

    geometry = _geometry_summary(qnode, parameter_samples, metric_approximation, tol)
    trainability = None
    if gradient_fn is not None:
        trainability = gradient_statistics(
            gradient_fn,
            parameter_samples,
            near_zero_tol=near_zero_tol,
        )

    return AnalysisReport(
        geometry=geometry,
        trainability=trainability,
        metadata={
            "n_qubits": int(n_qubits),
            "n_params": int(n_params),
            "device": device_name,
            "seed": seed,
        },
    )
