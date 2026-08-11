from typing import Callable, Iterable

import numpy as np

from ..core.result import TrainabilityResult


def gradient_statistics(
    gradient_fn: Callable[[np.ndarray], Iterable[float]],
    parameter_samples: np.ndarray,
    near_zero_tol: float = 1e-8,
) -> TrainabilityResult:
    """Aggregate gradient-based trainability diagnostics over parameter samples.

    ``gradient_fn`` is backend-agnostic: it only needs to accept one parameter
    vector and return a gradient vector. Backend adapters can therefore provide
    PennyLane, Qiskit, JAX, or other implementations later without changing the
    statistical analysis layer.
    """
    samples = np.asarray(parameter_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("parameter_samples must have shape (n_samples, n_params)")
    if near_zero_tol < 0:
        raise ValueError("near_zero_tol must be non-negative")

    gradients = []
    for theta in samples:
        grad = np.asarray(gradient_fn(theta), dtype=float).reshape(-1)
        if grad.size == 0:
            raise ValueError("gradient_fn returned an empty gradient")
        if not np.all(np.isfinite(grad)):
            raise ValueError("gradient_fn returned non-finite values")
        gradients.append(grad)

    sizes = {grad.size for grad in gradients}
    if len(sizes) != 1:
        raise ValueError("gradient_fn must return gradients with a consistent shape")

    matrix = np.stack(gradients, axis=0)
    flat = matrix.reshape(-1)
    sample_norms = np.linalg.norm(matrix, axis=1)

    return TrainabilityResult(
        mean_abs_gradient=float(np.mean(np.abs(flat))),
        gradient_variance=float(np.var(flat)),
        gradient_norm=float(np.mean(sample_norms)),
        near_zero_fraction=float(np.mean(np.abs(flat) <= near_zero_tol)),
        metadata={
            "n_samples": int(matrix.shape[0]),
            "n_params": int(matrix.shape[1]),
            "near_zero_tol": float(near_zero_tol),
            "per_parameter_variance": np.var(matrix, axis=0),
        },
    )
