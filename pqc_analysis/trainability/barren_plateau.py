from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Union

import numpy as np

from ..core.sampling import sample_parameters
from .gradients import gradient_statistics


@dataclass(frozen=True)
class BarrenPlateauScanResult:
    qubit_counts: np.ndarray
    gradient_variances: np.ndarray
    mean_abs_gradients: np.ndarray
    near_zero_fractions: np.ndarray
    log_variance_slope: float
    log_variance_intercept: float
    r_squared: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shows_exponential_suppression(self) -> bool:
        """Conservative heuristic for exponential gradient suppression.

        This is a diagnostic heuristic, not a proof of a barren plateau.
        Thresholds are stored in metadata and can be changed by callers.
        """
        min_r_squared = float(self.metadata.get("min_r_squared", 0.8))
        min_decay_rate = float(self.metadata.get("min_decay_rate", 0.1))
        return self.log_variance_slope <= -abs(min_decay_rate) and self.r_squared >= min_r_squared

    def summary(self) -> str:
        status = "consistent with exponential suppression" if self.shows_exponential_suppression else "no strong exponential suppression detected"
        return "\n".join(
            [
                "BARREN PLATEAU SCAN",
                "=" * 40,
                f"log-variance slope       {self.log_variance_slope:.4f}",
                f"R^2                      {self.r_squared:.4f}",
                f"Diagnosis                {status}",
            ]
        )


def _resolve_n_params(n_params: Union[int, Callable[[int], int]], n_qubits: int) -> int:
    value = n_params(n_qubits) if callable(n_params) else n_params
    value = int(value)
    if value <= 0:
        raise ValueError("n_params must resolve to a positive integer")
    return value


def barren_plateau_scan(
    gradient_factory: Callable[[int], Callable[[np.ndarray], Iterable[float]]],
    qubit_counts: Iterable[int],
    n_params: Union[int, Callable[[int], int]],
    *,
    samples: int = 100,
    init_strategy: str = "uniform",
    seed: int = 42,
    near_zero_tol: float = 1e-8,
    min_r_squared: float = 0.8,
    min_decay_rate: float = 0.1,
) -> BarrenPlateauScanResult:
    """Estimate how gradient variance scales with system size.

    Parameters
    ----------
    gradient_factory:
        Callable receiving ``n_qubits`` and returning a backend-specific
        gradient function for that circuit size.
    qubit_counts:
        System sizes to scan. At least two distinct sizes are required.
    n_params:
        Either a fixed parameter count or a callable ``n_params(n_qubits)``.

    Notes
    -----
    The routine fits ``log(Var[grad]) = a * n_qubits + b``. A negative slope
    with a strong linear fit is evidence consistent with exponential gradient
    suppression, but it is not by itself a mathematical proof of a barren
    plateau.
    """
    qubits = np.asarray(list(qubit_counts), dtype=int)
    if qubits.ndim != 1 or qubits.size < 2:
        raise ValueError("qubit_counts must contain at least two values")
    if np.any(qubits <= 0) or np.unique(qubits).size != qubits.size:
        raise ValueError("qubit_counts must contain distinct positive integers")
    if samples <= 0:
        raise ValueError("samples must be positive")

    variances = []
    mean_abs = []
    near_zero = []

    for index, n_qubits in enumerate(qubits):
        param_count = _resolve_n_params(n_params, int(n_qubits))
        theta_samples = sample_parameters(
            param_count,
            samples,
            strategy=init_strategy,
            seed=None if seed is None else seed + index,
        )
        gradient_fn = gradient_factory(int(n_qubits))
        stats = gradient_statistics(gradient_fn, theta_samples, near_zero_tol=near_zero_tol)
        variances.append(stats.gradient_variance)
        mean_abs.append(stats.mean_abs_gradient)
        near_zero.append(stats.near_zero_fraction)

    variances_array = np.asarray(variances, dtype=float)
    if np.any(variances_array <= 0) or not np.all(np.isfinite(variances_array)):
        raise ValueError("gradient variances must be positive and finite to fit log scaling")

    x = qubits.astype(float)
    y = np.log(variances_array)
    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = slope * x + intercept
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else (0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot)

    return BarrenPlateauScanResult(
        qubit_counts=qubits,
        gradient_variances=variances_array,
        mean_abs_gradients=np.asarray(mean_abs, dtype=float),
        near_zero_fractions=np.asarray(near_zero, dtype=float),
        log_variance_slope=float(slope),
        log_variance_intercept=float(intercept),
        r_squared=float(r_squared),
        metadata={
            "samples_per_size": int(samples),
            "init_strategy": init_strategy,
            "seed": seed,
            "near_zero_tol": float(near_zero_tol),
            "min_r_squared": float(min_r_squared),
            "min_decay_rate": float(min_decay_rate),
        },
    )
