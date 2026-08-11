import numpy as np


def metric_spectrum(metric: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return non-negative eigenvalues of a Fubini-Study metric tensor."""
    values = np.linalg.eigvalsh(np.asarray(metric, dtype=float))
    values[np.abs(values) < tol] = 0.0
    return np.clip(values, 0.0, None)


def metric_rank(metric: np.ndarray, tol: float = 1e-12) -> int:
    """Numerical rank of the metric tensor."""
    return int(np.linalg.matrix_rank(np.asarray(metric, dtype=float), tol=tol))


def condition_score(metric: np.ndarray, tol: float = 1e-12) -> float:
    """Inverse spectral condition number in [0, 1].

    A value near zero indicates a poorly conditioned parameter manifold.
    """
    eigvals = metric_spectrum(metric, tol=tol)
    positive = eigvals[eigvals > tol]
    if positive.size == 0:
        return 0.0
    return float(np.min(positive) / np.max(positive))


def effective_dimension(metric: np.ndarray, regularizer: float = 1.0, tol: float = 1e-12) -> float:
    """Compute a regularized spectral effective dimension."""
    if regularizer <= 0:
        raise ValueError("regularizer must be positive")
    eigvals = metric_spectrum(metric, tol=tol)
    return float(np.sum(eigvals / (eigvals + regularizer)))
