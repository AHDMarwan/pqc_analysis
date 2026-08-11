import numpy as np

from .spectrum import metric_rank


def redundant_parameter_ratio(metric: np.ndarray, tol: float = 1e-12) -> float:
    """Fraction of parameter directions not resolved by the local metric rank."""
    metric = np.asarray(metric, dtype=float)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("metric must be a square matrix")
    n_params = metric.shape[0]
    if n_params == 0:
        return 0.0
    return float((n_params - metric_rank(metric, tol=tol)) / n_params)
