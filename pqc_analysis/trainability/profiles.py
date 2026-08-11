from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GradientProfile:
    """Per-parameter and optional layer-wise gradient diagnostics."""

    mean_abs: np.ndarray
    variance: np.ndarray
    mean: np.ndarray
    near_zero_fraction: np.ndarray
    layer_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return int(self.mean_abs.size)

    def weakest_parameters(self, k: int = 5) -> Tuple[int, ...]:
        if k <= 0:
            raise ValueError("k must be positive")
        order = np.argsort(self.mean_abs)
        return tuple(int(i) for i in order[: min(k, self.n_params)])


def gradient_profile(
    gradient_fn,
    parameter_samples: np.ndarray,
    *,
    layer_groups: Optional[Mapping[str, Sequence[int]]] = None,
    near_zero_tol: float = 1e-8,
) -> GradientProfile:
    """Compute per-parameter and optional layer-wise gradient statistics.

    Parameters
    ----------
    gradient_fn:
        Callable mapping one parameter vector to one gradient vector.
    parameter_samples:
        Array with shape ``(n_samples, n_params)``.
    layer_groups:
        Optional mapping from layer names to parameter indices.
    near_zero_tol:
        Absolute gradient threshold used only for the near-zero statistic.
    """
    samples = np.asarray(parameter_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("parameter_samples must have shape (n_samples, n_params)")
    if near_zero_tol < 0:
        raise ValueError("near_zero_tol must be non-negative")

    gradients = []
    for theta in samples:
        grad = np.asarray(gradient_fn(theta), dtype=float).reshape(-1)
        if grad.size != samples.shape[1]:
            raise ValueError("gradient_fn output size must match the number of parameters")
        if not np.all(np.isfinite(grad)):
            raise ValueError("gradient_fn returned non-finite values")
        gradients.append(grad)

    matrix = np.stack(gradients, axis=0)
    layer_stats: Dict[str, Dict[str, float]] = {}

    if layer_groups is not None:
        n_params = matrix.shape[1]
        seen = set()
        for name, indices in layer_groups.items():
            idx = np.asarray(tuple(indices), dtype=int)
            if idx.size == 0:
                raise ValueError(f"layer group {name!r} must contain at least one index")
            if np.any(idx < 0) or np.any(idx >= n_params):
                raise ValueError(f"layer group {name!r} contains an out-of-range parameter index")
            duplicate = seen.intersection(int(i) for i in idx)
            if duplicate:
                raise ValueError("layer_groups must not overlap")
            seen.update(int(i) for i in idx)
            block = matrix[:, idx]
            layer_stats[str(name)] = {
                "mean_abs_gradient": float(np.mean(np.abs(block))),
                "gradient_variance": float(np.var(block)),
                "gradient_norm": float(np.mean(np.linalg.norm(block, axis=1))),
                "near_zero_fraction": float(np.mean(np.abs(block) <= near_zero_tol)),
                "n_params": float(idx.size),
            }

    return GradientProfile(
        mean_abs=np.mean(np.abs(matrix), axis=0),
        variance=np.var(matrix, axis=0),
        mean=np.mean(matrix, axis=0),
        near_zero_fraction=np.mean(np.abs(matrix) <= near_zero_tol, axis=0),
        layer_statistics=layer_stats,
        metadata={
            "n_samples": int(matrix.shape[0]),
            "n_params": int(matrix.shape[1]),
            "near_zero_tol": float(near_zero_tol),
        },
    )
