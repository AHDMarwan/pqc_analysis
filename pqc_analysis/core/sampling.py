from typing import Optional

import numpy as np


def sample_parameters(
    n_params: int,
    n_samples: int,
    strategy: str = "uniform",
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Generate reproducible PQC parameter samples.

    Parameters
    ----------
    n_params:
        Number of trainable parameters in the circuit.
    n_samples:
        Number of parameter vectors to draw.
    strategy:
        ``"uniform"`` samples from [-pi, pi]. ``"normal"`` samples from
        N(0, pi^2).
    seed:
        Random seed. Use ``None`` for non-deterministic sampling.
    """
    if n_params <= 0:
        raise ValueError("n_params must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)

    if strategy == "uniform":
        return rng.uniform(-np.pi, np.pi, size=(n_samples, n_params))
    if strategy == "normal":
        return rng.normal(0.0, np.pi, size=(n_samples, n_params))

    raise ValueError("strategy must be either 'uniform' or 'normal'")
