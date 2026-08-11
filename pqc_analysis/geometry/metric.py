from typing import Literal

import numpy as np
import pennylane as qml

MetricApproximation = Literal["full", "block-diag", "diag"]


def compute_metric_tensor(qnode, theta, approximation: MetricApproximation = "block-diag") -> np.ndarray:
    """Compute the Fubini-Study metric tensor for a PennyLane QNode.

    The approximation is explicit instead of hard-coded so analysis code can
    trade accuracy for execution cost.
    """
    if approximation not in {"full", "block-diag", "diag"}:
        raise ValueError("approximation must be 'full', 'block-diag', or 'diag'")

    metric_fn = qml.metric_tensor(qnode, approx=None if approximation == "full" else approximation)
    metric = metric_fn(theta)
    return np.asarray(qml.math.toarray(metric), dtype=float)
