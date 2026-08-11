from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class PersistenceSummary:
    entropy: float
    total_persistence: float
    max_persistence: float
    feature_count: int
    mean_persistence: float
    metadata: Dict[str, object] = field(default_factory=dict)


def summarize_persistence_diagram(diagram: np.ndarray) -> PersistenceSummary:
    """Summarize one persistence diagram without requiring plotting libraries.

    Infinite-death features are excluded from lifetime-based statistics because
    their persistence is not finite on the sampled filtration.
    """
    arr = np.asarray(diagram, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("diagram must have shape (n_features, 2)")
    if arr.size == 0:
        return PersistenceSummary(0.0, 0.0, 0.0, 0, 0.0, {"finite_features": 0})

    lifetimes = arr[:, 1] - arr[:, 0]
    finite = np.isfinite(lifetimes) & (lifetimes > 0)
    values = lifetimes[finite]
    if values.size == 0:
        return PersistenceSummary(0.0, 0.0, 0.0, 0, 0.0, {"finite_features": 0})

    total = float(np.sum(values))
    probabilities = values / total
    entropy = float(-np.sum(probabilities * np.log(probabilities))) if total > 0 else 0.0
    return PersistenceSummary(
        entropy=entropy,
        total_persistence=total,
        max_persistence=float(np.max(values)),
        feature_count=int(values.size),
        mean_persistence=float(np.mean(values)),
        metadata={"finite_features": int(values.size)},
    )


def flatten_persistence_summaries(diagrams: Sequence[np.ndarray]) -> Dict[str, float]:
    """Return H0/H1/... persistence summaries as one flat record."""
    output: Dict[str, float] = {}
    for dimension, diagram in enumerate(diagrams):
        summary = summarize_persistence_diagram(diagram)
        prefix = f"h{dimension}"
        output[f"{prefix}_persistence_entropy"] = summary.entropy
        output[f"{prefix}_total_persistence"] = summary.total_persistence
        output[f"{prefix}_max_persistence"] = summary.max_persistence
        output[f"{prefix}_feature_count"] = float(summary.feature_count)
        output[f"{prefix}_mean_persistence"] = summary.mean_persistence
    return output
