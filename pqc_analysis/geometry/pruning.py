from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PruningPlan:
    """Conservative parameter-pruning proposal derived from metric null spaces."""

    candidate_indices: Tuple[int, ...]
    redundancy_scores: np.ndarray
    estimated_rank: int
    n_params: int
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def candidate_fraction(self) -> float:
        return float(len(self.candidate_indices) / self.n_params) if self.n_params else 0.0


def _validate_metric(metric: np.ndarray) -> np.ndarray:
    arr = np.asarray(metric, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("metric must be a square matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError("metric contains non-finite values")
    return 0.5 * (arr + arr.T)


def geometry_pruning_plan(
    metric: np.ndarray,
    *,
    tol: float = 1e-10,
    max_candidates: Optional[int] = None,
) -> PruningPlan:
    """Rank parameters by their participation in the metric's null subspace.

    The score for parameter ``i`` is the diagonal element of the orthogonal
    projector onto the numerically null eigenspace. Scores near one mean that
    the local coordinate direction lies predominantly in a redundant subspace.

    This function proposes candidates only. Removing a parameter changes the
    circuit family, so callers should re-run geometry and task-level validation
    after pruning.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    arr = _validate_metric(metric)
    n_params = arr.shape[0]
    if max_candidates is not None and max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")

    eigvals, eigvecs = np.linalg.eigh(arr)
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    null_mask = eigvals <= tol * scale
    rank = int(np.count_nonzero(~null_mask))

    if not np.any(null_mask):
        scores = np.zeros(n_params, dtype=float)
        candidates: Tuple[int, ...] = ()
    else:
        null_vectors = eigvecs[:, null_mask]
        scores = np.sum(null_vectors ** 2, axis=1)
        nullity = int(np.count_nonzero(null_mask))
        limit = nullity if max_candidates is None else min(int(max_candidates), nullity)
        order = np.argsort(-scores)
        candidates = tuple(int(i) for i in order[:limit] if scores[i] > tol)

    return PruningPlan(
        candidate_indices=candidates,
        redundancy_scores=scores,
        estimated_rank=rank,
        n_params=n_params,
        metadata={
            "tol": float(tol),
            "nullity": int(n_params - rank),
            "eigenvalues": eigvals,
        },
    )


def aggregate_pruning_plan(
    metrics: Iterable[np.ndarray],
    *,
    tol: float = 1e-10,
    max_candidates: Optional[int] = None,
) -> PruningPlan:
    """Build a pruning plan robust across several sampled metric tensors.

    Redundancy scores are averaged over samples. The candidate count is bounded
    by the minimum observed nullity so the proposal is not driven by a single
    unusually singular parameter point.
    """
    metric_list = [_validate_metric(metric) for metric in metrics]
    if not metric_list:
        raise ValueError("metrics must contain at least one metric tensor")
    shape = metric_list[0].shape
    if any(metric.shape != shape for metric in metric_list):
        raise ValueError("all metrics must have the same shape")

    plans = [geometry_pruning_plan(metric, tol=tol) for metric in metric_list]
    scores = np.mean(np.stack([plan.redundancy_scores for plan in plans]), axis=0)
    min_nullity = min(int(plan.metadata["nullity"]) for plan in plans)
    limit = min_nullity if max_candidates is None else min(int(max_candidates), min_nullity)
    order = np.argsort(-scores)
    candidates = tuple(int(i) for i in order[:limit] if scores[i] > tol)

    return PruningPlan(
        candidate_indices=candidates,
        redundancy_scores=scores,
        estimated_rank=int(round(np.mean([plan.estimated_rank for plan in plans]))),
        n_params=shape[0],
        metadata={
            "tol": float(tol),
            "n_metric_samples": len(plans),
            "minimum_nullity": int(min_nullity),
            "rank_min": int(min(plan.estimated_rank for plan in plans)),
            "rank_max": int(max(plan.estimated_rank for plan in plans)),
        },
    )
