from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TopologyCorrelationResult:
    """Correlation study between topological and PQC diagnostic metrics.

    This result is experimental by design. Correlation does not establish
    predictive validity or causality; use it to generate and test hypotheses.
    """

    metric_names: Sequence[str]
    correlation_matrix: np.ndarray
    p_values: np.ndarray
    sample_size: int
    metadata: Dict[str, object] = field(default_factory=dict)

    def pair(self, left: str, right: str) -> Dict[str, float]:
        names = list(self.metric_names)
        if left not in names or right not in names:
            raise KeyError("requested metric is not present in this correlation study")
        i, j = names.index(left), names.index(right)
        return {
            "correlation": float(self.correlation_matrix[i, j]),
            "p_value": float(self.p_values[i, j]),
        }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom == 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denom)


def _permutation_p_value(x: np.ndarray, y: np.ndarray, observed: float, permutations: int, rng) -> float:
    if not np.isfinite(observed):
        return float("nan")
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(y)
        value = _pearson(x, permuted)
        if np.isfinite(value) and abs(value) >= abs(observed):
            exceedances += 1
    return float((exceedances + 1) / (permutations + 1))


def correlate_topology_with_diagnostics(
    records: Sequence[Mapping[str, float]],
    *,
    topology_metrics: Sequence[str],
    diagnostic_metrics: Sequence[str],
    method: str = "spearman",
    permutations: int = 2000,
    seed: int = 0,
) -> TopologyCorrelationResult:
    """Run a reproducible topology-vs-diagnostics correlation study.

    Parameters
    ----------
    records:
        Flat experiment records. Every selected metric must be finite in every
        retained record.
    topology_metrics:
        Metrics derived from persistent homology, e.g. persistence entropy or
        Betti/persistence summaries.
    diagnostic_metrics:
        Geometry/trainability/resource metrics to compare against topology.
    method:
        ``"spearman"`` (default, rank-based) or ``"pearson"``.
    permutations:
        Two-sided permutation-test repetitions for empirical p-values.

    Notes
    -----
    The function deliberately reports association only. A useful correlation
    must still be validated out-of-sample before being treated as predictive.
    """
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")

    names = tuple(topology_metrics) + tuple(diagnostic_metrics)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("selected metric names must be unique and contain at least two metrics")

    rows = []
    for record in records:
        if all(name in record and np.isfinite(float(record[name])) for name in names):
            rows.append([float(record[name]) for name in names])
    if len(rows) < 4:
        raise ValueError("at least four complete finite records are required")

    data = np.asarray(rows, dtype=float)
    transformed = data.copy()
    if method == "spearman":
        for column in range(transformed.shape[1]):
            transformed[:, column] = _rankdata(transformed[:, column])

    n_metrics = transformed.shape[1]
    corr = np.eye(n_metrics, dtype=float)
    p_values = np.zeros((n_metrics, n_metrics), dtype=float)
    rng = np.random.default_rng(seed)

    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            value = _pearson(transformed[:, i], transformed[:, j])
            p_value = _permutation_p_value(
                transformed[:, i], transformed[:, j], value, permutations, rng
            )
            corr[i, j] = corr[j, i] = value
            p_values[i, j] = p_values[j, i] = p_value

    return TopologyCorrelationResult(
        metric_names=names,
        correlation_matrix=corr,
        p_values=p_values,
        sample_size=int(data.shape[0]),
        metadata={
            "method": method,
            "permutations": int(permutations),
            "seed": int(seed),
            "topology_metrics": tuple(topology_metrics),
            "diagnostic_metrics": tuple(diagnostic_metrics),
            "experimental": True,
        },
    )
