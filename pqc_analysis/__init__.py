from .diagnostics import analyze
from .geometry import (
    compute_metric_tensor,
    condition_score,
    effective_dimension,
    metric_rank,
    metric_spectrum,
    redundant_parameter_ratio,
)
from .trainability import gradient_statistics

# Backward-compatible public API from v0.1.
from .geometry_analysis import pqc_geometry_analysis
from .topology_analysis import pqc_topology_analysis

__all__ = [
    "analyze",
    "compute_metric_tensor",
    "metric_spectrum",
    "metric_rank",
    "condition_score",
    "effective_dimension",
    "redundant_parameter_ratio",
    "gradient_statistics",
    "pqc_geometry_analysis",
    "pqc_topology_analysis",
]
