from .adapters import (
    make_pennylane_gradient_fn,
    make_qiskit_gradient_fn,
    pennylane_barren_plateau_scan,
    qiskit_barren_plateau_scan,
)
from .diagnostics import analyze
from .geometry import (
    compute_metric_tensor,
    condition_score,
    effective_dimension,
    metric_rank,
    metric_spectrum,
    redundant_parameter_ratio,
)
from .trainability import BarrenPlateauScanResult, barren_plateau_scan, gradient_statistics

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
    "BarrenPlateauScanResult",
    "barren_plateau_scan",
    "make_pennylane_gradient_fn",
    "pennylane_barren_plateau_scan",
    "make_qiskit_gradient_fn",
    "qiskit_barren_plateau_scan",
    "pqc_geometry_analysis",
    "pqc_topology_analysis",
]
