from .adapters import (
    make_pennylane_gradient_fn,
    make_qiskit_gradient_fn,
    pennylane_barren_plateau_scan,
    qiskit_barren_plateau_scan,
)
from .diagnostics import DiagnosticFinding, analyze, diagnose
from .geometry import (
    compute_metric_tensor,
    condition_score,
    effective_dimension,
    metric_rank,
    metric_spectrum,
    redundant_parameter_ratio,
)
from .trainability import BarrenPlateauScanResult, barren_plateau_scan, gradient_statistics


def pqc_geometry_analysis(*args, **kwargs):
    """Backward-compatible v0.1 geometry entry point.

    Legacy dependencies are imported only when this function is called so the
    v0.2 core remains lightweight.
    """
    try:
        from .geometry_analysis import pqc_geometry_analysis as legacy_geometry_analysis
    except ImportError as exc:
        raise ImportError(
            "The legacy geometry API requires optional dependencies. "
            "Install pqc_analysis with the 'legacy' extra."
        ) from exc
    return legacy_geometry_analysis(*args, **kwargs)


def pqc_topology_analysis(*args, **kwargs):
    """Backward-compatible v0.1 topology entry point with lazy dependencies."""
    try:
        from .topology_analysis import pqc_topology_analysis as legacy_topology_analysis
    except ImportError as exc:
        raise ImportError(
            "Topological analysis requires optional dependencies. "
            "Install pqc_analysis with the 'tda' extra."
        ) from exc
    return legacy_topology_analysis(*args, **kwargs)


__all__ = [
    "analyze",
    "diagnose",
    "DiagnosticFinding",
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
