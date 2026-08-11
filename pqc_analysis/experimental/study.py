from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..core.sampling import sample_parameters
from ..diagnostics.analyzer import analyze
from .topology_correlations import TopologyCorrelationResult, correlate_topology_with_diagnostics
from .topology_state_space import TopologyStateSpaceResult, analyze_topological_state_space


@dataclass(frozen=True)
class TopologyStudySpec:
    name: str
    circuit: Callable
    n_qubits: int
    n_params: int
    gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyStudyResult:
    records: Sequence[Dict[str, float]]
    topology_results: Sequence[TopologyStateSpaceResult]
    metadata: Dict[str, object] = field(default_factory=dict)

    def correlate(
        self,
        *,
        topology_metrics: Sequence[str],
        diagnostic_metrics: Sequence[str],
        method: str = "spearman",
        permutations: int = 2000,
        seed: int = 0,
    ) -> TopologyCorrelationResult:
        return correlate_topology_with_diagnostics(
            self.records,
            topology_metrics=topology_metrics,
            diagnostic_metrics=diagnostic_metrics,
            method=method,
            permutations=permutations,
            seed=seed,
        )


def run_topology_diagnostic_study(
    specs: Iterable[TopologyStudySpec],
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    geometry_samples: int = 30,
    topology_samples: int = 60,
    topology_max_dim: int = 1,
    init_strategy: str = "uniform",
    metric_approximation: str = "block-diag",
    near_zero_tol: float = 1e-8,
    device_name: str = "default.qubit",
) -> TopologyStudyResult:
    """Run matched topology, geometry and trainability measurements.

    Each architecture/seed pair is evaluated under an explicitly recorded
    protocol. Topology and geometry use independently sampled parameter sets
    generated from the same seed and initialization law to avoid accidental
    coupling through reused observations while preserving reproducibility.
    """
    specs = list(specs)
    seeds = [int(seed) for seed in seeds]
    if not specs or not seeds:
        raise ValueError("specs and seeds must be non-empty")
    if geometry_samples <= 0 or topology_samples <= 1:
        raise ValueError("geometry_samples must be positive and topology_samples > 1")

    records: List[Dict[str, float]] = []
    topology_results: List[TopologyStateSpaceResult] = []

    for spec in specs:
        if spec.n_qubits <= 0 or spec.n_params <= 0:
            raise ValueError("spec qubit and parameter counts must be positive")
        for seed in seeds:
            report = analyze(
                spec.circuit,
                n_qubits=spec.n_qubits,
                n_params=spec.n_params,
                samples=geometry_samples,
                init_strategy=init_strategy,
                seed=seed,
                metric_approximation=metric_approximation,
                gradient_fn=spec.gradient_fn,
                near_zero_tol=near_zero_tol,
                device_name=device_name,
            )

            theta_topology = sample_parameters(
                topology_samples,
                spec.n_params,
                strategy=init_strategy,
                seed=seed + 1_000_003,
            )
            topology = analyze_topological_state_space(
                spec.circuit,
                n_qubits=spec.n_qubits,
                n_params=spec.n_params,
                max_dim=topology_max_dim,
                seed=seed,
                parameter_samples=theta_topology,
                init_strategy=init_strategy,
                device_name=device_name,
            )
            topology_results.append(topology)

            row: Dict[str, float] = {
                "architecture": spec.name,
                "seed": float(seed),
                "n_qubits": float(spec.n_qubits),
                "n_params": float(spec.n_params),
                **topology.metrics,
            }
            if report.geometry is not None:
                row.update(
                    {
                        "metric_rank": float(report.geometry.metric_rank),
                        "redundant_parameter_ratio": float(report.geometry.redundant_parameter_ratio),
                        "condition_score": float(report.geometry.condition_score),
                        "effective_dimension": float(report.geometry.effective_dimension),
                    }
                )
            if report.trainability is not None:
                row.update(
                    {
                        "mean_abs_gradient": float(report.trainability.mean_abs_gradient),
                        "gradient_variance": float(report.trainability.gradient_variance),
                        "gradient_norm": float(report.trainability.gradient_norm),
                        "near_zero_fraction": float(report.trainability.near_zero_fraction),
                    }
                )
            for key, value in spec.metadata.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    row[str(key)] = float(value)
            records.append(row)

    return TopologyStudyResult(
        records=tuple(records),
        topology_results=tuple(topology_results),
        metadata={
            "geometry_samples": int(geometry_samples),
            "topology_samples": int(topology_samples),
            "topology_max_dim": int(topology_max_dim),
            "init_strategy": init_strategy,
            "metric_approximation": metric_approximation,
            "seeds": tuple(seeds),
            "experimental": True,
        },
    )
