from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .benchmark_matrix import BenchmarkMatrixCase, BenchmarkMatrixConfig, build_benchmark_matrix, matrix_to_study_specs
from .study import TopologyStudyResult, run_topology_diagnostic_study
from .topology_correlations import TopologyCorrelationResult


DEFAULT_TOPOLOGY_METRICS = (
    "h0_persistence_entropy",
    "h0_total_persistence",
    "h1_persistence_entropy",
    "h1_total_persistence",
    "h1_max_persistence",
    "h1_feature_count",
)

DEFAULT_DIAGNOSTIC_METRICS = (
    "gradient_variance",
    "mean_abs_gradient",
    "near_zero_fraction",
    "metric_rank",
    "effective_dimension",
    "redundant_parameter_ratio",
    "condition_score",
)


@dataclass(frozen=True)
class MatrixExperimentResult:
    cases: Sequence[BenchmarkMatrixCase]
    study: TopologyStudyResult
    correlations: TopologyCorrelationResult
    metadata: Dict[str, object] = field(default_factory=dict)

    def records(self) -> List[Dict[str, object]]:
        case_map = {case.name: case for case in self.cases}
        output: List[Dict[str, object]] = []
        for row in self.study.records:
            record = dict(row)
            case = case_map.get(str(record.get("architecture")))
            if case is not None:
                record.update(
                    {
                        "ansatz_family": case.ansatz_family,
                        "depth": case.depth,
                        "cost_type": case.cost_type,
                    }
                )
            output.append(record)
        return output

    def correlation_records(self) -> List[Dict[str, object]]:
        """Return only topology-vs-diagnostic pairs as flat CSV-friendly rows."""
        topology_metrics = tuple(self.correlations.metadata.get("topology_metrics", ()))
        diagnostic_metrics = tuple(self.correlations.metadata.get("diagnostic_metrics", ()))
        method = str(self.correlations.metadata.get("method", "unknown"))

        rows: List[Dict[str, object]] = []
        for topology_metric in topology_metrics:
            for diagnostic_metric in diagnostic_metrics:
                pair = self.correlations.pair(topology_metric, diagnostic_metric)
                rows.append(
                    {
                        "topology_metric": topology_metric,
                        "diagnostic_metric": diagnostic_metric,
                        "correlation": pair["correlation"],
                        "p_value": pair["p_value"],
                        "n": self.correlations.sample_size,
                        "method": method,
                    }
                )
        return rows


def run_benchmark_matrix_experiment(
    config: BenchmarkMatrixConfig = BenchmarkMatrixConfig(),
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    geometry_samples: int = 30,
    topology_samples: int = 60,
    topology_max_dim: int = 1,
    topology_metrics: Sequence[str] = DEFAULT_TOPOLOGY_METRICS,
    diagnostic_metrics: Sequence[str] = DEFAULT_DIAGNOSTIC_METRICS,
    correlation_method: str = "spearman",
    permutations: int = 2000,
    correlation_seed: int = 1234,
    init_strategy: str = "uniform",
    metric_approximation: str = "block-diag",
    device_name: str = "default.qubit",
) -> MatrixExperimentResult:
    """Run the full factorial topology/geometry/trainability benchmark matrix."""
    cases = build_benchmark_matrix(config, device_name=device_name)
    study = run_topology_diagnostic_study(
        matrix_to_study_specs(cases),
        seeds=seeds,
        geometry_samples=geometry_samples,
        topology_samples=topology_samples,
        topology_max_dim=topology_max_dim,
        init_strategy=init_strategy,
        metric_approximation=metric_approximation,
        device_name=device_name,
    )
    correlations = study.correlate(
        topology_metrics=topology_metrics,
        diagnostic_metrics=diagnostic_metrics,
        method=correlation_method,
        permutations=permutations,
        seed=correlation_seed,
    )
    return MatrixExperimentResult(
        cases=cases,
        study=study,
        correlations=correlations,
        metadata={
            "n_cases": len(cases),
            "n_records": len(study.records),
            "topology_metrics": tuple(topology_metrics),
            "diagnostic_metrics": tuple(diagnostic_metrics),
            "correlation_method": correlation_method,
            "permutations": int(permutations),
            "experimental": True,
        },
    )
