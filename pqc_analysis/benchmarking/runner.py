from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..core.result import AnalysisReport
from ..diagnostics.analyzer import analyze
from ..resources import estimate_gradient_resources


@dataclass(frozen=True)
class PQCSpec:
    """Description of one PennyLane-compatible PQC benchmark case."""

    name: str
    circuit: Callable
    n_qubits: int
    n_params: int
    gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    gradient_method: Optional[str] = None
    shots_per_circuit: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.n_qubits <= 0 or self.n_params <= 0:
            raise ValueError("n_qubits and n_params must be positive")
        if self.shots_per_circuit is not None and self.shots_per_circuit <= 0:
            raise ValueError("shots_per_circuit must be positive when provided")


@dataclass(frozen=True)
class BenchmarkRun:
    spec_name: str
    seed: int
    report: AnalysisReport
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    runs: Sequence[BenchmarkRun]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_records(self) -> List[Dict[str, Any]]:
        """Return flat, dataframe/CSV-friendly records for every run."""
        records: List[Dict[str, Any]] = []
        for run in self.runs:
            report = run.report
            row: Dict[str, Any] = {
                "architecture": run.spec_name,
                "seed": run.seed,
                "n_qubits": report.metadata.get("n_qubits"),
                "n_params": report.metadata.get("n_params"),
            }
            if report.geometry is not None:
                row.update(
                    {
                        "metric_rank": report.geometry.metric_rank,
                        "redundant_parameter_ratio": report.geometry.redundant_parameter_ratio,
                        "condition_score": report.geometry.condition_score,
                        "effective_dimension": report.geometry.effective_dimension,
                        "mean_log_volume": report.geometry.mean_log_volume,
                    }
                )
            if report.trainability is not None:
                row.update(
                    {
                        "mean_abs_gradient": report.trainability.mean_abs_gradient,
                        "gradient_variance": report.trainability.gradient_variance,
                        "gradient_norm": report.trainability.gradient_norm,
                        "near_zero_fraction": report.trainability.near_zero_fraction,
                    }
                )
            row.update(run.metadata)
            records.append(row)
        return records

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """Aggregate numeric metrics by architecture using mean and std."""
        records = self.to_records()
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in records:
            grouped.setdefault(str(row["architecture"]), []).append(row)

        excluded = {"architecture", "seed", "n_qubits", "n_params"}
        output: Dict[str, Dict[str, float]] = {}
        for architecture, rows in grouped.items():
            metric_names = sorted(
                {
                    key
                    for row in rows
                    for key, value in row.items()
                    if key not in excluded and isinstance(value, (int, float, np.integer, np.floating))
                }
            )
            summary: Dict[str, float] = {"runs": float(len(rows))}
            for metric in metric_names:
                values = np.asarray(
                    [row[metric] for row in rows if metric in row and np.isfinite(row[metric])],
                    dtype=float,
                )
                if values.size:
                    summary[f"{metric}_mean"] = float(np.mean(values))
                    summary[f"{metric}_std"] = float(np.std(values))
            output[architecture] = summary
        return output


def benchmark(
    specs: Iterable[PQCSpec],
    *,
    seeds: Iterable[int] = (0, 1, 2),
    samples: int = 50,
    init_strategy: str = "uniform",
    metric_approximation: str = "block-diag",
    near_zero_tol: float = 1e-8,
    tol: float = 1e-12,
    device_name: str = "default.qubit",
) -> BenchmarkResult:
    """Benchmark PQC architectures under a shared, reproducible protocol."""
    specs_list = list(specs)
    seed_list = [int(seed) for seed in seeds]
    if not specs_list:
        raise ValueError("specs must contain at least one PQCSpec")
    if not seed_list:
        raise ValueError("seeds must contain at least one value")
    if samples <= 0:
        raise ValueError("samples must be positive")

    names = [spec.name for spec in specs_list]
    if len(set(names)) != len(names):
        raise ValueError("PQCSpec names must be unique")

    runs: List[BenchmarkRun] = []
    for spec in specs_list:
        resource_metadata: Dict[str, Any] = {}
        if spec.gradient_method is not None:
            resources = estimate_gradient_resources(
                spec.n_params,
                gradient_method=spec.gradient_method,
                shots_per_circuit=spec.shots_per_circuit,
            )
            resource_metadata = {
                "gradient_method": resources.gradient_method,
                "circuit_evaluations_per_step": resources.circuit_evaluations_per_step,
                "shots_per_step": resources.shots_per_step,
            }

        for seed in seed_list:
            report = analyze(
                spec.circuit,
                n_qubits=spec.n_qubits,
                n_params=spec.n_params,
                samples=samples,
                init_strategy=init_strategy,
                seed=seed,
                metric_approximation=metric_approximation,
                gradient_fn=spec.gradient_fn,
                near_zero_tol=near_zero_tol,
                tol=tol,
                device_name=device_name,
            )
            runs.append(
                BenchmarkRun(
                    spec_name=spec.name,
                    seed=seed,
                    report=report,
                    metadata={**dict(spec.metadata), **resource_metadata},
                )
            )

    return BenchmarkResult(
        runs=tuple(runs),
        metadata={
            "samples_per_run": int(samples),
            "seeds": tuple(seed_list),
            "init_strategy": init_strategy,
            "metric_approximation": metric_approximation,
            "device": device_name,
        },
    )
