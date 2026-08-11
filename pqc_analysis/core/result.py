from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GeometryResult:
    metric_rank: float
    redundant_parameter_ratio: float
    condition_score: float
    effective_dimension: float
    mean_log_volume: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainabilityResult:
    mean_abs_gradient: float
    gradient_variance: float
    gradient_norm: float
    near_zero_fraction: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisReport:
    geometry: Optional[GeometryResult] = None
    trainability: Optional[TrainabilityResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["PQC ANALYSIS", "=" * 40]

        if self.geometry is not None:
            g = self.geometry
            lines.extend(
                [
                    "Geometry",
                    f"  Metric rank              {g.metric_rank:.4f}",
                    f"  Redundant param ratio    {g.redundant_parameter_ratio:.4f}",
                    f"  Condition score          {g.condition_score:.4e}",
                    f"  Effective dimension      {g.effective_dimension:.4f}",
                    f"  Mean log volume          {g.mean_log_volume:.4f}",
                ]
            )

        if self.trainability is not None:
            t = self.trainability
            lines.extend(
                [
                    "Trainability",
                    f"  Mean |gradient|          {t.mean_abs_gradient:.4e}",
                    f"  Gradient variance        {t.gradient_variance:.4e}",
                    f"  Gradient norm            {t.gradient_norm:.4e}",
                    f"  Near-zero fraction       {t.near_zero_fraction:.4f}",
                ]
            )

        return "\n".join(lines)
