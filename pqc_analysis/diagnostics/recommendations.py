from dataclasses import dataclass
from typing import Dict, List, Optional

from ..core.result import AnalysisReport


@dataclass(frozen=True)
class DiagnosticFinding:
    code: str
    severity: str
    message: str
    suggestion: str


def diagnose(
    report: AnalysisReport,
    *,
    redundancy_threshold: float = 0.25,
    condition_threshold: float = 1e-3,
    near_zero_threshold: float = 0.5,
    rank_fraction_threshold: float = 0.75,
) -> List[DiagnosticFinding]:
    """Generate transparent heuristic findings from an analysis report.

    The thresholds are explicit and configurable. Findings are engineering
    diagnostics, not mathematical proofs about trainability or quantum
    advantage.
    """
    thresholds: Dict[str, float] = {
        "redundancy_threshold": redundancy_threshold,
        "condition_threshold": condition_threshold,
        "near_zero_threshold": near_zero_threshold,
        "rank_fraction_threshold": rank_fraction_threshold,
    }
    for name, value in thresholds.items():
        if not 0.0 <= value <= 1.0 and name != "condition_threshold":
            raise ValueError(f"{name} must be in [0, 1]")
        if name == "condition_threshold" and value < 0.0:
            raise ValueError("condition_threshold must be non-negative")

    findings: List[DiagnosticFinding] = []
    geometry = report.geometry
    trainability = report.trainability

    if geometry is not None:
        if geometry.redundant_parameter_ratio >= redundancy_threshold:
            findings.append(
                DiagnosticFinding(
                    code="high_parameter_redundancy",
                    severity="warning",
                    message=(
                        f"Estimated redundant-direction ratio is "
                        f"{geometry.redundant_parameter_ratio:.1%}."
                    ),
                    suggestion=(
                        "Inspect the metric spectrum and test whether a smaller "
                        "ansatz preserves metric rank and task performance."
                    ),
                )
            )

        if geometry.condition_score <= condition_threshold:
            findings.append(
                DiagnosticFinding(
                    code="poor_metric_conditioning",
                    severity="warning",
                    message=(
                        f"Inverse metric condition score is "
                        f"{geometry.condition_score:.3e}."
                    ),
                    suggestion=(
                        "Inspect near-null metric directions, parameterization, "
                        "and initialization before increasing circuit depth."
                    ),
                )
            )

        n_params: Optional[int] = report.metadata.get("n_params")
        if n_params and n_params > 0:
            rank_fraction = geometry.metric_rank / n_params
            if rank_fraction <= rank_fraction_threshold:
                findings.append(
                    DiagnosticFinding(
                        code="low_metric_rank_fraction",
                        severity="warning",
                        message=f"Average metric rank uses only {rank_fraction:.1%} of parameter dimensions.",
                        suggestion="Investigate locally redundant parameters and repeated gate generators.",
                    )
                )

    if trainability is not None and trainability.near_zero_fraction >= near_zero_threshold:
        findings.append(
            DiagnosticFinding(
                code="many_near_zero_gradients",
                severity="warning",
                message=f"Near-zero gradients account for {trainability.near_zero_fraction:.1%} of sampled entries.",
                suggestion=(
                    "Run a system-size gradient-variance scan before attributing "
                    "the behavior to a barren plateau."
                ),
            )
        )

    if not findings:
        findings.append(
            DiagnosticFinding(
                code="no_threshold_warning",
                severity="info",
                message="No configured diagnostic threshold was crossed.",
                suggestion="Treat this as a screening result and validate on the target task and noise model.",
            )
        )

    return findings
