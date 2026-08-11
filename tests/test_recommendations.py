from pqc_analysis.core.result import AnalysisReport, GeometryResult, TrainabilityResult
from pqc_analysis.diagnostics import diagnose


def test_diagnose_flags_redundancy_conditioning_and_gradients():
    report = AnalysisReport(
        geometry=GeometryResult(
            metric_rank=2.0,
            redundant_parameter_ratio=0.5,
            condition_score=1e-5,
            effective_dimension=1.5,
            mean_log_volume=-3.0,
        ),
        trainability=TrainabilityResult(
            mean_abs_gradient=1e-5,
            gradient_variance=1e-8,
            gradient_norm=1e-4,
            near_zero_fraction=0.8,
        ),
        metadata={"n_params": 4},
    )

    codes = {finding.code for finding in diagnose(report)}
    assert "high_parameter_redundancy" in codes
    assert "poor_metric_conditioning" in codes
    assert "low_metric_rank_fraction" in codes
    assert "many_near_zero_gradients" in codes


def test_diagnose_returns_info_when_no_threshold_is_crossed():
    report = AnalysisReport(
        geometry=GeometryResult(
            metric_rank=4.0,
            redundant_parameter_ratio=0.0,
            condition_score=0.5,
            effective_dimension=3.0,
            mean_log_volume=-1.0,
        ),
        metadata={"n_params": 4},
    )

    findings = diagnose(report)
    assert len(findings) == 1
    assert findings[0].severity == "info"
