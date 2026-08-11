import numpy as np

from pqc_analysis.experimental.matrix_runner import MatrixExperimentResult
from pqc_analysis.experimental.topology_correlations import TopologyCorrelationResult


def test_correlation_records_exports_cross_group_pairs_only():
    result = TopologyCorrelationResult(
        metric_names=("h1_total_persistence", "gradient_variance", "metric_rank"),
        correlation_matrix=np.array(
            [
                [1.0, 0.5, -0.25],
                [0.5, 1.0, 0.1],
                [-0.25, 0.1, 1.0],
            ]
        ),
        p_values=np.array(
            [
                [0.0, 0.02, 0.2],
                [0.02, 0.0, 0.7],
                [0.2, 0.7, 0.0],
            ]
        ),
        sample_size=12,
        metadata={
            "method": "spearman",
            "topology_metrics": ("h1_total_persistence",),
            "diagnostic_metrics": ("gradient_variance", "metric_rank"),
        },
    )

    experiment = MatrixExperimentResult(cases=(), study=None, correlations=result)
    rows = experiment.correlation_records()

    assert len(rows) == 2
    assert rows[0] == {
        "topology_metric": "h1_total_persistence",
        "diagnostic_metric": "gradient_variance",
        "correlation": 0.5,
        "p_value": 0.02,
        "n": 12,
        "method": "spearman",
    }
    assert rows[1]["diagnostic_metric"] == "metric_rank"
    assert rows[1]["correlation"] == -0.25
