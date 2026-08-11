import numpy as np

from pqc_analysis.experimental import (
    correlate_topology_with_diagnostics,
    pure_state_bures_distance,
    summarize_persistence_diagram,
)


def test_pure_state_bures_distance_identity_and_orthogonal():
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    assert pure_state_bures_distance(zero, zero) == 0.0
    assert np.isclose(pure_state_bures_distance(zero, one), np.sqrt(2.0))


def test_persistence_summary_uses_finite_positive_lifetimes_only():
    diagram = np.array([[0.0, 1.0], [0.2, 0.7], [0.0, np.inf], [0.5, 0.4]])
    summary = summarize_persistence_diagram(diagram)
    assert summary.feature_count == 2
    assert np.isclose(summary.total_persistence, 1.5)
    assert np.isclose(summary.max_persistence, 1.0)


def test_topology_correlation_detects_monotonic_association():
    records = []
    for i in range(8):
        records.append(
            {
                "h1_persistence_entropy": float(i),
                "gradient_variance": float(2 * i + 1),
                "metric_rank": float(8 - i),
            }
        )

    result = correlate_topology_with_diagnostics(
        records,
        topology_metrics=["h1_persistence_entropy"],
        diagnostic_metrics=["gradient_variance", "metric_rank"],
        method="spearman",
        permutations=100,
        seed=7,
    )

    positive = result.pair("h1_persistence_entropy", "gradient_variance")
    negative = result.pair("h1_persistence_entropy", "metric_rank")
    assert np.isclose(positive["correlation"], 1.0)
    assert np.isclose(negative["correlation"], -1.0)
    assert 0.0 < positive["p_value"] <= 1.0
