import numpy as np
import pennylane as qml

from pqc_analysis.experimental import analyze_topological_state_space


def test_topological_state_space_produces_persistence_metrics():
    def circuit(theta):
        qml.RY(theta[0], wires=0)
        qml.RZ(theta[1], wires=0)
        return qml.state()

    result = analyze_topological_state_space(
        circuit,
        n_qubits=1,
        n_params=2,
        samples=10,
        max_dim=1,
        seed=3,
    )

    assert result.distance_matrix.shape == (10, 10)
    assert np.allclose(result.distance_matrix, result.distance_matrix.T)
    assert np.allclose(np.diag(result.distance_matrix), 0.0)
    assert "h0_persistence_entropy" in result.metrics
    assert "h1_feature_count" in result.metrics
    assert result.metadata["experimental"] is True
