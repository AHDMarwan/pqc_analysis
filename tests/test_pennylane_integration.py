import numpy as np
import pennylane as qml

import pqc_analysis as pqa


def test_pennylane_gradient_adapter_returns_expected_shape():
    n_qubits = 2

    def cost(theta):
        qml.RY(theta[0], wires=0)
        qml.RX(theta[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    gradient_fn = pqa.make_pennylane_gradient_fn(cost, n_qubits)
    gradient = gradient_fn(np.array([0.2, -0.4]))

    assert gradient.shape == (2,)
    assert np.all(np.isfinite(gradient))
    assert np.allclose(gradient[0], -np.sin(0.2), atol=1e-6)


def test_analyze_runs_on_small_pennylane_state_circuit():
    def circuit(theta):
        qml.RY(theta[0], wires=0)
        qml.RX(theta[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    samples = np.array(
        [
            [0.2, 0.3],
            [0.5, -0.7],
            [-0.4, 0.9],
        ]
    )

    report = pqa.analyze(
        circuit,
        n_qubits=2,
        n_params=2,
        parameter_samples=samples,
        metric_approximation="block-diag",
    )

    assert report.geometry is not None
    assert 0.0 <= report.geometry.redundant_parameter_ratio <= 1.0
    assert 0.0 <= report.geometry.condition_score <= 1.0
    assert report.geometry.metric_rank >= 0.0


def test_pennylane_barren_plateau_scan_returns_scaling_result():
    def circuit_factory(n_qubits):
        def cost(theta):
            for wire in range(n_qubits):
                qml.RY(theta[wire], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            return qml.expval(qml.PauliZ(0))

        return cost

    result = pqa.pennylane_barren_plateau_scan(
        circuit_factory,
        qubit_counts=[2, 3],
        n_params=lambda n: n,
        samples=8,
        seed=5,
    )

    assert result.gradient_variances.shape == (2,)
    assert np.all(np.isfinite(result.gradient_variances))
    assert np.isfinite(result.log_variance_slope)
