import pennylane as qml

import pqc_analysis as pqa


def test_benchmark_records_resource_costs():
    def circuit(theta):
        qml.RY(theta[0], wires=0)
        return qml.state()

    spec = pqa.PQCSpec(
        name="single_ry_costed",
        circuit=circuit,
        n_qubits=1,
        n_params=1,
        gradient_method="parameter-shift",
        shots_per_circuit=1000,
    )
    result = pqa.benchmark([spec], seeds=[7], samples=3)
    record = result.to_records()[0]

    assert record["circuit_evaluations_per_step"] == 2
    assert record["shots_per_step"] == 2000
    assert result.aggregate()["single_ry_costed"]["shots_per_step_mean"] == 2000.0
