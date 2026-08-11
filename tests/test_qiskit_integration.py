import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
pytest.importorskip("qiskit_algorithms")

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

import pqc_analysis as pqa


def test_qiskit_gradient_adapter_returns_finite_gradient():
    params = ParameterVector("theta", 2)
    circuit = QuantumCircuit(2)
    circuit.ry(params[0], 0)
    circuit.rx(params[1], 1)
    circuit.cx(0, 1)

    observable = SparsePauliOp.from_list([("ZI", 1.0)])
    gradient_fn = pqa.make_qiskit_gradient_fn(circuit, observable)
    gradient = gradient_fn(np.array([0.2, -0.4]))

    assert gradient.shape == (2,)
    assert np.all(np.isfinite(gradient))


def test_qiskit_barren_plateau_scan_runs_two_sizes():
    def problem_factory(n_qubits):
        params = ParameterVector("theta", n_qubits)
        circuit = QuantumCircuit(n_qubits)
        for wire in range(n_qubits):
            circuit.ry(params[wire], wire)
        for wire in range(n_qubits - 1):
            circuit.cx(wire, wire + 1)

        pauli = "Z" + "I" * (n_qubits - 1)
        observable = SparsePauliOp.from_list([(pauli, 1.0)])
        return circuit, observable

    result = pqa.qiskit_barren_plateau_scan(
        problem_factory,
        qubit_counts=[2, 3],
        n_params=lambda n: n,
        samples=6,
        seed=3,
    )

    assert result.gradient_variances.shape == (2,)
    assert np.all(np.isfinite(result.gradient_variances))
