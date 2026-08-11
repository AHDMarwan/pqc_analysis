import pennylane as qml

import pqc_analysis as pqa


N_QUBITS = 4
N_PARAMS = 2 * N_QUBITS


def state_circuit(theta):
    for wire in range(N_QUBITS):
        qml.RY(theta[wire], wires=wire)
        qml.RZ(theta[N_QUBITS + wire], wires=wire)
    for wire in range(N_QUBITS - 1):
        qml.CNOT(wires=[wire, wire + 1])
    return qml.state()


def main():
    report = pqa.analyze(
        state_circuit,
        n_qubits=N_QUBITS,
        n_params=N_PARAMS,
        samples=25,
        metric_approximation="block-diag",
        seed=7,
    )
    print(report.summary())

    print("\nDiagnostics")
    for finding in pqa.diagnose(report):
        print(f"[{finding.severity}] {finding.code}: {finding.message}")
        print(f"  suggestion: {finding.suggestion}")


if __name__ == "__main__":
    main()
