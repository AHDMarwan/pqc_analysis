import pennylane as qml

import pqc_analysis as pqa


def circuit_factory(n_qubits):
    def cost(theta):
        for wire in range(n_qubits):
            qml.RY(theta[wire], wires=wire)
        for wire in range(n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        return qml.expval(qml.PauliZ(0))

    return cost


def main():
    result = pqa.pennylane_barren_plateau_scan(
        circuit_factory,
        qubit_counts=[2, 4, 6, 8],
        n_params=lambda n: n,
        samples=50,
        seed=11,
    )

    print(result.summary())
    print("qubits:", result.qubit_counts)
    print("gradient variances:", result.gradient_variances)


if __name__ == "__main__":
    main()
