"""Experimental topology-vs-trainability study.

Requires: pip install -e ".[tda]"
"""

import pennylane as qml

from pqc_analysis.adapters import make_pennylane_gradient_fn
from pqc_analysis.experimental import TopologyStudySpec, run_topology_diagnostic_study


def make_ansatz(depth):
    n_qubits = 3
    n_params = n_qubits * depth

    def state_circuit(theta):
        index = 0
        for _ in range(depth):
            for wire in range(n_qubits):
                qml.RY(theta[index], wires=wire)
                index += 1
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
        return qml.state()

    def cost_circuit(theta):
        index = 0
        for _ in range(depth):
            for wire in range(n_qubits):
                qml.RY(theta[index], wires=wire)
                index += 1
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
        return qml.expval(qml.PauliZ(0))

    gradient_fn = make_pennylane_gradient_fn(cost_circuit, n_qubits)
    return state_circuit, gradient_fn, n_qubits, n_params


specs = []
for depth in (1, 2, 3):
    circuit, gradient_fn, n_qubits, n_params = make_ansatz(depth)
    specs.append(
        TopologyStudySpec(
            name=f"local_chain_depth_{depth}",
            circuit=circuit,
            n_qubits=n_qubits,
            n_params=n_params,
            gradient_fn=gradient_fn,
            metadata={"depth": depth},
        )
    )

study = run_topology_diagnostic_study(
    specs,
    seeds=range(5),
    geometry_samples=20,
    topology_samples=40,
    topology_max_dim=1,
)

association = study.correlate(
    topology_metrics=["h1_persistence_entropy", "h1_total_persistence"],
    diagnostic_metrics=["gradient_variance", "effective_dimension"],
    method="spearman",
    permutations=1000,
    seed=11,
)

print("records:", len(study.records))
print(
    "H1 entropy vs gradient variance:",
    association.pair("h1_persistence_entropy", "gradient_variance"),
)
