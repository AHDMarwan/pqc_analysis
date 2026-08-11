import numpy as np
import pennylane as qml

import pqc_analysis as pqa


n_qubits = 3
n_params = 6


def state_circuit(theta):
    for wire in range(n_qubits):
        qml.RY(theta[wire], wires=wire)
        qml.RZ(theta[n_qubits + wire], wires=wire)
    for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
    return qml.state()


def cost_circuit(theta):
    for wire in range(n_qubits):
        qml.RY(theta[wire], wires=wire)
        qml.RZ(theta[n_qubits + wire], wires=wire)
    for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
    return qml.expval(qml.PauliZ(0))


gradient_fn = pqa.make_pennylane_gradient_fn(cost_circuit, n_qubits)
rng = np.random.default_rng(7)
theta_samples = rng.uniform(-np.pi, np.pi, size=(30, n_params))

report = pqa.analyze(
    state_circuit,
    n_qubits=n_qubits,
    n_params=n_params,
    parameter_samples=theta_samples,
    gradient_fn=gradient_fn,
)
print(report.summary())

profile = pqa.gradient_profile(
    gradient_fn,
    theta_samples,
    layer_groups={"ry": [0, 1, 2], "rz": [3, 4, 5]},
)
print("weakest parameters:", profile.weakest_parameters(k=3))
print("layer statistics:", profile.layer_statistics)

# Build a metric at one reference parameter point and inspect local redundancy.
dev = qml.device("default.qubit", wires=n_qubits)
qnode = qml.QNode(state_circuit, dev, interface="autograd")
theta = qml.numpy.array(theta_samples[0], requires_grad=True)
metric = pqa.compute_metric_tensor(qnode, theta, approximation="block-diag")
plan = pqa.geometry_pruning_plan(metric)
print("pruning candidates:", plan.candidate_indices)
print("redundancy scores:", plan.redundancy_scores)

cost = pqa.estimate_training_resources(
    n_params=n_params,
    steps=100,
    gradient_method="parameter-shift",
    shots_per_circuit=1000,
)
print("shots per step:", cost.shots_per_step)
print("total shots:", cost.total_shots)
