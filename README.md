# PQC Analysis

**PQC Analysis** is a research-oriented Python toolkit for diagnosing parameterized quantum circuits (PQCs) used in variational quantum algorithms and quantum machine learning.

The project focuses on a question that circuit libraries do not answer directly: **is this PQC geometrically well-conditioned and practically trainable?**

## Current scope (v0.2)

PQC Analysis provides:

- **Quantum geometry** using the Fubini–Study metric tensor
- **Metric spectrum diagnostics**: rank, conditioning, effective dimension, and parameter redundancy
- **Gradient trainability statistics**: mean absolute gradient, variance, norm, and near-zero fraction
- **Barren-plateau scaling scans** based on the scaling of gradient variance with system size
- **Topological data analysis (TDA)** via persistent homology and Bures distances
- A unified `analyze(...)` API for PennyLane-compatible circuits
- Backend-independent trainability statistics, with a PennyLane gradient adapter included

The v0.1 functions remain available for backward compatibility.

## Installation

PennyLane currently requires Python 3.11 or newer, so PQC Analysis v0.2 uses the same minimum Python version.

```bash
python -m pip install git+https://github.com/AHDMarwan/pqc_analysis.git
```

For development:

```bash
git clone https://github.com/AHDMarwan/pqc_analysis.git
cd pqc_analysis
python -m pip install -e .
python -m pip install pytest
python -m pytest -q tests
```

## 1. Unified geometry analysis

A circuit passed to `analyze` should be compatible with a PennyLane QNode and accept a single parameter vector.

```python
import pennylane as qml
import pqc_analysis as pqa

n_qubits = 4
n_params = 8


def circuit(theta):
    for wire in range(n_qubits):
        qml.RY(theta[wire], wires=wire)
        qml.RZ(theta[n_qubits + wire], wires=wire)
    for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
    return qml.state()


report = pqa.analyze(
    circuit,
    n_qubits=n_qubits,
    n_params=n_params,
    samples=50,
    metric_approximation="block-diag",
)

print(report.summary())
```

The structured result can also be inspected programmatically:

```python
report.geometry.metric_rank
report.geometry.redundant_parameter_ratio
report.geometry.condition_score
report.geometry.effective_dimension
report.geometry.metadata
```

## 2. Gradient statistics

The statistical layer is backend-independent. Supply any callable that maps a parameter vector to a gradient vector:

```python
stats = pqa.gradient_statistics(
    gradient_fn=my_gradient_function,
    parameter_samples=theta_samples,
)

print(stats.gradient_variance)
print(stats.near_zero_fraction)
```

For scalar PennyLane cost circuits, a gradient adapter is provided:

```python
import pennylane as qml
import pqc_analysis as pqa

n_qubits = 4


def cost_circuit(theta):
    for wire in range(n_qubits):
        qml.RY(theta[wire], wires=wire)
    for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
    return qml.expval(qml.PauliZ(0))


gradient_fn = pqa.make_pennylane_gradient_fn(cost_circuit, n_qubits)
```

## 3. Barren-plateau scaling diagnostics

A barren plateau is fundamentally a scaling phenomenon. Rather than classifying a single small gradient as a barren plateau, PQC Analysis scans several system sizes and fits

```text
log(Var[gradient]) = a * n_qubits + b
```

A negative slope with a strong linear fit is evidence **consistent with exponential gradient suppression**. The diagnostic deliberately does not claim that this numerical fit alone proves a barren plateau.

```python
import pennylane as qml
import pqc_analysis as pqa


def circuit_factory(n_qubits):
    def cost(theta):
        index = 0
        for wire in range(n_qubits):
            qml.RY(theta[index], wires=wire)
            index += 1
        for wire in range(n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        return qml.expval(qml.PauliZ(0))
    return cost


scan = pqa.pennylane_barren_plateau_scan(
    circuit_factory,
    qubit_counts=[2, 4, 6, 8],
    n_params=lambda n: n,
    samples=100,
)

print(scan.summary())
print(scan.gradient_variances)
print(scan.log_variance_slope)
print(scan.r_squared)
```

The boolean property

```python
scan.shows_exponential_suppression
```

is a configurable heuristic based on the fitted decay rate and `R^2`; it should be reported together with the raw scaling data, not used as a standalone scientific conclusion.

## 4. Geometry primitives

The lower-level API is also public:

```python
pqa.compute_metric_tensor(qnode, theta, approximation="block-diag")
pqa.metric_spectrum(metric)
pqa.metric_rank(metric)
pqa.condition_score(metric)
pqa.effective_dimension(metric)
pqa.redundant_parameter_ratio(metric)
```

`compute_metric_tensor` supports `"full"`, `"block-diag"`, and `"diag"` approximations.

## 5. Topological analysis

The original TDA functionality remains available:

```python
from pqc_analysis import pqc_topology_analysis
```

It uses density matrices, pairwise Bures distances, persistent homology (`ripser`), and persistence entropy to characterize the topology of sampled PQC state spaces.

## Project direction

The next development milestones are:

1. richer trainability diagnostics and per-parameter gradient spectra
2. automated geometry/trainability recommendations
3. standardized architecture benchmarking
4. Qiskit adapters without changing the statistical core
5. shot-aware and hardware-aware diagnostics

The goal is for PQC Analysis to become an **analysis layer above quantum programming frameworks**, rather than another circuit-construction framework.

## Scientific use

For research results, always report numerical settings such as parameter initialization, number of samples, observable, circuit family, system sizes, metric approximation, differentiation method, random seed, and any heuristic thresholds. Trainability conclusions are architecture- and problem-dependent.

## License

MIT
