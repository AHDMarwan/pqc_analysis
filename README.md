# PQC Analysis

**PQC Analysis** is a research-oriented Python toolkit for diagnosing parameterized quantum circuits (PQCs) used in variational quantum algorithms and quantum machine learning.

The project focuses on a question that circuit libraries do not answer directly: **is this PQC geometrically well-conditioned and practically trainable?**

## Current scope (v0.2)

PQC Analysis provides:

- **Quantum geometry** using the Fubini–Study metric tensor
- **Metric spectrum diagnostics**: rank, conditioning, effective dimension, and parameter redundancy
- **Gradient trainability statistics**: mean absolute gradient, variance, norm, and near-zero fraction
- **Barren-plateau scaling scans** based on gradient-variance scaling with system size
- **Transparent diagnostic recommendations** with configurable thresholds
- **Topological data analysis (TDA)** via persistent homology and Bures distances
- A unified `analyze(...)` API for PennyLane-compatible circuits
- Backend-independent trainability statistics
- PennyLane and optional Qiskit gradient adapters

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
python -m pip install -e ".[dev]"
python -m pytest -q tests
```

For Qiskit support:

```bash
python -m pip install -e ".[qiskit]"
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

Structured fields are available directly:

```python
report.geometry.metric_rank
report.geometry.redundant_parameter_ratio
report.geometry.condition_score
report.geometry.effective_dimension
report.geometry.metadata
```

## 2. Diagnostic recommendations

Recommendations are explicit heuristics with configurable thresholds, not hidden model-generated judgments.

```python
for finding in pqa.diagnose(report):
    print(finding.severity, finding.code)
    print(finding.message)
    print(finding.suggestion)
```

Current diagnostics screen for high parameter redundancy, poor metric conditioning, low metric-rank fraction, and a large fraction of near-zero gradients. A single-circuit gradient diagnostic is never labelled a barren plateau; use a scaling scan for that question.

## 3. Gradient statistics

The statistical layer is backend-independent. Supply any callable that maps a parameter vector to a gradient vector:

```python
stats = pqa.gradient_statistics(
    gradient_fn=my_gradient_function,
    parameter_samples=theta_samples,
)

print(stats.gradient_variance)
print(stats.near_zero_fraction)
```

For scalar PennyLane cost circuits:

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

## 4. Barren-plateau scaling diagnostics

A barren plateau is fundamentally a scaling phenomenon. Rather than classifying one small gradient as a barren plateau, PQC Analysis scans several system sizes and fits

```text
log(Var[gradient]) = a * n_qubits + b
```

A negative slope with a strong linear fit is evidence **consistent with exponential gradient suppression**. The diagnostic deliberately does not claim that this numerical fit alone proves a barren plateau.

```python
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

`scan.shows_exponential_suppression` is a configurable heuristic based on the fitted decay rate and `R^2`. Report it together with the raw scaling data.

## 5. Qiskit adapter

Qiskit support is optional and reuses the same backend-independent trainability layer. The adapter uses Qiskit Algorithms' estimator-gradient interface.

```python
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import pqc_analysis as pqa

params = ParameterVector("theta", 4)
qc = QuantumCircuit(4)
for i in range(4):
    qc.ry(params[i], i)
for i in range(3):
    qc.cx(i, i + 1)

observable = SparsePauliOp.from_list([("ZIII", 1.0)])
gradient_fn = pqa.make_qiskit_gradient_fn(qc, observable)
```

A Qiskit scaling scan is also available through `pqa.qiskit_barren_plateau_scan(...)`. A custom EstimatorV2-compatible backend can be supplied when moving beyond local statevector analysis.

## 6. Geometry primitives

The lower-level API is public:

```python
pqa.compute_metric_tensor(qnode, theta, approximation="block-diag")
pqa.metric_spectrum(metric)
pqa.metric_rank(metric)
pqa.condition_score(metric)
pqa.effective_dimension(metric)
pqa.redundant_parameter_ratio(metric)
```

`compute_metric_tensor` supports `"full"`, `"block-diag"`, and `"diag"` approximations. `analyze(...)` allocates an auxiliary device wire when the full metric requires it.

## 7. Topological analysis

The original TDA functionality remains available:

```python
from pqc_analysis import pqc_topology_analysis
```

It uses density matrices, pairwise Bures distances, persistent homology (`ripser`), and persistence entropy to characterize sampled PQC state spaces.

## Examples

Runnable examples are provided in:

```text
examples/basic_analysis.py
examples/barren_plateau_scan.py
```

## Project direction

Next milestones:

1. standardized architecture benchmarking and experiment tables
2. richer per-parameter gradient spectra and layer-wise diagnostics
3. geometry-aware parameter pruning experiments
4. shot-aware and hardware-aware resource diagnostics
5. reproducible benchmark suites across PennyLane and Qiskit

The goal is for PQC Analysis to become an **analysis layer above quantum programming frameworks**, rather than another circuit-construction framework.

## Scientific use

For research results, always report parameter initialization, sample count, observable, circuit family, system sizes, metric approximation, differentiation method, random seed, and heuristic thresholds. Trainability conclusions are architecture- and problem-dependent.

## License

MIT
