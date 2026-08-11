# PQC Analysis

**PQC Analysis** is a research-oriented Python toolkit for diagnosing parameterized quantum circuits (PQCs) used in variational quantum algorithms and quantum machine learning.

The project focuses on a question that circuit libraries do not answer directly: **is this PQC geometrically well-conditioned, practically trainable, and resource-efficient?**

## Current scope (v0.2)

PQC Analysis provides:

- **Quantum geometry** using the Fubini–Study metric tensor
- **Metric spectrum diagnostics**: rank, conditioning, effective dimension, and parameter redundancy
- **Gradient trainability statistics**: mean absolute gradient, variance, norm, and near-zero fraction
- **Per-parameter and layer-wise gradient profiles**
- **Barren-plateau scaling scans** based on gradient-variance scaling with system size
- **Geometry-aware pruning plans** derived from metric null spaces
- **Shot and circuit-evaluation accounting** for common gradient estimators
- **Standardized architecture benchmarking** across seeds with flat/CSV-friendly records
- **Transparent diagnostic recommendations** with configurable thresholds
- **Topological data analysis (TDA)** via persistent homology and Bures distances
- PennyLane and optional Qiskit gradient adapters

The v0.1 functions remain available for backward compatibility.

## Installation

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

Optional integrations:

```bash
python -m pip install -e ".[qiskit]"
python -m pip install -e ".[tda]"
python -m pip install -e ".[legacy]"
```

## Unified geometry analysis

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

Structured fields are available directly through `report.geometry`, `report.trainability`, and `report.metadata`.

## Diagnostic recommendations

Recommendations are explicit heuristics with configurable thresholds, not hidden model-generated judgments.

```python
for finding in pqa.diagnose(report):
    print(finding.severity, finding.code)
    print(finding.message)
    print(finding.suggestion)
```

A single small gradient is never labelled a barren plateau; use a scaling scan for that question.

## Per-parameter and layer-wise trainability

```python
profile = pqa.gradient_profile(
    gradient_fn,
    theta_samples,
    layer_groups={
        "encoding": [0, 1, 2, 3],
        "variational_1": [4, 5, 6, 7],
        "variational_2": [8, 9, 10, 11],
    },
)

print(profile.mean_abs)
print(profile.variance)
print(profile.near_zero_fraction)
print(profile.layer_statistics)
print(profile.weakest_parameters(k=5))
```

This makes it possible to distinguish a globally weak gradient signal from a localized trainability problem in one layer or parameter subset.

## Barren-plateau scaling diagnostics

A barren plateau is fundamentally a scaling phenomenon. PQC Analysis scans several system sizes and fits

```text
log(Var[gradient]) = a * n_qubits + b
```

A negative slope with a strong linear fit is evidence **consistent with exponential gradient suppression**; the fit alone is not presented as a proof of a barren plateau.

```python
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

## Geometry-aware parameter pruning

For a sampled Fubini–Study metric tensor, the local null space identifies parameter combinations that do not change the represented state to first order. PQC Analysis converts that null space into a conservative ranking of candidate parameters:

```python
plan = pqa.geometry_pruning_plan(metric)

print(plan.candidate_indices)
print(plan.redundancy_scores)
print(plan.estimated_rank)
```

For a more robust proposal across several parameter points:

```python
plan = pqa.aggregate_pruning_plan(metric_samples)
```

The pruning API intentionally **does not mutate the circuit automatically**. Removing a parameter changes the circuit family; candidates should be pruned experimentally and the geometry, trainability, and task metric should then be re-evaluated.

## Shot and circuit-evaluation accounting

```python
cost = pqa.estimate_training_resources(
    n_params=48,
    steps=200,
    gradient_method="parameter-shift",
    shots_per_circuit=1000,
    include_objective_evaluation=True,
)

print(cost.circuit_evaluations_per_step)
print(cost.shots_per_step)
print(cost.total_circuit_evaluations)
print(cost.total_shots)
```

Supported algorithmic estimators include parameter-shift, SPSA, forward/central finite differences, adjoint, and backpropagation. For adjoint/backpropagation, PQC Analysis deliberately leaves fixed hardware circuit/shot counts unspecified because those methods are simulator/backpropagation strategies rather than a universal hardware execution count.

## Standardized architecture benchmarking

```python
specs = [
    pqa.PQCSpec(
        name="ansatz_a",
        circuit=circuit_a,
        n_qubits=6,
        n_params=24,
        gradient_fn=gradient_a,
        gradient_method="parameter-shift",
        shots_per_circuit=1000,
    ),
    pqa.PQCSpec(
        name="ansatz_b",
        circuit=circuit_b,
        n_qubits=6,
        n_params=18,
        gradient_fn=gradient_b,
        gradient_method="spsa",
        shots_per_circuit=1000,
    ),
]

result = pqa.benchmark(specs, seeds=[0, 1, 2, 3, 4], samples=100)

records = result.to_records()
summary = result.aggregate()
```

Benchmark records can contain geometry, trainability, and resource-cost fields under the same reproducible protocol.

## Qiskit adapter

Qiskit support is optional and reuses the backend-independent trainability layer.

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

A Qiskit scaling scan is available through `pqa.qiskit_barren_plateau_scan(...)`.

## Geometry primitives

```python
pqa.compute_metric_tensor(qnode, theta, approximation="block-diag")
pqa.metric_spectrum(metric)
pqa.metric_rank(metric)
pqa.condition_score(metric)
pqa.effective_dimension(metric)
pqa.redundant_parameter_ratio(metric)
```

`compute_metric_tensor` supports `"full"`, `"block-diag"`, and `"diag"` approximations.

## Topological analysis

The original TDA functionality remains available through `pqa.pqc_topology_analysis(...)`. It uses density matrices, pairwise Bures distances, persistent homology (`ripser`), and persistence entropy.

## Examples

Runnable examples are provided under `examples/`, including basic analysis and barren-plateau scans.

## Research direction

The next research-oriented steps are validation studies rather than adding arbitrary metrics: benchmark known trainable/untrainable ansatz families, measure how well geometric redundancy scores predict safe pruning, compare gradient estimators under equal shot budgets, and test whether combined geometry/trainability diagnostics predict downstream optimization performance.

The goal is for PQC Analysis to become an **analysis layer above quantum programming frameworks**, rather than another circuit-construction framework.

## Scientific use

For research results, report parameter initialization, sample count, observable, circuit family, system sizes, metric approximation, differentiation method, random seed, shot budget, layer grouping, pruning tolerance, and heuristic thresholds. Trainability and redundancy conclusions are architecture- and problem-dependent.

## License

MIT
