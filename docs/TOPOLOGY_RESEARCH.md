# Experimental topology research program

This module tests a specific research hypothesis:

> Do topological signatures of the PQC-accessible pure-state manifold carry reproducible information about trainability and geometric redundancy?

The topology layer is deliberately marked **experimental**. Persistent-homology summaries are not presented as established trainability diagnostics.

## State-space construction

For a parameterized circuit

`|psi(theta)> = U(theta)|0>`

we sample parameter vectors and evaluate pure states. Pairwise distances use the pure-state Bures distance

`d_B(psi, phi) = sqrt(2 * (1 - |<psi|phi>|))`.

For pure states this is equivalent to the density-matrix Bures construction but avoids explicit density matrices and matrix square roots.

The resulting precomputed distance matrix is passed to Ripser to obtain persistent-homology diagrams.

## Topological observables

For each homology dimension H0, H1, ... the experimental API records:

- persistence entropy;
- total persistence;
- maximum persistence;
- finite feature count;
- mean persistence.

Infinite-death features are excluded from lifetime-based summaries and remain available in the raw diagrams.

## Matched diagnostic study

`run_topology_diagnostic_study(...)` evaluates the same architecture/seed protocol across:

- Fubini-Study metric rank;
- redundant-parameter ratio;
- metric conditioning;
- effective dimension;
- gradient variance;
- mean absolute gradient;
- gradient norm;
- near-zero gradient fraction;
- persistent-homology summaries.

Topology and geometry samples use the same initialization law and deterministic seed ledger, but independent parameter draws. This prevents an accidental correlation caused by evaluating exactly the same sampled points while preserving reproducibility.

## Association analysis

`TopologyStudyResult.correlate(...)` supports Pearson or Spearman association and computes empirical two-sided permutation p-values.

A correlation is only a hypothesis-generating result. It must not be described as predictive without additional validation.

## Minimum standard for a predictive claim

A topology metric should only be called predictive of trainability if it satisfies all of the following:

1. The relationship is replicated across multiple circuit families, qubit counts, depths, and random seeds.
2. The prediction is evaluated out-of-sample, e.g. architecture-held-out or size-held-out cross-validation.
3. Baselines using simpler quantities such as depth, parameter count, metric rank, or entangling-gate count are included.
4. Uncertainty is reported with bootstrap confidence intervals or an equivalent resampling protocol.
5. Multiple-hypothesis correction is used when many topology/diagnostic pairs are screened.
6. Results are robust to the parameter initialization distribution and topology sample count.
7. The effect survives reasonable changes in persistence filtering and homology dimension.

## Implemented benchmark matrix

The repository now includes a factorial benchmark generator through `BenchmarkMatrixConfig` and `build_benchmark_matrix(...)`.

Default exploratory matrix:

- ansatz families: `hardware_efficient`, `alternating`, `tree`;
- qubits: 2, 4, 6;
- depths: 1, 2, 4;
- costs: local Z expectation and global Pauli-Z string;
- total configurations before seeds: 54.

The smaller pilot profile uses:

- ansatz families: hardware-efficient and tree;
- qubits: 2 and 4;
- depths: 1 and 2;
- local/global costs;
- 16 configurations before seeds.

Run programmatically with:

```python
from pqc_analysis.experimental import (
    BenchmarkMatrixConfig,
    run_benchmark_matrix_experiment,
)

result = run_benchmark_matrix_experiment(
    BenchmarkMatrixConfig(),
    seeds=range(20),
    geometry_samples=50,
    topology_samples=100,
    topology_max_dim=1,
    permutations=5000,
)

records = result.records()
correlations = result.correlation_records()
```

A command-line runner is available at `scripts/run_topology_matrix.py` and writes:

- `records.csv` with one matched geometry/trainability/topology row per architecture/seed;
- `correlations.csv` with topology-diagnostic association estimates and permutation p-values;
- `metadata.json` with the experiment protocol.

GitHub Actions also exposes the manual `Topology Benchmark Matrix` workflow with `pilot` and `full` profiles. The full profile is intentionally manual because it is substantially more expensive than CI-scale validation.

## Primary hypotheses

The first screen tests:

- H1 persistence entropy vs. gradient variance;
- H1 total persistence vs. Fubini-Study effective dimension;
- H0/H1 summaries vs. metric rank and redundancy;
- whether topology adds predictive value after controlling for depth, parameter count, and metric-spectrum features.

## Next statistical phase

Raw correlations are not the endpoint. Once the matrix produces stable data, the next analysis should add:

- architecture-held-out and qubit-size-held-out validation;
- bootstrap confidence intervals;
- false-discovery-rate correction across screened topology/diagnostic pairs;
- regression baselines using depth, n_params, qubit count, cost locality, and geometric metrics;
- incremental predictive-value tests comparing baseline models against baseline + topology;
- sensitivity analysis over topology sample counts and initialization laws.

The final question is not whether a correlation exists in one benchmark, but whether topology provides incremental, reproducible information beyond cheaper geometric and structural diagnostics.
