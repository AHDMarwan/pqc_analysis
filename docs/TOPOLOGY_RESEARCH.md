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

## First benchmark matrix

Recommended first experiment:

- architectures: hardware-efficient, strongly-entangling, tree/QCNN-like, layered local ansatz, and a symmetry-preserving ansatz;
- qubits: 4, 6, 8, 10;
- depths: shallow, medium, deep relative to system size;
- seeds: at least 20 per configuration;
- topology samples: start at 100 and run a convergence check at 50/100/200;
- homology: H0 and H1 initially;
- costs: one local observable and one global observable;
- gradients: report raw gradient variance and barren-plateau scaling separately.

Primary hypotheses to test:

- H1 persistence entropy vs. gradient variance;
- H1 total persistence vs. Fubini-Study effective dimension;
- H0/H1 summaries vs. metric rank and redundancy;
- whether topology adds predictive value after controlling for depth, parameter count, and metric-spectrum features.

The final question is not whether a correlation exists in one benchmark, but whether topology provides incremental, reproducible information beyond cheaper geometric and structural diagnostics.
