# Falsification-oriented experimental plan

## Principle

The code is designed to distinguish three layers of claims:

1. **Exact identities/theorems** that should survive every valid numerical realization up to floating-point tolerance.
2. **Finite-sample numerical estimators** whose bias/variance must be characterized.
3. **Empirical statements about VQC ensembles** that may be true, false, architecture-dependent, basis-dependent, or finite-size effects.

Only layer 1 and software integrity can fail CI. Layers 2–3 are reported without outcome-based gating.

## A. Direct theory controls

### A1. Fisher contraction
For every regular analytic tangent used in an experiment verify `F_full <= F_Q` within a conservative numerical tolerance. Any violation aborts the shard and records the condition.

### A2. Trace-overlap identity
For a fixed empirical covariance `C_hat` and physical readout projector `P`, verify numerically that the mean direct retention equals `Tr(P C_hat)`. This is a construction identity, not a statistical hypothesis.

### A3. Isotropic projector law
Generate independent uniform unit vectors in score space. For ranks spanning small/intermediate fractions of the dimension, compare retention to the exact Beta distribution, including mean, variance and a KS diagnostic. The KS p-value is reported rather than used as a brittle CI threshold.

### A4. Relative-orientation rank law without isotropy
Generate a deliberately anisotropic tangent ensemble and Haar-random rank-r projectors. The average over readout orientation should approach `r/N` although the tangent covariance is nonflat. This directly targets the manuscript's distinction between global isotropy and relative-orientation typicality.

### A5. Ky-Fan recovery
For empirical `C_hat`, compare the physical readout retention with the leading-eigenvalue Ky-Fan sum. Because optimizing and evaluating on the same finite tangent sample is optimistic, also learn the leading eigenspace on a training split and report recovery on an independent tangent split.

## B. Exact manuscript reproduction

Frozen profile `reproduce`:

- `n = 6, 8, 10`
- `d = 6n`
- generic: RY-RZ/CZ-line, SU(2)/CNOT-line, SU(2)/Haar-U(4)-brickwork
- symmetry control: half-filled U(1) RZ-XY-line
- 6 fixed-circuit instances per generic family and size
- 12 fixed-circuit U(1) instances per size
- 48 normalized Gaussian parameter-space tangent directions per fixed circuit
- computational-basis measurement
- one- and two-body diagonal Pauli-Z readout spans
- support threshold `p_z > 1e-13`
- circuit-level nonparametric bootstrap

The pre-existing printed means are checked only here, after the simulation implementation is independently exercised by unit tests.

## C. Expanded empirical tests

### C1. Architecture and finite-size robustness (`pra_core`)
Use five generic families and two U(1) topologies, `n=6,8,10,12`, `d=6n`, 20 independent circuits per family/cell, `M=128` tangents.

Falsification targets:
- If generic low-weight `rho_k` varies strongly across families, replace a universal “generic” statement by architecture-specific claims.
- If U(1) enhancement disappears for the ring topology, interpret the original anomaly as topology-specific rather than symmetry-generic.
- No asymptotic exponent is fitted from three or four sizes unless diagnostics support the model and uncertainty is reported.

### C2. Depth dependence (`pra_depth`)
Sweep `d/n = 0.5,1,2,4,6,8` at `n=6,8,10`. This tests whether near-rank typicality is genuinely a deep-circuit phenomenon and whether the U(1) structure persists or crosses over with depth.

### C3. Spectrum/compressibility (`pra_spectrum`)
At `n=6,8`, depths `2n` and `6n`, use `M=1024` tangents for selected representative generic and U(1) families.

Report:
- empirical spectrum `lambda_j(C_hat)`;
- pairwise-purity `d_eff`;
- physical readout retention;
- in-sample Ky-Fan sum;
- split-sample Ky-Fan recovery;
- Haar-random projector mean and spread.

This avoids claiming “full spectral structure” from a 48-sample covariance estimator.

### C4. Measurement/initialization/noise robustness (`pra_robustness`)
At `n=8,10`, compare:
- parameter initialization: uniform `[-pi,pi]`, normal `N(0,1)`, narrow normal `N(0,0.1^2)`;
- measurement basis: computational, global X product basis, independently sampled local-Haar product basis;
- classical bit-flip readout noise: 0, 1%, 5%.

Interpretation is conditional. If the effect is confined to the computational basis, the paper should say so; that does not invalidate the relative-geometry theory.

### C5. Tangent-sample convergence (`pra_convergence`)
Use nested prefixes `M=32,64,128,256` on exactly the same circuit and direction stream. This isolates estimator convergence from between-circuit variability.

## D. Statistical protocol

- Independent unit: fixed circuit instance.
- Tangent directions within one circuit estimate that circuit's local tangent ensemble and are not treated as independent circuits.
- Report means, medians, standard deviations, and circuit-bootstrap 95% intervals.
- Report raw per-circuit rows and seed ledger.
- For U(1)/generic ratios bootstrap both circuit groups independently.
- Report actual-retention ratios separately from normalized-enhancement ratios because support/rank baselines differ.
- No p-hacking: seeds, profile grids and stopping rules are fixed before looking at expanded outcomes.
- No deletion based on whether a result agrees with the theory.

## E. Decision rules for the manuscript

- **Exact mathematical identity fails:** debug implementation first; if independently verified as a real counterexample, revise the theorem.
- **Reproduction fails:** do not tune seeds; identify whether the legacy result depended on undocumented implementation details and disclose it.
- **Generic rank typicality fails in expanded families:** narrow the empirical claim.
- **U(1) enhancement fails under topology change:** narrow it to the tested architecture/topology.
- **Measurement-basis dependence is strong:** emphasize relative geometry and explicitly condition the numerical claim on the measurement.
- **Finite-size trends reverse:** remove asymptotic language.

A negative result is scientifically acceptable; an unreported negative result is not.
