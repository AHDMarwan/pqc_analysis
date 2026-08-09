# PRA numerical-review checklist

This checklist is intentionally adversarial: it lists questions a skeptical referee could reasonably ask and where the suite addresses them.

- **Can the published table be independently regenerated?** `reproduce` profile + strict rounded-mean check.
- **Are tangent directions pseudoreplicates?** No. Fixed circuits are the bootstrap unit; tangent directions estimate the within-circuit local ensemble.
- **Are results driven by one ansatz?** `pra_core` includes multiple entangler/connectivity families and separate per-family reporting.
- **Is the symmetry anomaly actually a topology artifact?** U(1) line and ring controls are compared.
- **Is `d=6n` cherry-picked?** `pra_depth` sweeps six depth ratios.
- **Are `n=6,8,10` too small?** Core extension adds `n=12`; the manuscript should still avoid unsupported asymptotic exponents.
- **Is the effect measurement-basis specific?** Computational, X and local-Haar product bases are tested.
- **Does ideal noiseless readout hide fragility?** Exact classical bit-flip channels at 1% and 5% are tested.
- **Does initialization matter?** Three parameter distributions are fixed in advance.
- **Is 48 tangents enough to infer a spectrum?** No such claim is made. `pra_spectrum` increases M to 1024 and uses hold-out recovery.
- **Is a large in-sample Ky-Fan value just overfitting?** Split-sample leading-eigenspace recovery is reported.
- **Does the rank law require global isotropy?** A deliberately anisotropic control averages over random relative readout orientation.
- **Are numerical derivatives trustworthy?** Production tangents are analytic; state normalization, horizontality and Fisher contraction are hard invariants. Legacy finite-difference checks are preserved in provenance notebooks.
- **Were seeds/results selected after inspection?** The master seed, task grids and seed derivation are frozen and committed.
- **Can negative results disappear from the pipeline?** No outcome-based scientific threshold gates CI; raw circuit rows and manifests are artifacts.
- **Can a reader identify the exact environment?** Dependencies are pinned and each shard writes environment metadata.
- **Can a reader audit generated outputs?** Final output contains SHA-256 manifests, seed ledgers and open CSV tables.
- **Does data availability satisfy Physical Review expectations?** Code is public and executable; before submission the exact release and final data should be archived under a persistent DOI.
