# Audit notes on the legacy notebooks

## 1. U(1) confirmatory scheduling inconsistency

The confirmatory notebook defines a target of 12 U(1) instances but its visible generic scheduling loop uses the common `INSTANCES=6` range. A literal clean execution of that schedule would therefore generate only six U(1) fixed circuits per size, while the manuscript explicitly reports twelve.

The clean reproduction implementation uses the manuscript's declared statistical protocol: six circuits for each generic family and twelve U(1) circuits per size. With this correction, a fresh run reproduces all printed Table-I means to the stated rounding. This is strong evidence that 12 was the intended U(1) sample count, but the discrepancy is preserved here rather than silently erased.

## 2. Independent reproduction result

The clean implementation recovered the following means:

| n | group | rho1 | rho2 | d_eff(pair) | Ffull/FQ |
|---:|---|---:|---:|---:|---:|
| 6 | generic | 0.958442 | 0.993350 | 46.1301 | 0.488544 |
| 6 | U(1) | 2.076641 | 1.186344 | 9.52400 | 0.474446 |
| 8 | generic | 0.931258 | 0.972936 | 136.070 | 0.489653 |
| 8 | U(1) | 4.224422 | 1.895219 | 16.9695 | 0.478514 |
| 10 | generic | 0.926089 | 0.964361 | 351.575 | 0.491892 |
| 10 | U(1) | 9.824011 | 3.607351 | 27.4862 | 0.469952 |

All means match the manuscript's rounded values. Bootstrap endpoints are close but need not be bit-identical because the bootstrap RNG stream is not a scientific observable; the expanded suite records the bootstrap seed deterministically.

At `n=10`, the independently reconstructed U(1)/generic **actual one-body retention ratio** is `38.9117`, with a deterministic 5000-resample circuit-bootstrap interval approximately `[37.26, 40.72]`, consistent with the manuscript's reported 38.9 `[37.3,40.7]`.

## 3. Pairwise effective dimension is a diagnostic

The code preserves the manuscript's unbiased U-statistic for `Tr(C^2)`. Its inverse is not unbiased, so `d_eff_pairwise` is not labeled as an unbiased estimator of the population effective dimension.

## 4. Spectrum rank limitation

With `M=48` tangent directions, the sample covariance has rank at most 48 regardless of the score-space dimension. Therefore the confirmatory experiment is adequate for the pairwise second-moment diagnostic but not for detailed reconstruction of a high-dimensional spectrum. `pra_spectrum` uses `M=1024` at smaller sizes and adds split-sample evaluation.

## 5. Test-discovered implementation correction

During the clean rewrite, the U(1)-line parameter count was initially coded as a constant number of XY gates per layer. Unit tests caught a parameter-cursor mismatch because alternating open-chain brickwork layers contain different numbers of bonds. The production implementation now counts the actual bonds layer by layer. This correction was made before expanded scientific runs and is documented as part of the audit trail.
