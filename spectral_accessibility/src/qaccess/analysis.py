from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .statistics import aggregate_circuit_results, group_ratio_bootstrap, pooled_group_results


def _concat(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(root.rglob(filename))
    frames = [pd.read_csv(p) for p in paths if p.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _plot_core(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return
    base = df[(df.parameter_distribution == "uniform_pi") & (df.measurement_basis == "computational") & (df.bitflip_rate == 0)].copy()
    if base.empty:
        return
    for k in sorted(base.k.unique()):
        sub = base[base.k == k]
        plt.figure(figsize=(8, 5))
        for family, g in sub.groupby("family"):
            g2 = g.groupby("n", as_index=False).enhancement.mean()
            plt.plot(g2.n, g2.enhancement, marker="o", label=family)
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.yscale("log")
        plt.xlabel("qubits n")
        plt.ylabel(r"$\rho_k$ = observed / rank baseline")
        plt.title(fr"Low-weight accessibility enhancement, $k={k}$")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(outdir / f"enhancement_vs_n_k{k}.pdf")
        plt.savefig(outdir / f"enhancement_vs_n_k{k}.png", dpi=220)
        plt.close()

    one = base[base.k == 1]
    if not one.empty:
        plt.figure(figsize=(8, 5))
        for family, g in one.groupby("family"):
            g2 = g.groupby("n", as_index=False).deff_pairwise.mean()
            plt.plot(g2.n, g2.deff_pairwise, marker="o", label=family)
        plt.yscale("log")
        plt.xlabel("qubits n")
        plt.ylabel(r"pairwise $d_{\rm eff}$")
        plt.title("Tangent effective-dimension diagnostic")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(outdir / "deff_vs_n.pdf")
        plt.savefig(outdir / "deff_vs_n.png", dpi=220)
        plt.close()

        plt.figure(figsize=(8, 5))
        for family, g in one.groupby("family"):
            g2 = g.groupby("n", as_index=False).Ffull_over_FQ_mean.mean()
            plt.plot(g2.n, g2.Ffull_over_FQ_mean, marker="o", label=family)
        plt.xlabel("qubits n")
        plt.ylabel(r"$F_{\rm full}/F_Q$")
        plt.title("Complete-record Fisher retention")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(outdir / "ffull_over_fq_vs_n.pdf")
        plt.savefig(outdir / "ffull_over_fq_vs_n.png", dpi=220)
        plt.close()


def _plot_depth(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty or df.depth_factor.nunique() < 2:
        return
    base = df[(df.parameter_distribution == "uniform_pi") & (df.measurement_basis == "computational") & (df.bitflip_rate == 0) & (df.k == 1)]
    if base.empty:
        return
    for n in sorted(base.n.unique()):
        plt.figure(figsize=(8, 5))
        for family, g in base[base.n == n].groupby("family"):
            g2 = g.groupby("depth_factor", as_index=False).enhancement.mean()
            plt.plot(g2.depth_factor, g2.enhancement, marker="o", label=family)
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.yscale("log")
        plt.xlabel(r"depth ratio $d/n$")
        plt.ylabel(r"$\rho_1$")
        plt.title(f"Depth sweep, n={n}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(outdir / f"depth_sweep_n{n}.pdf")
        plt.close()


def _plot_spectrum(spec: pd.DataFrame, outdir: Path) -> None:
    if spec.empty:
        return
    for (n, fac), cell in spec.groupby(["n", "depth_factor"]):
        plt.figure(figsize=(8, 5))
        for family, g in cell.groupby("family"):
            jid = sorted(g.job_id.unique())[0]
            s = g[g.job_id == jid].sort_values("eigen_index")
            plt.plot(s.eigen_index, s.eigenvalue, label=family)
        plt.yscale("log")
        plt.xlabel("eigenvalue index")
        plt.ylabel(r"$\lambda_j(\hat C)$")
        plt.title(fr"Empirical covariance spectra, n={n}, d/n={fac:g}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(outdir / f"spectrum_n{n}_f{fac:g}.pdf")
        plt.close()


def _reproduction_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[(df.parameter_distribution == "uniform_pi") & (df.measurement_basis == "computational") & (df.bitflip_rate == 0)]
    generic = base[base.group == "generic"]
    u1 = base[base.family == "U1-RZ-XY-line"]
    rows = []
    for n in sorted(base.n.unique()):
        for group_name, g in [("generic", generic[generic.n == n]), ("U1", u1[u1.n == n])]:
            if g.empty:
                continue
            k1 = g[g.k == 1]
            k2 = g[g.k == 2]
            rows.append({
                "n": n,
                "group": group_name,
                "rho1": k1.enhancement.mean(),
                "rho2": k2.enhancement.mean(),
                "deff": k1.deff_pairwise.mean(),
                "Ffull_over_FQ": k1.Ffull_over_FQ_mean.mean(),
                "circuits": k1.job_id.nunique(),
            })
    return pd.DataFrame(rows)


def analyze_results(input_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    circuits = _concat(input_root, "circuit_summary.csv")
    if circuits.empty:
        raise RuntimeError(f"no circuit_summary.csv files found under {input_root}")
    circuits = circuits.drop_duplicates()
    circuits.to_csv(output_dir / "all_circuit_results.csv", index=False)
    seeds = _concat(input_root, "seed_ledger.csv")
    if not seeds.empty:
        seeds.drop_duplicates().to_csv(output_dir / "seed_ledger.csv", index=False)
    tangents = _concat(input_root, "tangent_summary.csv")
    if not tangents.empty:
        tangents.drop_duplicates().to_csv(output_dir / "all_tangent_results.csv", index=False)
    spectrum = _concat(input_root, "spectrum.csv")
    if not spectrum.empty:
        spectrum.drop_duplicates().to_csv(output_dir / "all_spectra.csv", index=False)

    agg = aggregate_circuit_results(circuits)
    pooled = pooled_group_results(circuits)
    ratio_rho = group_ratio_bootstrap(circuits, metric="enhancement")
    ratio_retention = group_ratio_bootstrap(circuits, metric="actual_retention")
    ratio_deff = group_ratio_bootstrap(circuits, metric="deff_pairwise")
    agg.to_csv(output_dir / "aggregate_by_family.csv", index=False)
    pooled.to_csv(output_dir / "aggregate_by_group.csv", index=False)
    ratio_rho.to_csv(output_dir / "u1_to_generic_enhancement_ratio.csv", index=False)
    ratio_retention.to_csv(output_dir / "u1_to_generic_actual_retention_ratio.csv", index=False)
    ratio_deff.to_csv(output_dir / "u1_to_generic_deff_ratio.csv", index=False)
    repro = _reproduction_table(circuits)
    if not repro.empty:
        repro.to_csv(output_dir / "reproduction_table.csv", index=False)
    _plot_core(circuits, figures)
    _plot_depth(circuits, figures)
    _plot_spectrum(spectrum, figures)

    report = [
        "# Spectral accessibility numerical report", "",
        "Scientific outcomes are reported, not used as CI pass/fail criteria. CI failures are reserved for implementation invariants and explicit exact-reproduction checks.", "",
        f"Independent fixed-circuit conditions: **{circuits.job_id.nunique()}**",
        f"Circuit-summary rows: **{len(circuits)}**", "", "## Profiles present", "",
        ", ".join(sorted(map(str, circuits.profile.unique()))), "",
    ]
    if not repro.empty:
        report.extend(["## Compact reproduction table", "", "```text", repro.to_string(index=False), "```", ""])
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    manifest = []
    for p in sorted(output_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.csv":
            manifest.append({"path": str(p.relative_to(output_dir)), "bytes": p.stat().st_size, "sha256": _sha256(p)})
    pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)
