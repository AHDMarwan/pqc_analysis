from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd

from .circuits import (
    family_group,
    parameter_count,
    sample_parameter_directions,
    sample_parameters,
    simulate_vqc_tangent_batch,
)
from .geometry import (
    covariance_spectrum,
    cross_validated_kyfan_recovery,
    direct_readout_retention,
    kyfan_sum,
    normalized_visible_scores_from_probabilities,
    pairwise_purity_and_deff,
    random_projector_retention_control,
    tangent_readout_retentions,
    walsh_readout_basis,
)
from .measurement import (
    apply_independent_bitflip_noise,
    probability_tangent_batch,
    rotate_measurement_basis,
)
from .profiles import Profile
from .seeding import stable_seed


def _environment() -> dict[str, str]:
    import scipy
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def _qfi(phis: np.ndarray) -> np.ndarray:
    return 4.0 * np.sum(np.abs(phis) ** 2, axis=1).real


def _analyze_condition(
    *,
    psi: np.ndarray,
    phis: np.ndarray,
    n: int,
    family: str,
    instance: int,
    depth: int,
    depth_factor: float,
    pcount: int,
    requested_tangents: int,
    profile: Profile,
    parameter_distribution: str,
    basis: str,
    bitflip_rate: float,
    master_seed: int,
    job_id: str,
    prefix: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if prefix is not None:
        phis = phis[:prefix]
        requested_tangents = prefix

    basis_seed = stable_seed(job_id + f"|basis|{basis}", master_seed)
    psi_m, phis_m = rotate_measurement_basis(psi, phis, n, basis, basis_seed)
    p, dp = probability_tangent_batch(psi_m, phis_m)
    p, dp = apply_independent_bitflip_noise(p, dp, n, bitflip_rate)
    fq = _qfi(phis_m)
    info = normalized_visible_scores_from_probabilities(p, dp, fq)
    u = info["U"]
    if len(u) < max(6, requested_tangents // 2):
        raise RuntimeError(f"too few regular tangents in {job_id}: {len(u)}/{requested_tangents}")
    if np.any(info["Ffull"][info["regular"]] > info["FQ"][info["regular"]] * (1 + 2e-7)):
        raise RuntimeError(f"Ffull > FQ in {job_id}")

    purity, deff = pairwise_purity_and_deff(u)
    support_idx = np.flatnonzero(info["support"])
    ffull = info["Ffull"][info["regular"]]
    fqreg = info["FQ"][info["regular"]]
    eig = covariance_spectrum(u) if profile.spectrum else np.array([], dtype=float)

    common = {
        "profile": profile.name,
        "job_id": job_id,
        "family": family,
        "group": family_group(family),
        "n": n,
        "depth": depth,
        "depth_factor": depth_factor,
        "instance": instance,
        "parameter_distribution": parameter_distribution,
        "measurement_basis": basis,
        "bitflip_rate": float(bitflip_rate),
        "tangent_prefix": int(prefix if prefix is not None else requested_tangents),
        "parameter_count": pcount,
        "requested_tangents": requested_tangents,
        "regular_tangents": len(u),
        "support_size": len(support_idx),
        "Nsupp": len(support_idx) - 1,
        "FQ_mean": float(np.mean(fqreg)),
        "Ffull_mean": float(np.mean(ffull)),
        "Ffull_over_FQ_mean": float(np.mean(ffull / fqreg)),
        "pairwise_purity": purity,
        "deff_pairwise": deff,
    }

    circuit_rows: list[dict[str, Any]] = []
    tangent_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []

    for k in [x for x in profile.readout_orders if x < n]:
        diag = direct_readout_retention(u, info["p_support"], support_idx, n, k)
        row = dict(common)
        row.update({
            "k": k,
            "readout_rank": diag["rank"],
            "actual_retention": diag["retention"],
            "rank_baseline": diag["baseline"],
            "enhancement": diag["enhancement"],
        })
        if profile.spectrum:
            optimum = kyfan_sum(eig, int(diag["rank"]))
            row["sample_kyfan"] = optimum
            row["alignment_to_sample_kyfan"] = diag["retention"] / optimum if optimum > 0 else np.nan
            cv = cross_validated_kyfan_recovery(
                u,
                int(diag["rank"]),
                stable_seed(job_id + f"|cv_kyfan|k{k}", master_seed),
            )
            row["cv_kyfan_train_recovery"] = cv["train"]
            row["cv_kyfan_test_recovery"] = cv["test"]
            row["cv_kyfan_train_size"] = cv["train_size"]
            row["cv_kyfan_test_size"] = cv["test_size"]
            if profile.spectrum_random_projectors:
                ctrl = random_projector_retention_control(
                    u,
                    int(diag["rank"]),
                    profile.spectrum_random_projectors,
                    stable_seed(job_id + f"|random_projectors|k{k}", master_seed),
                )
                row["random_projector_mean"] = ctrl["mean"]
                row["random_projector_std"] = ctrl["std"]
                row["random_projector_expected"] = ctrl["expected"]
        circuit_rows.append(row)

        if profile.save_tangent_rows:
            q, rank = walsh_readout_basis(info["p_support"], support_idx, n, k)
            vals = tangent_readout_retentions(u, q)
            reg_idx = np.flatnonzero(info["regular"])
            for j, source_idx in enumerate(reg_idx):
                tangent_rows.append({
                    "profile": profile.name,
                    "job_id": job_id,
                    "family": family,
                    "group": family_group(family),
                    "n": n,
                    "depth": depth,
                    "depth_factor": depth_factor,
                    "instance": instance,
                    "parameter_distribution": parameter_distribution,
                    "measurement_basis": basis,
                    "bitflip_rate": float(bitflip_rate),
                    "k": k,
                    "tangent_index": int(source_idx),
                    "FQ": float(fq[source_idx]),
                    "Ffull": float(info["Ffull"][source_idx]),
                    "Ffull_over_FQ": float(info["Ffull"][source_idx] / fq[source_idx]),
                    "readout_retention": float(vals[j]),
                    "readout_rank": int(rank),
                })

    if profile.spectrum:
        for j, val in enumerate(eig, start=1):
            spectrum_rows.append({
                "profile": profile.name,
                "job_id": job_id,
                "family": family,
                "group": family_group(family),
                "n": n,
                "depth": depth,
                "depth_factor": depth_factor,
                "instance": instance,
                "parameter_distribution": parameter_distribution,
                "measurement_basis": basis,
                "bitflip_rate": float(bitflip_rate),
                "eigen_index": j,
                "eigenvalue": float(val),
            })
    return circuit_rows, tangent_rows, spectrum_rows


def run_task(
    *,
    profile: Profile,
    family: str,
    n: int,
    depth_factor: float,
    instance_start: int,
    instance_stop: int,
    tangents: int,
    parameter_distribution: str,
    output_dir: Path,
    master_seed: int = 20260809,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    depth = max(1, int(round(depth_factor * n)))
    circuit_rows: list[dict[str, Any]] = []
    tangent_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for instance in range(instance_start, instance_stop):
        base_job_id = f"{family}|n{n}|d{depth}|i{instance}"
        seed = stable_seed(base_job_id, master_seed)
        rng = np.random.default_rng(seed)
        pcount = parameter_count(n, depth, family)
        theta = sample_parameters(rng, pcount, parameter_distribution)
        directions = sample_parameter_directions(rng, tangents, pcount)
        arch_seed = stable_seed(base_job_id + "|arch", master_seed)
        psi, phis, pcount2 = simulate_vqc_tangent_batch(n, depth, family, theta, directions, arch_seed)
        if pcount2 != pcount:
            raise RuntimeError("parameter-count mismatch")

        state_norm_error = abs(float(np.vdot(psi, psi).real) - 1.0)
        horizontal_max = float(np.max(np.abs(phis @ np.conjugate(psi))))
        if state_norm_error > 5e-12 or horizontal_max > 5e-10:
            raise RuntimeError(
                f"state/tangent invariant failed in {base_job_id}: norm={state_norm_error}, horizontal={horizontal_max}"
            )

        seed_rows.append({
            "job_id": base_job_id,
            "master_seed": master_seed,
            "simulation_seed": seed,
            "architecture_seed": arch_seed,
            "family": family,
            "n": n,
            "depth": depth,
            "instance": instance,
            "parameter_distribution": parameter_distribution,
        })

        prefixes = profile.tangent_prefixes or (tangents,)
        for prefix in prefixes:
            if prefix > tangents:
                raise ValueError("tangent prefix cannot exceed simulated tangents")
            for basis in profile.measurement_bases:
                for noise in profile.bitflip_rates:
                    condition_id = base_job_id + f"|init={parameter_distribution}|basis={basis}|bf={noise:g}|M={prefix}"
                    c, t, s = _analyze_condition(
                        psi=psi,
                        phis=phis,
                        n=n,
                        family=family,
                        instance=instance,
                        depth=depth,
                        depth_factor=depth_factor,
                        pcount=pcount,
                        requested_tangents=prefix,
                        profile=profile,
                        parameter_distribution=parameter_distribution,
                        basis=basis,
                        bitflip_rate=noise,
                        master_seed=master_seed,
                        job_id=condition_id,
                        prefix=prefix,
                    )
                    circuit_rows.extend(c)
                    tangent_rows.extend(t)
                    spectrum_rows.extend(s)

    pd.DataFrame(circuit_rows).to_csv(output_dir / "circuit_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output_dir / "seed_ledger.csv", index=False)
    if tangent_rows:
        pd.DataFrame(tangent_rows).to_csv(output_dir / "tangent_summary.csv", index=False)
    if spectrum_rows:
        pd.DataFrame(spectrum_rows).to_csv(output_dir / "spectrum.csv", index=False)
    metadata = {
        "profile": profile.to_dict(),
        "task": {
            "family": family,
            "n": n,
            "depth_factor": depth_factor,
            "instance_start": instance_start,
            "instance_stop": instance_stop,
            "tangents": tangents,
            "parameter_distribution": parameter_distribution,
        },
        "master_seed": master_seed,
        "environment": _environment(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
