from __future__ import annotations

import numpy as np
import pandas as pd

from .seeding import stable_seed


def bootstrap_mean_ci(
    values,
    *,
    confidence: float = 0.95,
    n_resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    means = x[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def aggregate_circuit_results(df: pd.DataFrame, master_seed: int = 20260809) -> pd.DataFrame:
    keys = ["profile", "family", "group", "n", "depth_factor", "parameter_distribution", "measurement_basis", "bitflip_rate", "tangent_prefix", "k"]
    metrics = ["enhancement", "actual_retention", "deff_pairwise", "Ffull_over_FQ_mean"]
    out = []
    for key, g in df.groupby(keys, dropna=False):
        base = dict(zip(keys, key))
        base["circuits"] = int(g["job_id"].nunique())
        for metric in metrics:
            mean, lo, hi = bootstrap_mean_ci(
                g[metric].values,
                seed=stable_seed("aggregate|" + "|".join(map(str, key)) + f"|{metric}", master_seed),
            )
            row = dict(base)
            row.update({
                "metric": metric,
                "mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "median": float(np.nanmedian(g[metric].values)),
                "std": float(np.nanstd(g[metric].values, ddof=1)) if len(g) > 1 else np.nan,
            })
            out.append(row)
    return pd.DataFrame(out)


def pooled_group_results(df: pd.DataFrame, master_seed: int = 20260809) -> pd.DataFrame:
    keys = ["profile", "group", "n", "depth_factor", "parameter_distribution", "measurement_basis", "bitflip_rate", "tangent_prefix", "k"]
    metrics = ["enhancement", "actual_retention", "deff_pairwise", "Ffull_over_FQ_mean"]
    out = []
    for key, g in df.groupby(keys, dropna=False):
        base = dict(zip(keys, key))
        base["circuits"] = int(g["job_id"].nunique())
        for metric in metrics:
            mean, lo, hi = bootstrap_mean_ci(
                g[metric].values,
                seed=stable_seed("pooled|" + "|".join(map(str, key)) + f"|{metric}", master_seed),
            )
            out.append({**base, "metric": metric, "mean": mean, "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(out)


def group_ratio_bootstrap(
    df: pd.DataFrame,
    *,
    metric: str,
    n_resamples: int = 5000,
    master_seed: int = 20260809,
) -> pd.DataFrame:
    condition_keys = ["profile", "n", "depth_factor", "parameter_distribution", "measurement_basis", "bitflip_rate", "tangent_prefix", "k"]
    rows = []
    for key, g in df.groupby(condition_keys, dropna=False):
        a = g[g.group == "u1"][metric].dropna().to_numpy(float)
        b = g[g.group == "generic"][metric].dropna().to_numpy(float)
        if len(a) < 2 or len(b) < 2:
            continue
        seed = stable_seed("ratio|" + "|".join(map(str, key)) + f"|{metric}", master_seed)
        rng = np.random.default_rng(seed)
        ia = rng.integers(0, len(a), size=(n_resamples, len(a)))
        ib = rng.integers(0, len(b), size=(n_resamples, len(b)))
        ratios = a[ia].mean(axis=1) / b[ib].mean(axis=1)
        base = dict(zip(condition_keys, key))
        rows.append({
            **base,
            "metric": metric,
            "u1_circuits": len(a),
            "generic_circuits": len(b),
            "ratio": float(a.mean() / b.mean()),
            "ci95_low": float(np.quantile(ratios, 0.025)),
            "ci95_high": float(np.quantile(ratios, 0.975)),
        })
    return pd.DataFrame(rows)
