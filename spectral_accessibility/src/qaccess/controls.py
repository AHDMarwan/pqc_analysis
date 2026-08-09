from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, kstest

from .geometry import pairwise_purity_and_deff, random_projector_retention_control
from .seeding import stable_seed


def isotropic_controls(master_seed: int = 20260809) -> pd.DataFrame:
    rows = []
    for n_score in (31, 63, 255):
        rng = np.random.default_rng(stable_seed(f"iso|{n_score}", master_seed))
        m = 2000
        u = rng.normal(size=(m, n_score))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        purity, deff = pairwise_purity_and_deff(u)
        for rank in sorted(set([1, max(2, n_score // 8), max(3, n_score // 3)])):
            q = np.eye(n_score)[:, :rank]
            values = np.sum((u @ q) ** 2, axis=1)
            a, b = rank / 2.0, (n_score - rank) / 2.0
            ks = kstest(values, beta(a, b).cdf)
            rows.append({
                "N": n_score,
                "M": m,
                "rank": rank,
                "sample_mean": float(values.mean()),
                "expected_mean": rank / n_score,
                "sample_variance": float(values.var(ddof=1)),
                "expected_variance": 2 * rank * (n_score - rank) / (n_score**2 * (n_score + 2)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "purity": purity,
                "expected_purity": 1 / n_score,
                "deff_pairwise": deff,
            })
    return pd.DataFrame(rows)


def anisotropic_relative_orientation_control(master_seed: int = 20260809) -> pd.DataFrame:
    rng = np.random.default_rng(stable_seed("anisotropic-control", master_seed))
    n_score = 63
    scales = np.geomspace(1.0, 0.03, n_score)
    u = rng.normal(size=(1500, n_score)) * scales[None, :]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rows = []
    for rank in (4, 12, 24):
        ctrl = random_projector_retention_control(
            u, rank, samples=300, seed=stable_seed(f"anisotropic-r{rank}", master_seed)
        )
        rows.append({"N": n_score, "rank": rank, **ctrl})
    return pd.DataFrame(rows)
