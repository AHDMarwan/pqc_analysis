from __future__ import annotations

from itertools import combinations
import numpy as np


def normalized_visible_scores_from_probabilities(
    p: np.ndarray,
    dp: np.ndarray,
    fq: np.ndarray,
    floor: float = 1e-13,
    dp_outside_support_tolerance: float = 1e-9,
) -> dict[str, np.ndarray]:
    support = p > floor
    if np.any(np.abs(dp[:, ~support]) > dp_outside_support_tolerance):
        raise RuntimeError("nonregular probability tangent outside numerical support")
    ps = p[support]
    dps = dp[:, support]
    ffull = np.sum(dps * dps / ps[None, :], axis=1)
    regular = (ffull > 1e-13) & (fq > 1e-13) & np.isfinite(ffull) & np.isfinite(fq)
    u = dps[regular] / np.sqrt(ps)[None, :]
    u = u / np.sqrt(ffull[regular])[:, None]
    return {"p": p, "support": support, "p_support": ps, "dp_support": dps, "Ffull": ffull, "FQ": fq, "regular": regular, "U": u}


def pairwise_purity_and_deff(u: np.ndarray) -> tuple[float, float]:
    """Unbiased U-statistic for Tr(C^2); inverse is a diagnostic, not unbiased."""
    m = u.shape[0]
    if m < 2:
        return np.nan, np.nan
    g = u @ u.T
    off = np.sum(g * g) - np.sum(np.diag(g) ** 2)
    purity = max(float(off / (m * (m - 1))), 1e-15)
    return purity, 1.0 / purity


def covariance_spectrum(u: np.ndarray) -> np.ndarray:
    m, s = u.shape
    gram = (u @ u.T) / m if m <= s else (u.T @ u) / m
    eig = np.linalg.eigvalsh((gram + gram.T) / 2.0)
    eig = np.sort(np.clip(eig, 0.0, None))[::-1]
    eig = eig[eig > 1e-13]
    if eig.sum() > 0:
        eig /= eig.sum()
    return eig


def walsh_readout_basis(
    p_support: np.ndarray,
    support_indices: np.ndarray,
    n: int,
    k: int,
    svd_tolerance: float = 1e-10,
) -> tuple[np.ndarray, int]:
    bits = ((support_indices[:, None] >> (n - 1 - np.arange(n))) & 1)
    z = 1 - 2 * bits
    cols = []
    for order in range(1, k + 1):
        for subset in combinations(range(n), order):
            cols.append(np.prod(z[:, subset], axis=1).astype(float))
    if not cols:
        return np.zeros((len(p_support), 0)), 0
    f = np.column_stack(cols)
    means = p_support @ f
    w = np.sqrt(p_support)[:, None] * (f - means[None, :])
    q, s, _ = np.linalg.svd(w, full_matrices=False)
    threshold = svd_tolerance * max(1.0, s[0] if len(s) else 1.0)
    rank = int(np.sum(s > threshold))
    return q[:, :rank], rank


def direct_readout_retention(
    u: np.ndarray,
    p_support: np.ndarray,
    support_indices: np.ndarray,
    n: int,
    k: int,
    svd_tolerance: float = 1e-10,
) -> dict[str, float]:
    q, rank = walsh_readout_basis(p_support, support_indices, n, k, svd_tolerance=svd_tolerance)
    if rank == 0:
        return {"retention": 0.0, "rank": 0, "baseline": np.nan, "enhancement": np.nan}
    coeff = u @ q
    retention = float(np.mean(np.sum(coeff * coeff, axis=1)))
    n_score = len(p_support) - 1
    baseline = rank / n_score if n_score > 0 else np.nan
    return {"retention": retention, "rank": int(rank), "baseline": float(baseline), "enhancement": float(retention / baseline) if baseline > 0 else np.nan}


def tangent_readout_retentions(u: np.ndarray, q: np.ndarray) -> np.ndarray:
    coeff = u @ q
    return np.sum(coeff * coeff, axis=1)


def kyfan_sum(eig: np.ndarray, rank: int) -> float:
    if rank <= 0:
        return 0.0
    return float(np.sum(eig[: min(rank, len(eig))]))


def cross_validated_kyfan_recovery(
    u: np.ndarray,
    rank: int,
    seed: int,
    train_fraction: float = 0.5,
) -> dict[str, float]:
    """Learn a leading covariance subspace on one split, evaluate on hold-out."""
    m, n_score = u.shape
    if m < 4 or not 0 < rank < n_score:
        return {"train": np.nan, "test": np.nan, "train_size": 0, "test_size": 0}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    n_train = min(max(2, int(round(train_fraction * m))), m - 2)
    train = u[perm[:n_train]]
    test = u[perm[n_train:]]
    c_train = (train.T @ train) / len(train)
    eigvals, eigvecs = np.linalg.eigh((c_train + c_train.T) / 2.0)
    idx = np.argsort(eigvals)[::-1][:rank]
    q = eigvecs[:, idx]
    return {
        "train": float(np.mean(np.sum((train @ q) ** 2, axis=1))),
        "test": float(np.mean(np.sum((test @ q) ** 2, axis=1))),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
    }


def random_projector_retention_control(
    u: np.ndarray,
    rank: int,
    samples: int,
    seed: int,
) -> dict[str, float]:
    n_score = u.shape[1]
    if not 0 < rank < n_score:
        return {"mean": np.nan, "std": np.nan, "expected": rank / n_score}
    rng = np.random.default_rng(seed)
    vals = np.empty(samples, dtype=float)
    for i in range(samples):
        z = rng.normal(size=(n_score, rank))
        q, _ = np.linalg.qr(z, mode="reduced")
        proj = u @ q
        vals[i] = np.mean(np.sum(proj * proj, axis=1))
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)), "expected": float(rank / n_score)}
