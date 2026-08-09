from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def compare_reproduction(
    observed_path: Path,
    reference_path: Path,
    *,
    mean_tolerance: float = 0.0006,
) -> pd.DataFrame:
    obs = pd.read_csv(observed_path)
    ref = pd.read_csv(reference_path)
    merged = ref.merge(obs, on=["n", "group"], suffixes=("_reported", "_observed"), how="left")
    for metric in ["rho1", "rho2", "deff", "Ffull_over_FQ"]:
        merged[f"abs_error_{metric}"] = np.abs(merged[f"{metric}_observed"] - merged[f"{metric}_reported"])
    merged["core_means_match"] = (
        (merged.abs_error_rho1 <= mean_tolerance)
        & (merged.abs_error_rho2 <= mean_tolerance)
        & (merged.abs_error_Ffull_over_FQ <= mean_tolerance)
    )
    merged["deff_relative_error"] = merged.abs_error_deff / merged.deff_reported
    merged["deff_matches"] = merged.deff_relative_error <= 0.001
    return merged
