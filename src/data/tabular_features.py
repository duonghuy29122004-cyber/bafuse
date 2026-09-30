"""
Leak-free tabular features for sklearn / tiny-model baselines.

The old run_sklearn_pipeline.py baked `capacity_fade` from capacity_ahr
(the SOH label) into X. This module never uses capacity as a feature.

Train-only statistics are used for z-scored / normalised derived columns
so val/test cannot leak population moments.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dataset import get_cutoff_voltage, get_nominal_capacity

TABULAR_FEATURE_COLS: List[str] = [
    "voltage_mean",
    "voltage_min",
    "voltage_max",
    "voltage_std",
    "current_mean",
    "current_std",
    "temp_mean",
    "temp_std",
    "duration_s",
    "impedance_ohm",
    "re_ohm",
    "rct_ohm",
    "cycle_age_norm",
    "cutoff_v",
    "voltage_droop",
    "impedance_rise",
]


def attach_duration_from_raw(
    paired_df: pd.DataFrame,
    discharge_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Add duration_s from raw time series when pairing pickle has no duration."""
    out = paired_df.copy()
    if "duration_s" in out.columns and out["duration_s"].notna().all():
        return out
    dur = (
        discharge_raw.groupby(["battery_id", "cycle_idx"])["time_s"]
        .agg(lambda s: float(np.nanmax(s) - np.nanmin(s)))
        .rename("duration_s")
        .reset_index()
        .rename(columns={"cycle_idx": "discharge_cycle"})
    )
    out = out.drop(columns=["duration_s"], errors="ignore")
    out = out.merge(dur, on=["battery_id", "discharge_cycle"], how="left")
    return out


def soh_from_capacity(df: pd.DataFrame) -> np.ndarray:
    nominal = df["battery_id"].map(get_nominal_capacity).astype(float)
    return np.clip(df["capacity_ahr"].astype(float) / nominal, 0.0, 1.0).to_numpy()


def fit_feature_stats(train_df: pd.DataFrame) -> Dict[str, float]:
    ages = train_df["discharge_cycle"].astype(float)
    n_ref = float(np.percentile(ages, 90)) if len(ages) > 10 else 168.0
    n_ref = max(n_ref, 1.0)
    imp = train_df["impedance_ohm"].astype(float)
    return {
        "n_ref": n_ref,
        "imp_mean": float(imp.mean()),
        "imp_std": float(imp.std()) + 1e-8,
        "dur_median": float(train_df["duration_s"].median())
        if "duration_s" in train_df.columns
        else 0.0,
    }


def _add_derived(df: pd.DataFrame, stats: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    if "duration_s" not in out.columns:
        out["duration_s"] = stats.get("dur_median", 0.0)
    out["duration_s"] = out["duration_s"].fillna(stats.get("dur_median", 0.0))
    out["cutoff_v"] = out["battery_id"].map(get_cutoff_voltage).astype(float)
    out["cycle_age_norm"] = out["discharge_cycle"].astype(float) / stats["n_ref"]
    vrange = (4.2 - out["cutoff_v"]).clip(lower=0.1)
    out["voltage_droop"] = (out["voltage_min"] - out["cutoff_v"]) / vrange
    out["impedance_rise"] = (
        out["impedance_ohm"].astype(float) - stats["imp_mean"]
    ) / stats["imp_std"]
    for col in ("re_ohm", "rct_ohm", "voltage_std", "current_std", "temp_std"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    return out


def build_xy(
    df: pd.DataFrame,
    stats: Dict[str, float],
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    cols = feature_cols or TABULAR_FEATURE_COLS
    enriched = _add_derived(df, stats)
    present = [c for c in cols if c in enriched.columns]
    X = enriched[present].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
    y = soh_from_capacity(enriched).astype(np.float32)
    return X, y, present
