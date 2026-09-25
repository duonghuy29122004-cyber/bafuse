"""
Shared evaluation metrics for BaFuse v2.

Provides:
    compute_metrics()              — MAE, RMSE, R² for any predictions/targets
    compute_per_battery_metrics()  — breakdown by battery_id
    compute_degradation_metrics()  — per-mode (LLI/LAM/CL) MAE/RMSE/R²
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_metrics(
    predictions: np.ndarray,
    targets:     np.ndarray,
    prefix:      str = "",
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R² between predictions and targets.

    Args:
        predictions : (N,) predicted values.
        targets     : (N,) ground-truth values.
        prefix      : Optional string prefix for metric keys (e.g. "soh_").

    Returns:
        Dict with keys: {prefix}mae, {prefix}rmse, {prefix}r2.
    """
    preds   = np.asarray(predictions, dtype=float).ravel()
    tgts    = np.asarray(targets,     dtype=float).ravel()

    if len(preds) == 0:
        return {f"{prefix}mae": np.nan, f"{prefix}rmse": np.nan, f"{prefix}r2": np.nan}

    mae  = float(np.mean(np.abs(preds - tgts)))
    rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
    ss_r = np.sum((tgts - preds) ** 2)
    ss_t = np.sum((tgts - np.mean(tgts)) ** 2)
    r2   = float(1.0 - ss_r / (ss_t + 1e-8))

    return {
        f"{prefix}mae":  mae,
        f"{prefix}rmse": rmse,
        f"{prefix}r2":   r2,
    }


def compute_per_battery_metrics(
    predictions: np.ndarray,
    targets:     np.ndarray,
    battery_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute MAE/RMSE/R² broken down by battery_id.

    Returns:
        Dict mapping battery_id → metrics dict.
    """
    df = pd.DataFrame({
        "pred":       np.asarray(predictions).ravel(),
        "target":     np.asarray(targets).ravel(),
        "battery_id": battery_ids,
    })
    result: Dict[str, Dict[str, float]] = {}
    for bid, grp in df.groupby("battery_id"):
        result[str(bid)] = compute_metrics(
            grp["pred"].values, grp["target"].values
        )
    return result


def compute_degradation_metrics(
    lli_pred: np.ndarray, lli_target: np.ndarray,
    lam_pred: np.ndarray, lam_target: np.ndarray,
    cl_pred:  np.ndarray, cl_target:  np.ndarray,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R² for each model-derived degradation-mode estimate.

    NOTE: Targets (LLI/LAM/CL) are model-derived from ECM fitting — they are
    NOT absolute physical ground truth.

    Returns:
        Flat dict with keys: mae_lli, rmse_lli, r2_lli, mae_lam, ...
    """
    metrics: Dict[str, float] = {}
    for name, pred, target in [
        ("lli", lli_pred, lli_target),
        ("lam", lam_pred, lam_target),
        ("cl",  cl_pred,  cl_target),
    ]:
        m = compute_metrics(pred, target, prefix=f"{name}_")
        metrics.update(m)
    return metrics
