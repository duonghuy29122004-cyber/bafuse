"""
Evaluation of BaFuse model on test set.

Computes:
- Prediction error (MAE, RMSE, R²)
- Per-battery errors
- Contribution of each modality to final prediction
- Uncertainty quantification via MC Dropout
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R² on raw [0,1] SOH values.
    MAPE is computed as a percentage (multiply by 100 internally).
    For reporting: MAE% = mae * 100, RMSE% = rmse * 100.
    """
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    mape_denom = np.abs(targets) + 1e-8
    mape = float(np.mean(np.abs(preds - targets) / mape_denom) * 100)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def evaluate(model, test_loader: DataLoader, device: torch.device) -> Dict:
    """
    Evaluate model on test set.

    Returns:
        metrics_dict with overall and per-battery performance.
    """
    model.eval()

    all_preds: List[float] = []
    all_targets: List[float] = []
    all_battery_ids: List[str] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = _batch_to_device(batch, device)
            outputs = model(batch["discharge"], batch["eis"], batch["physics"])
            pred = outputs["soh_pred"].view(-1).cpu().numpy()
            target = batch["soh_label"].view(-1).cpu().numpy()

            all_preds.extend(pred.tolist())
            all_targets.extend(target.tolist())
            all_battery_ids.extend(
                batch["battery_id"] if isinstance(batch["battery_id"], list)
                else [batch["battery_id"]] * len(pred)
            )

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    overall = _compute_metrics(preds, targets)
    logger.info(
        f"Test results: MAE={overall['mae']*100:.2f}%  RMSE={overall['rmse']*100:.2f}%  "
        f"R^2={overall['r2']:.4f}  MAPE={overall['mape']:.2f}%"
    )

    # Per-battery breakdown
    df = pd.DataFrame({"battery_id": all_battery_ids, "pred": preds, "target": targets})
    per_battery: Dict[str, Dict] = {}
    for bid, grp in df.groupby("battery_id"):
        per_battery[str(bid)] = _compute_metrics(grp["pred"].values, grp["target"].values)

    return {
        "overall": overall,
        "per_battery": per_battery,
        "predictions": preds,
        "targets": targets,
        "battery_ids": all_battery_ids,
    }


# ---------------------------------------------------------------------------
# Modality contribution analysis
# ---------------------------------------------------------------------------

def analyze_modality_contribution(
    model, test_loader: DataLoader, device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Analyse each modality's contribution to SoH prediction.

    Uses attention weights (if available) and an ablation strategy:
    zeroing out one encoder's output at a time and measuring MAE increase.
    """
    model.eval()

    # ---- Collect full-model baseline predictions ----
    base_preds, all_targets = [], []
    d_latents, e_latents, p_latents = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = _batch_to_device(batch, device)
            out = model(batch["discharge"], batch["eis"], batch["physics"])
            base_preds.extend(out["soh_pred"].view(-1).cpu().numpy().tolist())
            all_targets.extend(batch["soh_label"].view(-1).cpu().numpy().tolist())
            d_latents.append(out["discharge_latent"].cpu().numpy())
            e_latents.append(out["eis_latent"].cpu().numpy())
            p_latents.append(out["physics_latent"].cpu().numpy())

    base_preds = np.array(base_preds)
    all_targets = np.array(all_targets)
    base_mae = float(np.mean(np.abs(base_preds - all_targets)))

    # ---- Ablation: zero-out each modality ----
    def _run_with_zeroed(modality: str) -> float:
        preds = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = _batch_to_device(batch, device)
                d = batch["discharge"]
                e = batch["eis"]
                p = batch["physics"]
                if modality == "discharge":
                    d = torch.zeros_like(d)
                elif modality == "eis":
                    e = torch.zeros_like(e)
                else:
                    p = torch.zeros_like(p)
                out = model(d, e, p)
                preds.extend(out["soh_pred"].view(-1).cpu().numpy().tolist())
        return float(np.mean(np.abs(np.array(preds) - all_targets)))

    mae_no_discharge = _run_with_zeroed("discharge")
    mae_no_eis = _run_with_zeroed("eis")
    mae_no_physics = _run_with_zeroed("physics")

    # Contribution = MAE increase when ablated (normalised)
    delta_d = max(mae_no_discharge - base_mae, 0.0)
    delta_e = max(mae_no_eis - base_mae, 0.0)
    delta_p = max(mae_no_physics - base_mae, 0.0)
    total = delta_d + delta_e + delta_p + 1e-8

    contributions = {
        "discharge": delta_d / total,
        "eis": delta_e / total,
        "physics": delta_p / total,
        "base_mae": base_mae,
        "mae_no_discharge": mae_no_discharge,
        "mae_no_eis": mae_no_eis,
        "mae_no_physics": mae_no_physics,
    }
    logger.info(f"Modality contributions: {contributions}")
    return contributions


# ---------------------------------------------------------------------------
# Uncertainty quantification via MC Dropout
# ---------------------------------------------------------------------------

def compute_uncertainty(
    model,
    test_loader: DataLoader,
    device: torch.device,
    num_mc_samples: int = 30,
) -> Dict[str, np.ndarray]:
    """
    Estimate prediction uncertainty using MC Dropout.

    Enables dropout at inference time and runs multiple stochastic passes.

    Returns:
        Dict with 'mean', 'std', 'targets'
    """
    # Enable dropout in eval mode (apply to all Dropout layers)
    def _enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    model.eval()
    model.apply(_enable_dropout)

    mc_preds = []   # (num_mc_samples, N)
    all_targets = []

    with torch.no_grad():
        for _ in range(num_mc_samples):
            preds_pass = []
            for i, batch in enumerate(test_loader):
                batch = _batch_to_device(batch, device)
                out = model(batch["discharge"], batch["eis"], batch["physics"])
                preds_pass.extend(out["soh_pred"].view(-1).cpu().numpy().tolist())
                if _ == 0:
                    all_targets.extend(batch["soh_label"].view(-1).cpu().numpy().tolist())
            mc_preds.append(preds_pass)

    mc_preds = np.array(mc_preds)  # (S, N)
    model.eval()  # restore eval-only dropout

    return {
        "mean": mc_preds.mean(axis=0),
        "std": mc_preds.std(axis=0),
        "targets": np.array(all_targets),
        "all_samples": mc_preds,
    }

