"""
Training loop for BaFuse model.

Handles:
- Forward pass through model
- Loss computation
- Backprop and optimisation
- Validation and early stopping
- Checkpoint saving
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import yaml
import logging
import sys
import os

# Allow running from src/ directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.bafuse import BaFuse
from losses import SoHPredictionLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move all tensor values in a batch dict to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R²."""
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: BaFuse,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Train for one epoch.

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = _batch_to_device(batch, device)

        discharge = batch["discharge"]   # (B, seq_len, F) or (B, F)
        eis = batch["eis"]               # (B, eis_F)
        physics = batch["physics"]       # (B, phys_F)
        soh_label = batch["soh_label"]   # (B,)
        cycle_idx = batch.get("cycle_idx")

        optimizer.zero_grad()
        outputs = model(discharge, eis, physics)
        pred = outputs["soh_pred"]

        loss = criterion(pred, soh_label, cycle_age=cycle_idx)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(train_loader), 1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: BaFuse,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Validate model.

    Returns:
        Tuple of (avg_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_preds: list = []
    all_targets: list = []

    with torch.no_grad():
        for batch in val_loader:
            batch = _batch_to_device(batch, device)

            discharge = batch["discharge"]
            eis = batch["eis"]
            physics = batch["physics"]
            soh_label = batch["soh_label"]
            cycle_idx = batch.get("cycle_idx")

            outputs = model(discharge, eis, physics)
            pred = outputs["soh_pred"]

            loss = criterion(pred, soh_label, cycle_age=cycle_idx)
            total_loss += loss.item()

            all_preds.append(pred.view(-1).cpu().numpy())
            all_targets.append(soh_label.view(-1).cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = _compute_metrics(preds, targets)
    avg_loss = total_loss / max(len(val_loader), 1)

    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Full training procedure
# ---------------------------------------------------------------------------

def train(
    config_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
) -> Tuple[BaFuse, Dict]:
    """
    Full training procedure with early stopping and checkpointing.

    Returns:
        Tuple of (trained_model, training_history)
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info(f"Training on device: {torch_device}")

    # ---- Model ----
    model_cfg = cfg["model"]
    enc_cfg = cfg["encoders"]
    fus_cfg = cfg["fusion"]

    model = BaFuse(
        discharge_input_size=model_cfg["discharge_input_size"],
        eis_num_frequencies=model_cfg.get("eis_num_frequencies", 3),
        physics_num_features=model_cfg.get("physics_num_features", 4),
        latent_dim=model_cfg.get("latent_dim", 64),
        fusion_method=model_cfg.get("fusion_method", "cross_attention"),
        fusion_dim=model_cfg.get("fusion_dim", 128),
    ).to(torch_device)

    # ---- Loss ----
    loss_cfg = cfg["loss"]
    criterion = SoHPredictionLoss(
        base_loss=loss_cfg.get("base_loss", "mse"),
        lambda_physics=loss_cfg.get("lambda_physics", 0.1),
        lambda_smooth=loss_cfg.get("lambda_smooth", 0.05),
    )

    # ---- Optimiser ----
    train_cfg = cfg["training"]
    lr = float(train_cfg.get("learning_rate", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-5))
    opt_name = train_cfg.get("optimizer", "adam").lower()

    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    # ---- LR Scheduler: linear warmup then cosine ----
    num_epochs    = int(train_cfg.get("num_epochs", 100))
    warmup_epochs = int(train_cfg.get("scheduler", {}).get("warmup_epochs", 5))

    def _lr_lambda(epoch: int) -> float:
        """Linear warmup for first warmup_epochs, then cosine decay."""
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # ---- Early stopping ----
    es_cfg = train_cfg.get("early_stopping", {})
    patience = int(es_cfg.get("patience", 15))
    min_delta = float(es_cfg.get("min_delta", 0.001))
    best_val_loss = float("inf")
    best_val_mae  = float("inf")   # P4: track MAE — more stable than loss on small val set
    patience_counter = 0

    # ---- Checkpoint dir ----
    out_cfg = cfg.get("output", {})
    ckpt_dir = Path(out_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: Dict = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": [],
    }

    logger.info(f"Starting training for {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, torch_device)
        val_loss, val_metrics = validate(model, val_loader, criterion, torch_device)

        # Step LR every epoch (lambda handles warmup+cosine)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_r2"].append(val_metrics["r2"])

        logger.info(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"MAE={val_metrics['mae']:.3f}  RMSE={val_metrics['rmse']:.3f}  "
            f"R²={val_metrics['r2']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best checkpoint — use val MAE (more stable than loss on small val set)
        current_mae = val_metrics["mae"]
        if current_mae < best_val_mae - min_delta:
            best_val_loss = val_loss
            best_val_mae  = current_mae
            patience_counter = 0
            ckpt_path = ckpt_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info(f"  ✓ Saved best checkpoint (MAE={current_mae:.3f}%) → {ckpt_path}")
        else:
            patience_counter += 1

        if es_cfg.get("enabled", True) and patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # Load best weights before returning
    best_ckpt = ckpt_dir / "best_model.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=torch_device)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded best checkpoint weights")

    return model, history
