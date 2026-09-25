"""
Multi-Task Trainer for BaFuse v2 (Stage-2/3).

Implements a dataset-aware training strategy:

    Task A (NASA batch):
        discharge + EIS + physics → BaFuseV2.forward() → SOH loss

    Task B (Mendeley batch, optional):
        EIS → BaFuseV2.forward_eis_only() → degradation loss (LLI/LAM/CL)

    L_total = λ_soh * L_soh + λ_deg * L_deg

IMPORTANT:
  - NASA samples do NOT have LLI/LAM/CL labels.
  - Mendeley samples provide model-derived degradation-mode labels only.
  - Cross-dataset sample pairing is NOT done (cells are different physically).

Training modes (configurable):
    "nasa_only"   — SOH only, no degradation task (original BaFuse behaviour).
    "staged"      — Stage-2: joint training after Stage-1 EIS pre-training.
    "joint"       — Both tasks from scratch simultaneously.

Usage:
    trainer = MultiTaskTrainer(model, config)
    history = trainer.fit(
        nasa_train_loader, nasa_val_loader,
        mendeley_train_loader, mendeley_val_loader  # optional
    )
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.bafuse_v2 import BaFuseV2
from ..losses import SoHPredictionLoss, DegradationLoss

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _soh_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_r = np.sum((targets - preds) ** 2)
    ss_t = np.sum((targets - np.mean(targets)) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _deg_metrics(
    lli_p: np.ndarray, lli_t: np.ndarray,
    lam_p: np.ndarray, lam_t: np.ndarray,
    cl_p:  np.ndarray, cl_t:  np.ndarray,
) -> Dict[str, float]:
    def _mae(a, b): return float(np.mean(np.abs(a - b)))
    def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

    return {
        "mae_lli":  _mae(lli_p, lli_t),
        "mae_lam":  _mae(lam_p, lam_t),
        "mae_cl":   _mae(cl_p, cl_t),
        "rmse_lli": _rmse(lli_p, lli_t),
        "rmse_lam": _rmse(lam_p, lam_t),
        "rmse_cl":  _rmse(cl_p, cl_t),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class MultiTaskTrainer:
    """
    Dataset-aware multi-task trainer for BaFuseV2.

    Args:
        model      : BaFuseV2 instance (may carry Stage-1 EIS weights).
        config     : Training configuration dict with sections:
            training: {lr, weight_decay, num_epochs, patience, min_delta,
                       lambda_soh, lambda_deg, optimizer, training_mode}
            loss:     {base_loss, lambda_physics, lambda_smooth,
                       w_lli, w_lam, w_cl}
            output:   {checkpoint_dir}
        device     : "cpu" | "cuda".
    """

    def __init__(
        self,
        model:  BaFuseV2,
        config: Dict,
        device: str = "cpu",
    ):
        self.model  = model
        self.cfg    = config
        self.device = torch.device(device)
        self.model.to(self.device)

        t_cfg  = config.get("training", {})
        l_cfg  = config.get("loss",     {})
        o_cfg  = config.get("output",   {})

        self.training_mode = t_cfg.get("training_mode", "nasa_only")
        self.lambda_soh    = float(t_cfg.get("lambda_soh", 1.0))
        self.lambda_deg    = float(t_cfg.get("lambda_deg", 0.5))

        # ---- Loss functions ------------------------------------------------
        self.soh_criterion = SoHPredictionLoss(
            base_loss=l_cfg.get("base_loss", "mse"),
            lambda_physics=float(l_cfg.get("lambda_physics", 0.1)),
            lambda_smooth=float(l_cfg.get("lambda_smooth", 0.05)),
        )
        self.deg_criterion = DegradationLoss(
            base_loss=l_cfg.get("base_loss", "mse"),
            w_lli=float(l_cfg.get("w_lli", 1.0)),
            w_lam=float(l_cfg.get("w_lam", 1.0)),
            w_cl=float(l_cfg.get("w_cl",  1.0)),
        )

        # ---- Optimiser -----------------------------------------------------
        lr  = float(t_cfg.get("learning_rate", 5e-4))
        wd  = float(t_cfg.get("weight_decay",  1e-5))
        opt = t_cfg.get("optimizer", "adam").lower()

        if opt == "adamw":
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # ---- LR scheduler --------------------------------------------------
        num_epochs    = int(t_cfg.get("num_epochs", 100))
        warmup_epochs = int(t_cfg.get("warmup_epochs", 5))

        def _lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            p = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            return max(0.05, 0.5 * (1.0 + np.cos(np.pi * p)))

        self.scheduler   = optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        self.num_epochs  = num_epochs
        self.patience    = int(t_cfg.get("patience",  15))
        self.min_delta   = float(t_cfg.get("min_delta", 1e-3))

        # ---- Checkpoint dir ------------------------------------------------
        self.ckpt_dir = Path(o_cfg.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Single-epoch training
    # -----------------------------------------------------------------------

    def _train_epoch(
        self,
        nasa_loader:      DataLoader,
        mendeley_loader:  Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        One training epoch.

        NASA batches → SOH objective.
        Mendeley batches → degradation objective (if loader provided).
        """
        self.model.train()
        total_soh_loss = 0.0
        total_deg_loss = 0.0
        n_nasa     = 0
        n_mendeley = 0

        # ---- NASA batches (Task A) -----------------------------------------
        for batch in nasa_loader:
            batch = _to_device(batch, self.device)
            self.optimizer.zero_grad()

            out = self.model(
                batch["discharge"], batch["eis"], batch["physics"]
            )
            soh_loss = self.soh_criterion(
                out["soh_pred"],
                batch["soh_label"],
                cycle_age=batch.get("cycle_idx"),
                battery_ids=batch.get("battery_id"),
            )
            loss = self.lambda_soh * soh_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_soh_loss += soh_loss.item()
            n_nasa += 1

        # ---- Mendeley batches (Task B, optional) ---------------------------
        if mendeley_loader is not None and self.training_mode in ("joint", "staged"):
            for batch in mendeley_loader:
                batch = _to_device(batch, self.device)
                self.optimizer.zero_grad()

                out = self.model.forward_eis_only(batch["eis"])
                deg_loss = self.deg_criterion(
                    lli_pred=out["lli_pred"],
                    lam_pred=out["lam_pred"],
                    cl_pred=out["cl_pred"],
                    lli_target=batch["lli_label"],
                    lam_target=batch["lam_label"],
                    cl_target=batch["cl_label"],
                )
                loss = self.lambda_deg * deg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_deg_loss += deg_loss.item()
                n_mendeley += 1

        return {
            "soh_loss": total_soh_loss / max(n_nasa, 1),
            "deg_loss": total_deg_loss / max(n_mendeley, 1),
        }

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_soh(
        self, loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, self.device)
                out = self.model(batch["discharge"], batch["eis"], batch["physics"])
                loss = self.soh_criterion(
                    out["soh_pred"], batch["soh_label"],
                    cycle_age=batch.get("cycle_idx"),
                )
                total_loss += loss.item()
                all_preds.append(out["soh_pred"].view(-1).cpu().numpy())
                all_targets.append(batch["soh_label"].view(-1).cpu().numpy())

        metrics = _soh_metrics(
            np.concatenate(all_preds), np.concatenate(all_targets)
        )
        return total_loss / max(len(loader), 1), metrics

    def _validate_deg(
        self, loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        lli_p, lli_t, lam_p, lam_t, cl_p, cl_t = [], [], [], [], [], []

        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, self.device)
                out = self.model.forward_eis_only(batch["eis"])
                loss = self.deg_criterion(
                    lli_pred=out["lli_pred"],  lam_pred=out["lam_pred"],
                    cl_pred=out["cl_pred"],
                    lli_target=batch["lli_label"], lam_target=batch["lam_label"],
                    cl_target=batch["cl_label"],
                )
                total_loss += loss.item()
                lli_p.append(out["lli_pred"].view(-1).cpu().numpy())
                lli_t.append(batch["lli_label"].view(-1).cpu().numpy())
                lam_p.append(out["lam_pred"].view(-1).cpu().numpy())
                lam_t.append(batch["lam_label"].view(-1).cpu().numpy())
                cl_p.append(out["cl_pred"].view(-1).cpu().numpy())
                cl_t.append(batch["cl_label"].view(-1).cpu().numpy())

        metrics = _deg_metrics(
            np.concatenate(lli_p), np.concatenate(lli_t),
            np.concatenate(lam_p), np.concatenate(lam_t),
            np.concatenate(cl_p),  np.concatenate(cl_t),
        )
        return total_loss / max(len(loader), 1), metrics

    # -----------------------------------------------------------------------
    # Main fit loop
    # -----------------------------------------------------------------------

    def fit(
        self,
        nasa_train_loader:      DataLoader,
        nasa_val_loader:        DataLoader,
        mendeley_train_loader:  Optional[DataLoader] = None,
        mendeley_val_loader:    Optional[DataLoader] = None,
    ) -> Dict:
        """
        Full training loop.

        Returns:
            History dict with per-epoch losses and metrics.
        """
        best_val_mae  = float("inf")
        patience_ctr  = 0
        start_time    = time.time()

        history: Dict[str, List] = {
            "train_soh_loss": [],
            "train_deg_loss": [],
            "val_soh_loss": [],
            "val_mae": [], "val_rmse": [], "val_r2": [],
            "val_deg_loss": [],
        }

        logger.info(
            f"MultiTaskTrainer.fit(): mode={self.training_mode}, "
            f"epochs={self.num_epochs}, λ_soh={self.lambda_soh}, "
            f"λ_deg={self.lambda_deg}"
        )

        for epoch in range(1, self.num_epochs + 1):
            # --- train epoch ------------------------------------------------
            t_losses = self._train_epoch(nasa_train_loader, mendeley_train_loader)

            # --- validate ---------------------------------------------------
            v_soh_loss, v_soh_met = self._validate_soh(nasa_val_loader)
            history["train_soh_loss"].append(t_losses["soh_loss"])
            history["train_deg_loss"].append(t_losses["deg_loss"])
            history["val_soh_loss"].append(v_soh_loss)
            history["val_mae"].append(v_soh_met["mae"])
            history["val_rmse"].append(v_soh_met["rmse"])
            history["val_r2"].append(v_soh_met["r2"])

            deg_line = ""
            if mendeley_val_loader is not None and self.training_mode in ("joint", "staged"):
                v_deg_loss, v_deg_met = self._validate_deg(mendeley_val_loader)
                history["val_deg_loss"].append(v_deg_loss)
                deg_line = (
                    f"  deg_val={v_deg_loss:.4f} "
                    f"MAE(LLI={v_deg_met['mae_lli']:.3f} "
                    f"LAM={v_deg_met['mae_lam']:.3f} "
                    f"CL={v_deg_met['mae_cl']:.3f})"
                )

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:3d}/{self.num_epochs}  "
                f"soh_train={t_losses['soh_loss']:.4f}  "
                f"soh_val={v_soh_loss:.4f}  "
                f"MAE={v_soh_met['mae']:.3f}  RMSE={v_soh_met['rmse']:.3f}  "
                f"R²={v_soh_met['r2']:.4f}{deg_line}"
            )

            # --- checkpoint + early stopping (driven by SOH MAE) -----------
            if v_soh_met["mae"] < best_val_mae - self.min_delta:
                best_val_mae = v_soh_met["mae"]
                patience_ctr = 0
                self._save_best(epoch, v_soh_met)
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        elapsed = time.time() - start_time
        history["elapsed_sec"] = elapsed
        history["best_val_mae"] = best_val_mae
        logger.info(
            f"Training complete in {elapsed:.0f}s  "
            f"best_val_MAE={best_val_mae:.3f}"
        )

        # Load best weights
        best_ckpt = self.ckpt_dir / "best_model.pth"
        if best_ckpt.exists():
            state = torch.load(
                str(best_ckpt), map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(state["model_state_dict"])
            logger.info("Loaded best checkpoint weights")

        return history

    # -----------------------------------------------------------------------

    def _save_best(self, epoch: int, metrics: Dict):
        path = self.ckpt_dir / "best_model.pth"
        torch.save(
            {
                "epoch":            epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state":  self.optimizer.state_dict(),
                "val_metrics":      metrics,
                "training_mode":    self.training_mode,
            },
            str(path),
        )
        logger.info(
            f"  ✓ Checkpoint saved (epoch {epoch}, "
            f"MAE={metrics.get('mae', '?'):.3f}) → {path}"
        )

    def save_checkpoint(self, path: str, epoch: int = 0, metrics: Dict = None):
        torch.save(
            {
                "epoch":            epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state":  self.optimizer.state_dict(),
                "val_metrics":      metrics or {},
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}")

    def predict_soh(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        """Run inference on a NASA DataLoader, return predictions + targets."""
        self.model.eval()
        preds, targets, bids = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, self.device)
                out = self.model(batch["discharge"], batch["eis"], batch["physics"])
                preds.append(out["soh_pred"].view(-1).cpu().numpy())
                targets.append(batch["soh_label"].view(-1).cpu().numpy())
                bids.extend(
                    batch["battery_id"]
                    if isinstance(batch["battery_id"], list)
                    else [str(batch["battery_id"])] * len(out["soh_pred"])
                )
        return {
            "predictions": np.concatenate(preds),
            "targets":     np.concatenate(targets),
            "battery_ids": bids,
        }

    def predict_deg(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        """Run degradation inference on a Mendeley DataLoader."""
        self.model.eval()
        lli_p, lam_p, cl_p = [], [], []
        lli_t, lam_t, cl_t = [], [], []
        cell_ids, cycles = [], []

        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, self.device)
                out = self.model.forward_eis_only(batch["eis"])
                lli_p.append(out["lli_pred"].view(-1).cpu().numpy())
                lam_p.append(out["lam_pred"].view(-1).cpu().numpy())
                cl_p.append(out["cl_pred"].view(-1).cpu().numpy())
                lli_t.append(batch["lli_label"].view(-1).cpu().numpy())
                lam_t.append(batch["lam_label"].view(-1).cpu().numpy())
                cl_t.append(batch["cl_label"].view(-1).cpu().numpy())
                cell_ids.extend(
                    batch["cell_id"]
                    if isinstance(batch["cell_id"], list)
                    else [int(batch["cell_id"])] * len(out["lli_pred"])
                )
                cycles.append(batch["aging_cycle"].cpu().numpy())

        return {
            "lli_pred":    np.concatenate(lli_p),
            "lam_pred":    np.concatenate(lam_p),
            "cl_pred":     np.concatenate(cl_p),
            "lli_target":  np.concatenate(lli_t),
            "lam_target":  np.concatenate(lam_t),
            "cl_target":   np.concatenate(cl_t),
            "cell_ids":    cell_ids,
            "aging_cycles": np.concatenate(cycles),
        }
