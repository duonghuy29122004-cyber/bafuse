"""
Stage-1 Degradation Trainer — EIS encoder + DegradationHead pre-training.

Uses the Mendeley dataset exclusively.

Training objective:
    EIS features → EIS encoder → DegradationHead (EIS-only path)
        → LLI / LAM / CL prediction
        → DegradationLoss

This stage fine-tunes the EIS encoder to produce representations that are
predictive of model-derived degradation-mode estimates. The learned EIS encoder
weights are then transferred into BaFuseV2 for Stage-2/3 training.

IMPORTANT: Labels (LLI, LAM, CL) are model-derived from ECM fitting — they are
NOT absolute physical ground truth.

Usage:
    trainer = DegradationTrainer(model, config)
    history = trainer.fit(train_loader, val_loader)
    trainer.save_checkpoint("experiments/degradation/best.pth")
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
from ..losses import DegradationLoss

logger = logging.getLogger(__name__)


def _batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _deg_metrics(
    lli_p: np.ndarray, lli_t: np.ndarray,
    lam_p: np.ndarray, lam_t: np.ndarray,
    cl_p:  np.ndarray, cl_t:  np.ndarray,
) -> Dict[str, float]:
    """MAE per degradation mode plus mean."""
    mae_lli = float(np.mean(np.abs(lli_p - lli_t)))
    mae_lam = float(np.mean(np.abs(lam_p - lam_t)))
    mae_cl  = float(np.mean(np.abs(cl_p  - cl_t)))
    return {
        "mae_lli": mae_lli,
        "mae_lam": mae_lam,
        "mae_cl":  mae_cl,
        "mae_mean": (mae_lli + mae_lam + mae_cl) / 3.0,
    }


class DegradationTrainer:
    """
    Trainer for Stage-1 degradation-mode pre-training on Mendeley data.

    Only the EIS encoder and degradation_head_eis_only are trained.
    All other BaFuseV2 parameters are frozen.

    Args:
        model        : BaFuseV2 instance.
        deg_config   : Dict with keys:
            lr, weight_decay, num_epochs, patience, min_delta,
            w_lli, w_lam, w_cl, base_loss, checkpoint_dir.
        device       : Torch device string.
    """

    def __init__(
        self,
        model:       BaFuseV2,
        deg_config:  Dict,
        device:      str = "cpu",
    ):
        self.model     = model
        self.cfg       = deg_config
        self.device    = torch.device(device)
        self.model.to(self.device)

        # ---- Freeze everything except EIS encoder + deg head (EIS-only) ----
        self._freeze_non_deg_params()

        # ---- Loss ----------------------------------------------------------
        self.criterion = DegradationLoss(
            base_loss=deg_config.get("base_loss", "mse"),
            w_lli=deg_config.get("w_lli", 1.0),
            w_lam=deg_config.get("w_lam", 1.0),
            w_cl=deg_config.get("w_cl",  1.0),
        )

        # ---- Optimiser (only unfrozen params) ------------------------------
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable,
            lr=float(deg_config.get("lr", 1e-3)),
            weight_decay=float(deg_config.get("weight_decay", 1e-5)),
        )

        # ---- LR scheduler --------------------------------------------------
        num_epochs = int(deg_config.get("num_epochs", 50))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-5
        )

        self.ckpt_dir = Path(deg_config.get("checkpoint_dir", "experiments/degradation"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------

    def _freeze_non_deg_params(self):
        """Freeze everything except EIS encoder and EIS-only degradation head."""
        frozen, trainable = 0, 0
        for name, param in self.model.named_parameters():
            if "eis_encoder" in name or "degradation_head_eis_only" in name:
                param.requires_grad = True
                trainable += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()
        logger.info(
            f"Stage-1 freeze: {trainable:,} trainable params "
            f"(EIS encoder + deg head), {frozen:,} frozen"
        )

    def unfreeze_all(self):
        """Unfreeze all parameters (call before Stage-2/3 transfer)."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")

    # -----------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            batch = _batch_to_device(batch, self.device)
            eis = batch["eis"]

            self.optimizer.zero_grad()
            out = self.model.forward_eis_only(eis)

            loss = self.criterion(
                lli_pred=out["lli_pred"],
                lam_pred=out["lam_pred"],
                cl_pred=out["cl_pred"],
                lli_target=batch["lli_label"],
                lam_target=batch["lam_label"],
                cl_target=batch["cl_label"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def _validate(self, loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        lli_p, lli_t = [], []
        lam_p, lam_t = [], []
        cl_p,  cl_t  = [], []

        with torch.no_grad():
            for batch in loader:
                batch = _batch_to_device(batch, self.device)
                eis = batch["eis"]
                out = self.model.forward_eis_only(eis)

                loss = self.criterion(
                    lli_pred=out["lli_pred"],
                    lam_pred=out["lam_pred"],
                    cl_pred=out["cl_pred"],
                    lli_target=batch["lli_label"],
                    lam_target=batch["lam_label"],
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

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict:
        """
        Run Stage-1 pre-training.

        Returns:
            Training history dict.
        """
        num_epochs   = int(self.cfg.get("num_epochs", 50))
        patience     = int(self.cfg.get("patience", 10))
        min_delta    = float(self.cfg.get("min_delta", 1e-4))

        best_val_loss  = float("inf")
        patience_ctr   = 0
        start_time     = time.time()

        history: Dict[str, List] = {
            "train_loss": [], "val_loss": [],
            "val_mae_lli": [], "val_mae_lam": [], "val_mae_cl": [], "val_mae_mean": [],
        }

        logger.info(f"Stage-1 degradation pre-training: {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            t_loss = self._train_epoch(train_loader)
            v_loss, v_met = self._validate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["val_mae_lli"].append(v_met["mae_lli"])
            history["val_mae_lam"].append(v_met["mae_lam"])
            history["val_mae_cl"].append(v_met["mae_cl"])
            history["val_mae_mean"].append(v_met["mae_mean"])

            logger.info(
                f"Epoch {epoch:3d}/{num_epochs}  "
                f"train={t_loss:.4f}  val={v_loss:.4f}  "
                f"MAE(LLI={v_met['mae_lli']:.3f} LAM={v_met['mae_lam']:.3f} "
                f"CL={v_met['mae_cl']:.3f})"
            )

            if v_loss < best_val_loss - min_delta:
                best_val_loss = v_loss
                patience_ctr  = 0
                self.save_checkpoint(self.ckpt_dir / "best_degradation.pth", epoch, v_met)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        elapsed = time.time() - start_time
        history["elapsed_sec"] = elapsed
        history["best_val_loss"] = best_val_loss
        logger.info(
            f"Stage-1 complete in {elapsed:.0f}s. "
            f"Best val loss={best_val_loss:.4f}"
        )
        return history

    # -----------------------------------------------------------------------

    def save_checkpoint(
        self, path: Path, epoch: int = 0, metrics: Dict = None
    ):
        torch.save(
            {
                "epoch":             epoch,
                "model_state_dict":  self.model.state_dict(),
                "optimizer_state":   self.optimizer.state_dict(),
                "metrics":           metrics or {},
                "stage":             "degradation_pretrain",
            },
            str(path),
        )
        logger.info(f"Degradation checkpoint saved → {path}")

    def load_best_checkpoint(self, path: Optional[str] = None):
        ckpt_path = path or str(self.ckpt_dir / "best_degradation.pth")
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded Stage-1 checkpoint from {ckpt_path}")
