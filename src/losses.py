"""
Loss functions including physics-informed regularization.

Task 2 fix: _smoothness_loss now compares ONLY within the same battery_id.
Cross-battery comparison was a logic bug — different batteries age at different
rates, so 'older cycle => lower SoH' is only valid within one battery's trajectory.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional


class SoHPredictionLoss(nn.Module):
    """SoH prediction loss with physics-informed regularization."""

    def __init__(
        self,
        base_loss: str = "mse",
        lambda_physics: float = 0.1,
        lambda_smooth: float = 0.05,
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_smooth  = lambda_smooth
        self.base_loss_fn   = nn.MSELoss() if base_loss == "mse" else nn.L1Loss()

    def forward(
        self,
        pred:        torch.Tensor,
        target:      torch.Tensor,
        cycle_age:   Optional[torch.Tensor] = None,
        battery_ids: Optional[List[str]]    = None,  # Task 2: for within-battery monotonicity
    ) -> torch.Tensor:
        pred   = pred.view(-1)
        target = target.view(-1)

        loss = self.base_loss_fn(pred, target)

        if self.lambda_physics > 0:
            loss = loss + self.lambda_physics * self._physics_regularization(pred)

        if cycle_age is not None and self.lambda_smooth > 0:
            loss = loss + self.lambda_smooth * self._smoothness_loss(pred, cycle_age, battery_ids)

        return loss

    def _physics_regularization(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalise predictions outside [0, 100]."""
        return (torch.relu(-pred) + torch.relu(pred - 100.0)).mean()

    def _smoothness_loss(
        self,
        pred:        torch.Tensor,
        cycle_age:   torch.Tensor,
        battery_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Task 2 FIX: within-battery monotonicity only.

        For each battery separately, for pairs (i, j) where cycle_age[i] < cycle_age[j],
        penalise pred[i] < pred[j] (younger cycle predicted lower SoH than older cycle).

        If battery_ids is None, falls back to cross-battery mode (old behaviour).
        """
        cycle_age = cycle_age.view(-1).float()
        B = pred.size(0)
        if B < 2:
            return pred.new_zeros(1).squeeze()

        if battery_ids is None:
            # Fallback: original cross-battery behaviour
            age_diff = cycle_age.unsqueeze(1) - cycle_age.unsqueeze(0)
            soh_diff = pred.unsqueeze(1)      - pred.unsqueeze(0)
            mask     = (age_diff > 0).float()
            return torch.relu(soh_diff * mask).mean()

        # Within-battery-only
        bid_to_idx: Dict[str, List[int]] = defaultdict(list)
        for i, bid in enumerate(battery_ids):
            bid_to_idx[str(bid)].append(i)

        total = pred.new_zeros(1)
        n_pairs = 0

        for indices in bid_to_idx.values():
            if len(indices) < 2:
                continue
            idx_t = torch.tensor(indices, dtype=torch.long, device=pred.device)
            ages  = cycle_age[idx_t]
            ps    = pred[idx_t]

            age_diff = ages.unsqueeze(1) - ages.unsqueeze(0)   # (k, k)
            soh_diff = ps.unsqueeze(1)   - ps.unsqueeze(0)     # (k, k)
            mask     = (age_diff > 0).float()

            total   = total + torch.relu(soh_diff * mask).sum()
            n_pairs += int(mask.sum().item())

        if n_pairs == 0:
            return pred.new_zeros(1).squeeze()
        return (total / n_pairs).squeeze()


class ModalityContributionLoss(nn.Module):
    """Encourage modalities to contribute diverse (non-redundant) information."""

    def __init__(self, lambda_diversity: float = 0.01):
        super().__init__()
        self.lambda_diversity = lambda_diversity

    def forward(
        self,
        discharge_latent: torch.Tensor,
        eis_latent:       torch.Tensor,
        physics_latent:   torch.Tensor,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        def _cos(a, b):
            return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1).mean()

        sim = (_cos(discharge_latent, eis_latent).abs()
               + _cos(discharge_latent, physics_latent).abs()
               + _cos(eis_latent, physics_latent).abs()) / 3.0
        return self.lambda_diversity * sim


# ═══════════════════════════════════════════════════════════════════════════════
# BaFuse v2 — Degradation-mode loss
# ═══════════════════════════════════════════════════════════════════════════════

class DegradationLoss(nn.Module):
    """
    Weighted multi-output regression loss for model-derived degradation-mode labels.

    Targets: LLI (Loss of Lithium Inventory),
             LAM (Loss of Active Material),
             CL  (Conductivity Loss).

    IMPORTANT: Labels are model-derived from EIS/ECM fitting (Mendeley dataset).
    They are NOT absolute physical ground truth.

    L_deg = w_lli * L(lli_pred, lli_target)
          + w_lam * L(lam_pred, lam_target)
          + w_cl  * L(cl_pred,  cl_target)

    Args:
        base_loss : "mse" or "mae".
        w_lli     : Weight for LLI loss component.
        w_lam     : Weight for LAM loss component.
        w_cl      : Weight for CL  loss component.
    """

    def __init__(
        self,
        base_loss: str   = "mse",
        w_lli:     float = 1.0,
        w_lam:     float = 1.0,
        w_cl:      float = 1.0,
    ):
        super().__init__()
        self.w_lli = w_lli
        self.w_lam = w_lam
        self.w_cl  = w_cl
        self.loss_fn = nn.MSELoss() if base_loss == "mse" else nn.L1Loss()

    def forward(
        self,
        lli_pred:   torch.Tensor,
        lam_pred:   torch.Tensor,
        cl_pred:    torch.Tensor,
        lli_target: torch.Tensor,
        lam_target: torch.Tensor,
        cl_target:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            *_pred   : (B, 1) or (B,)
            *_target : (B,)

        Returns:
            Weighted scalar loss.
        """
        l_lli = self.loss_fn(lli_pred.view(-1), lli_target.view(-1))
        l_lam = self.loss_fn(lam_pred.view(-1), lam_target.view(-1))
        l_cl  = self.loss_fn(cl_pred.view(-1),  cl_target.view(-1))

        return self.w_lli * l_lli + self.w_lam * l_lam + self.w_cl * l_cl

    def component_losses(
        self,
        lli_pred:   torch.Tensor,
        lam_pred:   torch.Tensor,
        cl_pred:    torch.Tensor,
        lli_target: torch.Tensor,
        lam_target: torch.Tensor,
        cl_target:  torch.Tensor,
    ) -> Dict[str, float]:
        """Return un-weighted individual component losses (for logging)."""
        with torch.no_grad():
            return {
                "loss_lli": float(self.loss_fn(lli_pred.view(-1), lli_target.view(-1))),
                "loss_lam": float(self.loss_fn(lam_pred.view(-1), lam_target.view(-1))),
                "loss_cl":  float(self.loss_fn(cl_pred.view(-1),  cl_target.view(-1))),
            }


class MultiTaskLoss(nn.Module):
    """
    Combined SOH + Degradation loss for multi-task training.

    L_total = lambda_soh * L_soh + lambda_deg * L_deg

    Where:
        L_soh — SoHPredictionLoss (NASA batches)
        L_deg — DegradationLoss   (Mendeley batches)

    The two losses are kept separate in the forward signature so that
    dataset-specific batches can contribute only their own objective.

    Args:
        lambda_soh : Weight for SOH loss.
        lambda_deg : Weight for degradation loss.
        soh_kwargs : Dict of kwargs forwarded to SoHPredictionLoss.
        deg_kwargs : Dict of kwargs forwarded to DegradationLoss.
    """

    def __init__(
        self,
        lambda_soh: float = 1.0,
        lambda_deg: float = 0.5,
        soh_kwargs: Optional[Dict] = None,
        deg_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.lambda_soh = lambda_soh
        self.lambda_deg = lambda_deg
        self.soh_loss = SoHPredictionLoss(**(soh_kwargs or {}))
        self.deg_loss = DegradationLoss(**(deg_kwargs or {}))

    def forward_soh(
        self,
        pred:        torch.Tensor,
        target:      torch.Tensor,
        cycle_age:   Optional[torch.Tensor] = None,
        battery_ids: Optional[List[str]]    = None,
    ) -> torch.Tensor:
        """SOH loss only (use for NASA batches)."""
        return self.lambda_soh * self.soh_loss(
            pred, target, cycle_age=cycle_age, battery_ids=battery_ids
        )

    def forward_deg(
        self,
        lli_pred:   torch.Tensor,
        lam_pred:   torch.Tensor,
        cl_pred:    torch.Tensor,
        lli_target: torch.Tensor,
        lam_target: torch.Tensor,
        cl_target:  torch.Tensor,
    ) -> torch.Tensor:
        """Degradation loss only (use for Mendeley batches)."""
        return self.lambda_deg * self.deg_loss(
            lli_pred, lam_pred, cl_pred, lli_target, lam_target, cl_target
        )
