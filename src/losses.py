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
