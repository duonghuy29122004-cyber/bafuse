"""
Loss functions including physics-informed regularization.

Strategies:
- MSE loss for SoH prediction
- Physics-informed regularization (e.g., monotonic SoH, smooth degradation)
- Ablation loss to encourage each modality to contribute unique information
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class SoHPredictionLoss(nn.Module):
    """
    SoH prediction loss with optional physics-informed regularization.
    """

    def __init__(
        self,
        base_loss: str = "mse",
        lambda_physics: float = 0.1,
        lambda_smooth: float = 0.05,
    ):
        """
        Args:
            base_loss: "mse" or "mae" for regression
            lambda_physics: Weight for physics constraints
            lambda_smooth: Weight for smoothness (monotonic degradation)
        """
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_smooth = lambda_smooth

        if base_loss == "mse":
            self.base_loss_fn = nn.MSELoss()
        else:
            self.base_loss_fn = nn.L1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cycle_age: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss with physics constraints.

        Args:
            pred: Predicted SoH (batch_size, 1)
            target: Ground truth SoH (batch_size, 1)
            cycle_age: (optional) Cycle age per sample, for smoothness constraint

        Returns:
            Scalar loss value
        """
        # Ensure shapes match
        pred = pred.view(-1)
        target = target.view(-1)

        # Base regression loss
        loss = self.base_loss_fn(pred, target)

        # Physics-informed bound constraint
        if self.lambda_physics > 0:
            physics_loss = self._physics_regularization(pred)
            loss = loss + self.lambda_physics * physics_loss

        # Smoothness: SoH should degrade monotonically with cycle age
        if cycle_age is not None and self.lambda_smooth > 0:
            smooth_loss = self._smoothness_loss(pred, cycle_age, target)
            loss = loss + self.lambda_smooth * smooth_loss

        return loss

    def _physics_regularization(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Penalise predictions outside physically valid range [0, 100].
        Uses a soft ReLU (hinge) penalty.
        """
        # Penalty for pred < 0
        lower_violation = torch.relu(-pred)          # positive where pred < 0
        # Penalty for pred > 100
        upper_violation = torch.relu(pred - 100.0)   # positive where pred > 100
        return (lower_violation + upper_violation).mean()

    def _smoothness_loss(
        self,
        pred: torch.Tensor,
        cycle_age: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage monotonically decreasing SoH with increasing cycle age.

        For pairs (i, j) where cycle_age[i] < cycle_age[j], penalise
        pred[i] < pred[j] (i.e., younger battery predicted lower SoH).
        Uses vectorised pairwise hinge for efficiency.
        """
        cycle_age = cycle_age.view(-1).float()
        B = pred.size(0)
        if B < 2:
            return pred.new_zeros(1).squeeze()

        # Pairwise differences
        age_diff = cycle_age.unsqueeze(1) - cycle_age.unsqueeze(0)   # (B, B)
        soh_diff = pred.unsqueeze(1) - pred.unsqueeze(0)             # (B, B)

        # Where age_diff > 0: older battery → lower SoH expected → soh_diff should be < 0
        # Penalise positive soh_diff when age_diff > 0
        mask = (age_diff > 0).float()
        violation = torch.relu(soh_diff) * mask   # (B, B)
        return violation.mean()


class ModalityContributionLoss(nn.Module):
    """
    Encourage each modality to contribute unique information to SoH.

    Prevents one modality from dominating while others are ignored.
    """

    def __init__(self, lambda_diversity: float = 0.01):
        """
        Args:
            lambda_diversity: Weight for diversity term
        """
        super().__init__()
        self.lambda_diversity = lambda_diversity

    def forward(
        self,
        discharge_latent: torch.Tensor,
        eis_latent: torch.Tensor,
        physics_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diversity loss encouraging low correlation between modalities.

        Penalises high cosine similarity between latent representations.

        Args:
            discharge_latent: (batch_size, latent_dim)
            eis_latent:       (batch_size, latent_dim)
            physics_latent:   (batch_size, latent_dim)

        Returns:
            Scalar loss
        """
        def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a_n = nn.functional.normalize(a, dim=-1)
            b_n = nn.functional.normalize(b, dim=-1)
            return (a_n * b_n).sum(dim=-1).mean()

        import torch.nn.functional as F  # noqa: F401 – already imported above but kept local

        sim_de = _cosine_sim(discharge_latent, eis_latent)
        sim_dp = _cosine_sim(discharge_latent, physics_latent)
        sim_ep = _cosine_sim(eis_latent, physics_latent)

        # Penalise high similarity (want diversity)
        diversity_loss = (sim_de.abs() + sim_dp.abs() + sim_ep.abs()) / 3.0
        return self.lambda_diversity * diversity_loss
