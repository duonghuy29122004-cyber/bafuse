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
        lambda_smooth: float = 0.05
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
            self.base_loss = nn.MSELoss()
        else:
            self.base_loss = nn.L1Loss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cycle_age: Optional[torch.Tensor] = None
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
        # Base regression loss
        loss = self.base_loss(pred, target)
        
        # Physics-informed constraints
        if self.lambda_physics > 0:
            # SoH should be bounded [0, 100]
            physics_loss = self._physics_regularization(pred)
            loss = loss + self.lambda_physics * physics_loss
        
        # Smoothness: SoH should degrade monotonically
        if cycle_age is not None and self.lambda_smooth > 0:
            smooth_loss = self._smoothness_loss(pred, cycle_age, target)
            loss = loss + self.lambda_smooth * smooth_loss
        
        return loss
    
    def _physics_regularization(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Penalize out-of-bounds predictions.
        """
        # TODO: Implement penalty for SoH < 0 or > 100
        pass
    
    def _smoothness_loss(
        self,
        pred: torch.Tensor,
        cycle_age: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage smooth, monotonic degradation.
        """
        # TODO: Implement smoothness constraint based on cycle age
        pass


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
        physics_latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diversity loss encouraging low correlation between modalities.
        
        Args:
            discharge_latent: (batch_size, latent_dim)
            eis_latent: (batch_size, latent_dim)
            physics_latent: (batch_size, latent_dim)
        
        Returns:
            Scalar loss
        """
        # TODO: Compute correlation between latent representations
        # Loss should penalize high correlation (encourage diversity)
        pass
