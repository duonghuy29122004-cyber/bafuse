"""
Evaluation of BaFuse model on test set.

Computes:
- Prediction error (MAE, RMSE, R2)
- Per-battery errors
- Contribution of each modality to final prediction
- Uncertainty quantification
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from pathlib import Path

from models.bafuse import BaFuse


def evaluate(
    model: BaFuse,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Computes:
    - MAE, RMSE, R2
    - Per-battery statistics
    
    Args:
        model: Trained BaFuse model
        test_loader: Test DataLoader
        device: torch device
    
    Returns:
        metrics_dict with performance metrics
    """
    # TODO: Implement evaluation
    pass


def analyze_modality_contribution(
    model: BaFuse,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Analyze each modality's contribution to SoH prediction.
    
    Methods:
    - Attention weights from fusion layer
    - Ablation: compare full model vs. model without one modality
    - Feature importance: gradient-based or SHAP
    
    Returns:
        contribution_dict with modality contributions
    """
    # TODO: Implement modality contribution analysis
    pass


def compute_uncertainty(
    model: BaFuse,
    test_loader: DataLoader,
    device: torch.device,
    num_mc_samples: int = 30
) -> Dict[str, np.ndarray]:
    """
    Estimate prediction uncertainty using MC Dropout.
    
    Args:
        model: Trained BaFuse model
        test_loader: Test DataLoader
        device: torch device
        num_mc_samples: Number of stochastic forward passes
    
    Returns:
        uncertainty_dict with mean and std predictions
    """
    # TODO: Implement uncertainty quantification
    pass
