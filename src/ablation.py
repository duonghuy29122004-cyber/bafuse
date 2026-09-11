"""
Ablation studies: assess contribution of each modality.

Approaches:
1. Unimodal models: train only with discharge / only EIS / only physics
2. Pairwise models: discharge+EIS, discharge+physics, EIS+physics
3. Full model: all three
4. Compare performance to quantify synergy

Results:
- Individual modality performance
- Pairwise performance
- Full model performance
- Synergy metrics (full > sum of pairs)
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from pathlib import Path

from models.bafuse import BaFuse
from losses import SoHPredictionLoss


class UnimodalModel(torch.nn.Module):
    """
    Single modality model for ablation studies.
    
    Trains only discharge, only EIS, or only physics signal.
    """
    
    def __init__(self, modality: str, latent_dim: int = 64):
        """
        Args:
            modality: "discharge", "eis", or "physics"
            latent_dim: Latent dimension
        """
        super().__init__()
        self.modality = modality
        # TODO: Implement single-modality encoder + prediction head
        pass


def run_ablation_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device = "cuda"
) -> Dict[str, Dict]:
    """
    Run full ablation study.
    
    Train and evaluate:
    1. Unimodal models (discharge, EIS, physics)
    2. Pairwise models (D+E, D+P, E+P)
    3. Full model (D+E+P)
    
    Args:
        train_loader, val_loader, test_loader: DataLoaders
        device: torch device
    
    Returns:
        ablation_results: dict with performance of each configuration
    """
    # TODO: Implement full ablation study
    pass


def compute_synergy_metrics(ablation_results: Dict) -> Dict[str, float]:
    """
    Compute synergy between modalities.
    
    Synergy = Performance(full) - max(Performance(pairwise))
    
    Indicates how much additional benefit comes from combining all three.
    """
    # TODO: Compute synergy metrics
    pass
