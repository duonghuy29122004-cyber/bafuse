"""
Unified BaFuse model: Encoders + Fusion + SoH Prediction Head.

Complete pipeline:
1. Encode discharge, EIS, physics independently
2. Fuse via cross-attention or weighted combination
3. Predict SoH (scalar 0-100 or percentile)
4. Return predictions + attention weights for interpretability
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .encoders import DischargeEncoder, EISEncoder, PhysicsEncoder
from .fusion import CrossAttentionFusion


class BaFuse(nn.Module):
    """
    Multimodal BaFuse model for EV battery SoH estimation.
    """
    
    def __init__(
        self,
        discharge_input_size: int,
        eis_num_frequencies: int,
        physics_num_features: int,
        latent_dim: int = 64,
        fusion_method: str = "cross_attention",
        fusion_dim: int = 128
    ):
        """
        Initialize BaFuse model.
        
        Args:
            discharge_input_size: Number of discharge features
            eis_num_frequencies: Number of EIS frequency points
            physics_num_features: Number of physics-derived features
            latent_dim: Latent dimension for each encoder
            fusion_method: "cross_attention", "weighted", or "concat"
            fusion_dim: Fused feature dimension
        """
        super().__init__()
        
        # Encoders for each modality
        self.discharge_encoder = DischargeEncoder(discharge_input_size, latent_dim=latent_dim)
        self.eis_encoder = EISEncoder(eis_num_frequencies, latent_dim=latent_dim)
        self.physics_encoder = PhysicsEncoder(physics_num_features, latent_dim=latent_dim)
        
        # Fusion mechanism
        if fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(latent_dim, fusion_dim=fusion_dim)
        else:
            # TODO: Support weighted and concat fusion methods
            self.fusion = CrossAttentionFusion(latent_dim, fusion_dim=fusion_dim)
        
        # SoH prediction head
        self.soh_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # SoH prediction (0-100)
        )
    
    def forward(
        self,
        discharge: torch.Tensor,
        eis: torch.Tensor,
        physics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BaFuse.
        
        Args:
            discharge: (batch_size, seq_len, discharge_features)
            eis: (batch_size, eis_frequencies, 2)  [real, imag]
            physics: (batch_size, physics_features)
        
        Returns:
            dict with:
                - soh_pred: (batch_size, 1) predicted SoH [0, 100]
                - discharge_latent: latent representation
                - eis_latent: latent representation
                - physics_latent: latent representation
                - fused: fused multimodal representation
                - attention_weights: (optional) attention matrices for interpretability
        """
        # Encode each modality
        discharge_latent = self.discharge_encoder(discharge)
        eis_latent = self.eis_encoder(eis)
        physics_latent = self.physics_encoder(physics)
        
        # Fuse representations
        fused, fusion_info = self.fusion(discharge_latent, eis_latent, physics_latent)
        
        # Predict SoH
        soh_pred = self.soh_head(fused)
        
        return {
            "soh_pred": soh_pred,
            "discharge_latent": discharge_latent,
            "eis_latent": eis_latent,
            "physics_latent": physics_latent,
            "fused": fused,
            **fusion_info  # Include attention weights if available
        }
