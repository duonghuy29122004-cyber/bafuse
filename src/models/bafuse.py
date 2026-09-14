"""
Unified BaFuse model: Encoders + Fusion + SoH Prediction Head.

Complete pipeline:
1. Encode discharge, EIS, physics independently
2. Fuse via cross-attention or weighted combination
3. Predict SoH (scalar 0-100)
4. Return predictions + attention weights for interpretability
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .encoders import DischargeEncoder, EISEncoder, PhysicsEncoder
from .fusion import CrossAttentionFusion, WeightedFusion, ConcatFusion


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
        fusion_dim: int = 128,
    ):
        """
        Args:
            discharge_input_size: Number of discharge features per time-step
            eis_num_frequencies:  Number of EIS features (or frequency points)
            physics_num_features: Number of physics-derived features
            latent_dim:           Latent dimension for each encoder output
            fusion_method:        "cross_attention" | "weighted" | "concat"
            fusion_dim:           Fused feature dimension
        """
        super().__init__()

        # --- Encoders ---
        self.discharge_encoder = DischargeEncoder(
            input_size=discharge_input_size, latent_dim=latent_dim
        )
        self.eis_encoder = EISEncoder(
            num_frequencies=eis_num_frequencies, latent_dim=latent_dim
        )
        self.physics_encoder = PhysicsEncoder(
            num_physics_features=physics_num_features, latent_dim=latent_dim
        )

        # --- Fusion ---
        if fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(latent_dim=latent_dim, fusion_dim=fusion_dim)
        elif fusion_method == "weighted":
            self.fusion = WeightedFusion(latent_dim=latent_dim, fusion_dim=fusion_dim)
        else:  # concat
            self.fusion = ConcatFusion(latent_dim=latent_dim, fusion_dim=fusion_dim)

        # --- SoH prediction head ---
        self.soh_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),   # SoH prediction in [0, 100]
        )

    def forward(
        self,
        discharge: torch.Tensor,
        eis: torch.Tensor,
        physics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BaFuse.

        Args:
            discharge: (B, seq_len, discharge_features) or (B, discharge_features)
            eis:       (B, eis_features) or (B, num_freq, 2)
            physics:   (B, physics_features)

        Returns:
            dict with:
                soh_pred:           (B, 1)
                discharge_latent:   (B, latent_dim)
                eis_latent:         (B, latent_dim)
                physics_latent:     (B, latent_dim)
                fused:              (B, fusion_dim)
                attention_weights:  optional
        """
        # Encode each modality
        discharge_latent = self.discharge_encoder(discharge)   # (B, latent_dim)
        eis_latent = self.eis_encoder(eis)                     # (B, latent_dim)
        physics_latent = self.physics_encoder(physics)         # (B, latent_dim)

        # Fuse
        fused, fusion_info = self.fusion(discharge_latent, eis_latent, physics_latent)

        # Predict SoH
        soh_pred = self.soh_head(fused)

        return {
            "soh_pred": soh_pred,
            "discharge_latent": discharge_latent,
            "eis_latent": eis_latent,
            "physics_latent": physics_latent,
            "fused": fused,
            **fusion_info,
        }
