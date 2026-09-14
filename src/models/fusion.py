"""
Fusion mechanisms combining discharge, EIS, and physics encodings.

Fusion strategies:
- Simple concatenation + MLP
- Cross-attention (query one modality against others)
- Physics-informed weighted fusion (learned weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion.

    Each modality attends to the other two, then the attended representations
    are concatenated and projected to fusion_dim.
    """

    def __init__(self, latent_dim: int = 64, num_heads: int = 4, fusion_dim: int = 128):
        """
        Args:
            latent_dim: Input latent dimension (same for all encoders)
            num_heads: Number of attention heads
            fusion_dim: Output fused dimension
        """
        super().__init__()
        # Multi-head self-attention over the stacked 3 tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(latent_dim)
        # Project 3 * latent_dim → fusion_dim
        self.proj = nn.Sequential(
            nn.Linear(3 * latent_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        discharge_latent: torch.Tensor,
        eis_latent: torch.Tensor,
        physics_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            discharge_latent: (B, latent_dim)
            eis_latent:       (B, latent_dim)
            physics_latent:   (B, latent_dim)

        Returns:
            fused:            (B, fusion_dim)
            info:             dict with attention_weights
        """
        # Stack into sequence: (B, 3, latent_dim)
        tokens = torch.stack([discharge_latent, eis_latent, physics_latent], dim=1)

        # Self-attention across modalities
        attended, attn_weights = self.attn(tokens, tokens, tokens)  # (B, 3, latent_dim)
        attended = self.norm(attended + tokens)                      # residual

        # Flatten and project
        flat = attended.reshape(attended.size(0), -1)   # (B, 3*latent_dim)
        fused = self.proj(flat)                          # (B, fusion_dim)

        return fused, {"attention_weights": attn_weights}


class WeightedFusion(nn.Module):
    """
    Learn scalar weights for each modality.

    output = w_d * discharge + w_e * eis + w_p * physics  (softmax-normalised)
    Then project to fusion_dim.
    """

    def __init__(self, latent_dim: int = 64, fusion_dim: int = 128):
        super().__init__()
        # Learnable raw logits; softmax gives normalised weights
        self.raw_weights = nn.Parameter(torch.zeros(3))
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, fusion_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        discharge_latent: torch.Tensor,
        eis_latent: torch.Tensor,
        physics_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            fused: (B, fusion_dim)
            info:  dict with per-modality weights
        """
        w = F.softmax(self.raw_weights, dim=0)   # (3,)
        combined = (
            w[0] * discharge_latent
            + w[1] * eis_latent
            + w[2] * physics_latent
        )  # (B, latent_dim)
        fused = self.proj(combined)
        return fused, {
            "weights": {
                "discharge": w[0].item(),
                "eis": w[1].item(),
                "physics": w[2].item(),
            }
        }


class ConcatFusion(nn.Module):
    """
    Simple concatenation + MLP fusion (baseline).
    """

    def __init__(self, latent_dim: int = 64, fusion_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * latent_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        discharge_latent: torch.Tensor,
        eis_latent: torch.Tensor,
        physics_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            fused: (B, fusion_dim)
            info:  empty dict for API compatibility
        """
        cat = torch.cat([discharge_latent, eis_latent, physics_latent], dim=-1)
        fused = self.mlp(cat)
        return fused, {}
