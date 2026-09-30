"""
TinySOH — small CNN + MLP baseline for NASA SOH.

~15k parameters. Same forward signature as BaFuseV2 (discharge, eis, physics)
so it can reuse BaFuseDataset / evaluate().

This is the architecture-level baseline that BaFuseV2 should beat, not a
constant mean predictor.
"""

import torch
import torch.nn as nn
from typing import Dict


class TinySOH(nn.Module):
    """1-D CNN on the discharge curve, concat EIS + physics, shallow MLP head."""

    def __init__(
        self,
        discharge_channels: int = 3,
        eis_dim: int = 3,
        physics_dim: int = 4,
        conv_channels: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(discharge_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        fused_in = conv_channels + eis_dim + physics_dim
        self.head = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        discharge: torch.Tensor,
        eis: torch.Tensor,
        physics: torch.Tensor,
        predict_deg: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Dataset yields (B, seq_len, channels). Aggregate fallback is (B, F).
        if discharge.dim() == 2:
            x = discharge.unsqueeze(1)
        else:
            x = discharge.transpose(1, 2)
        curve = self.conv(x).squeeze(-1)
        fused = torch.cat([curve, eis, physics], dim=-1)
        soh = self.head(fused)
        return {"soh_pred": soh, "fused": fused}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
