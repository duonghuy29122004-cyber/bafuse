"""
DegradationHead — multi-output regression head for battery degradation-mode estimation.

Outputs three model-derived degradation-mode labels:
  - LLI  : Loss of Lithium Inventory   (%)
  - LAM  : Loss of Active Material     (%)
  - CL   : Conductivity Loss           (%)

IMPORTANT: These labels are model-derived from EIS / equivalent-circuit fitting
(Mendeley dataset). They should NOT be interpreted as absolute physical ground
truth. Use terminology "model-derived degradation-mode estimates" in any report.

Design:
  Fusion representation  (B, fusion_dim)
        ↓
  Shared MLP trunk
        ↓
  Three parallel output branches → (LLI, LAM, CL)

The head can also be used standalone on top of the EIS encoder latent (for
Stage-1 Mendeley-only pre-training) by passing the EIS latent directly.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class DegradationHead(nn.Module):
    """
    Multi-output regression head predicting model-derived degradation-mode labels.

    Args:
        input_dim    : Dimension of the input representation (fusion_dim or latent_dim).
        hidden_dim   : Hidden layer width in the shared trunk.
        dropout      : Dropout probability.
        output_scale : Optional per-output scaling factors [lli, lam, cl].
                       If None, raw outputs are returned (suitable for normalized targets).
    """

    def __init__(
        self,
        input_dim:    int   = 128,
        hidden_dim:   int   = 128,
        dropout:      float = 0.2,
        output_scale: Tuple[float, float, float] = None,
    ):
        super().__init__()

        # ---- Shared trunk ------------------------------------------------
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        out_in = hidden_dim // 2

        # ---- Three parallel heads ----------------------------------------
        self.head_lli = nn.Linear(out_in, 1)   # Loss of Lithium Inventory
        self.head_lam = nn.Linear(out_in, 1)   # Loss of Active Material
        self.head_cl  = nn.Linear(out_in, 1)   # Conductivity Loss

        # Optional scaling (e.g. 100 when targets are in %)
        if output_scale is not None:
            self.register_buffer(
                "output_scale",
                torch.tensor(output_scale, dtype=torch.float32),
            )
        else:
            self.output_scale = None

        self._init_weights()

    # -----------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) — fused representation OR EIS latent.

        Returns:
            dict with keys:
                'lli_pred'  : (B, 1)
                'lam_pred'  : (B, 1)
                'cl_pred'   : (B, 1)
                'deg_pred'  : (B, 3)  concatenated [LLI, LAM, CL]
        """
        shared = self.trunk(x)              # (B, hidden_dim // 2)

        lli = self.head_lli(shared)         # (B, 1)
        lam = self.head_lam(shared)         # (B, 1)
        cl  = self.head_cl(shared)          # (B, 1)

        if self.output_scale is not None:
            lli = lli * self.output_scale[0]
            lam = lam * self.output_scale[1]
            cl  = cl  * self.output_scale[2]

        deg_pred = torch.cat([lli, lam, cl], dim=-1)   # (B, 3)

        return {
            "lli_pred":  lli,
            "lam_pred":  lam,
            "cl_pred":   cl,
            "deg_pred":  deg_pred,
        }
