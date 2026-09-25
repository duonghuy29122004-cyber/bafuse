"""
BaFuseV2 — Extended multimodal model adding:

  1. DegradationHead for model-derived LLI / LAM / CL estimation.
  2. Modality-masking support (use_discharge / use_eis / use_physics flags)
     enabling the full A1–A7 ablation matrix without retraining.
  3. Backward-compatible with BaFuse checkpoints — the SOH head weights are
     identical; only the degradation head is new.

Architecture:

    NASA PCoE
        |
  +-----+-----+--------+
  |           |        |
  ↓           ↓        ↓
Discharge   EIS    Physics
Encoder   Encoder  Encoder
  |           |        |
  +-----------+---------+
              |
       CrossAttentionFusion
              |
      +-------+-------+
      |               |
      ↓               ↓
   SOH Head     Degradation Head
      |               |
      ↓               ↓
     SOH        LLI / LAM / CL
                      ↑
               Mendeley EIS

Modality masking:
  When use_discharge=False, the discharge tensor is replaced by zeros before
  the encoder, effectively removing that signal. This is functionally equivalent
  to training without that modality and is used exclusively for *evaluation-time*
  ablation; for training-time ablation the same flags disable the modality during
  a forward pass.

  NOTE: Zero-masking at inference gives a lower-bound contribution estimate.
  It does NOT capture what a model retrained without that modality would achieve.
  See experiments/run_ablation.py for full retrain-based ablation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoders import (
    DischargeEncoder,
    EISEncoder,
    PhysicsEncoder,
    PhysicsEncoderCNN,
)
from .fusion import CrossAttentionFusion, WeightedFusion, ConcatFusion
from .degradation_head import DegradationHead


class BaFuseV2(nn.Module):
    """
    Multimodal BaFuse model v2: SOH + degradation-mode estimation.

    Args:
        discharge_input_size  : Number of discharge features per time-step.
        eis_num_frequencies   : Number of EIS features.
        physics_num_features  : Number of physics-derived features.
        latent_dim            : Latent dimension for each encoder output.
        fusion_method         : "cross_attention" | "weighted" | "concat".
        fusion_dim            : Fused feature dimension.
        physics_encoder_type  : "mlp" | "cnn1d".
        deg_hidden_dim        : Hidden dim for DegradationHead.
        deg_dropout           : Dropout for DegradationHead.
        use_discharge         : If False, zero-mask discharge at inference (ablation).
        use_eis               : If False, zero-mask EIS at inference.
        use_physics           : If False, zero-mask physics at inference.
    """

    def __init__(
        self,
        discharge_input_size:  int   = 3,
        eis_num_frequencies:   int   = 3,
        physics_num_features:  int   = 4,
        latent_dim:            int   = 64,
        fusion_method:         str   = "cross_attention",
        fusion_dim:            int   = 128,
        physics_encoder_type:  str   = "mlp",
        deg_hidden_dim:        int   = 128,
        deg_dropout:           float = 0.2,
        # Ablation flags — default: all modalities active
        use_discharge:         bool  = True,
        use_eis:               bool  = True,
        use_physics:           bool  = True,
    ):
        super().__init__()

        # ---- store flags ------------------------------------------------
        self.use_discharge = use_discharge
        self.use_eis       = use_eis
        self.use_physics   = use_physics
        self.latent_dim    = latent_dim
        self.fusion_dim    = fusion_dim

        # ---- Encoders ---------------------------------------------------
        self.discharge_encoder = DischargeEncoder(
            input_size=discharge_input_size, latent_dim=latent_dim
        )
        self.eis_encoder = EISEncoder(
            num_frequencies=eis_num_frequencies, latent_dim=latent_dim
        )
        if physics_encoder_type == "cnn1d":
            self.physics_encoder = PhysicsEncoderCNN(
                num_physics_features=physics_num_features, latent_dim=latent_dim
            )
        else:
            self.physics_encoder = PhysicsEncoder(
                num_physics_features=physics_num_features, latent_dim=latent_dim
            )

        # ---- Fusion -----------------------------------------------------
        if fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(
                latent_dim=latent_dim, fusion_dim=fusion_dim
            )
        elif fusion_method == "weighted":
            self.fusion = WeightedFusion(latent_dim=latent_dim, fusion_dim=fusion_dim)
        else:
            self.fusion = ConcatFusion(latent_dim=latent_dim, fusion_dim=fusion_dim)

        # ---- SOH head ---------------------------------------------------
        self.soh_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # ---- Degradation head -------------------------------------------
        # Accepts fused representation (NASA path) OR EIS latent (Mendeley path)
        # Input dim is fusion_dim for full model; latent_dim for EIS-only path.
        self.degradation_head = DegradationHead(
            input_dim=fusion_dim,
            hidden_dim=deg_hidden_dim,
            dropout=deg_dropout,
        )

        # Separate head for EIS-only / Mendeley path (input_dim = latent_dim)
        self.degradation_head_eis_only = DegradationHead(
            input_dim=latent_dim,
            hidden_dim=deg_hidden_dim // 2,
            dropout=deg_dropout,
        )

    # -----------------------------------------------------------------------
    # Forward helpers
    # -----------------------------------------------------------------------

    def _encode(
        self,
        discharge: torch.Tensor,
        eis:       torch.Tensor,
        physics:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode each modality, applying zero-mask where flag is False."""

        if self.use_discharge:
            d_latent = self.discharge_encoder(discharge)
        else:
            d_latent = torch.zeros(
                discharge.size(0), self.latent_dim, device=discharge.device
            )

        if self.use_eis:
            e_latent = self.eis_encoder(eis)
        else:
            e_latent = torch.zeros(
                eis.size(0), self.latent_dim, device=eis.device
            )

        if self.use_physics:
            p_latent = self.physics_encoder(physics)
        else:
            p_latent = torch.zeros(
                physics.size(0), self.latent_dim, device=physics.device
            )

        return d_latent, e_latent, p_latent

    # -----------------------------------------------------------------------
    # Main forward pass (NASA path: SOH + optional degradation from fused rep)
    # -----------------------------------------------------------------------

    def forward(
        self,
        discharge:       torch.Tensor,
        eis:             torch.Tensor,
        physics:         torch.Tensor,
        predict_deg:     bool = False,   # also run degradation head on fused rep
    ) -> Dict[str, torch.Tensor]:
        """
        Full multimodal forward pass.

        Args:
            discharge    : (B, seq_len, F) or (B, F)
            eis          : (B, eis_F)
            physics      : (B, phys_F)
            predict_deg  : If True, also return degradation predictions from
                           the fused representation (research mode).

        Returns:
            dict with soh_pred, latents, fused, and optionally deg predictions.
        """
        d_latent, e_latent, p_latent = self._encode(discharge, eis, physics)

        fused, fusion_info = self.fusion(d_latent, e_latent, p_latent)

        soh_pred = self.soh_head(fused)

        out = {
            "soh_pred":          soh_pred,
            "discharge_latent":  d_latent,
            "eis_latent":        e_latent,
            "physics_latent":    p_latent,
            "fused":             fused,
            **fusion_info,
        }

        if predict_deg:
            deg_out = self.degradation_head(fused)
            out.update(deg_out)

        return out

    # -----------------------------------------------------------------------
    # Mendeley / EIS-only path (Stage 1: degrade head pre-training)
    # -----------------------------------------------------------------------

    def forward_eis_only(self, eis: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        EIS-encoder → degradation-head path.

        Used for Stage-1 pre-training on the Mendeley dataset where only EIS
        features and model-derived degradation-mode labels are available.

        NOTE: NASA samples do NOT have LLI/LAM/CL labels — do not call this
        method on NASA batches.

        Args:
            eis: (B, eis_features)

        Returns:
            dict with eis_latent, lli_pred, lam_pred, cl_pred, deg_pred.
        """
        e_latent = self.eis_encoder(eis)
        deg_out  = self.degradation_head_eis_only(e_latent)

        return {
            "eis_latent": e_latent,
            **deg_out,
        }

    # -----------------------------------------------------------------------
    # Modality-flag setters (useful for eval-time ablation without re-init)
    # -----------------------------------------------------------------------

    def set_modality_flags(
        self,
        use_discharge: Optional[bool] = None,
        use_eis:       Optional[bool] = None,
        use_physics:   Optional[bool] = None,
    ):
        """Set ablation flags at runtime (evaluation-time zero-masking)."""
        if use_discharge is not None:
            self.use_discharge = use_discharge
        if use_eis is not None:
            self.use_eis = use_eis
        if use_physics is not None:
            self.use_physics = use_physics

    def active_modalities(self) -> Dict[str, bool]:
        return {
            "discharge": self.use_discharge,
            "eis":       self.use_eis,
            "physics":   self.use_physics,
        }

    # -----------------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------------

    @classmethod
    def from_bafuse_v1_checkpoint(
        cls,
        checkpoint_path: str,
        map_location:    str = "cpu",
        **kwargs,
    ) -> "BaFuseV2":
        """
        Load a BaFuseV1 checkpoint into BaFuseV2.

        Encoder + fusion + SOH head weights are transferred. The degradation
        head is freshly initialized (random weights).

        Args:
            checkpoint_path : Path to .pth checkpoint file.
            map_location    : Torch device string.
            **kwargs        : Additional BaFuseV2 constructor arguments.

        Returns:
            BaFuseV2 instance with v1 weights loaded.
        """
        import torch
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Read model config from checkpoint if available
        cfg = ckpt.get("config", {}).get("model", {})
        init_kwargs = {
            "discharge_input_size": cfg.get("discharge_input_size", 3),
            "eis_num_frequencies":  cfg.get("eis_num_frequencies", 3),
            "physics_num_features": cfg.get("physics_num_features", 4),
            "latent_dim":           cfg.get("latent_dim", 64),
            "fusion_method":        cfg.get("fusion_method", "cross_attention"),
            "fusion_dim":           cfg.get("fusion_dim", 128),
            "physics_encoder_type": cfg.get("physics_encoder_type", "mlp"),
        }
        init_kwargs.update(kwargs)

        model = cls(**init_kwargs)

        v1_state = ckpt.get("model_state_dict", ckpt)
        v2_state = model.state_dict()

        transferred = 0
        skipped     = 0
        for key, val in v1_state.items():
            if key in v2_state and v2_state[key].shape == val.shape:
                v2_state[key] = val
                transferred  += 1
            else:
                skipped += 1

        model.load_state_dict(v2_state)
        import logging
        logging.getLogger(__name__).info(
            f"BaFuseV1→V2 transfer: {transferred} params loaded, "
            f"{skipped} skipped (new/shape-mismatch)"
        )
        return model
