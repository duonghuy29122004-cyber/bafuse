"""
BaFuseV2 - Extended multimodal model.

Adds:
  1. DegradationHead for model-derived LLI/LAM/CL estimation.
  2. Separate EIS encoder for Mendeley path (different feature dim to NASA).
  3. Modality-masking support for ablation (A1-A7) without retraining.
  4. Backward-compatible with BaFuseV1 checkpoints via from_bafuse_v1_checkpoint().

Architecture:
    NASA: Discharge + EIS + Physics -> CrossAttentionFusion -> SOH Head
    Mendeley: EIS (8D) -> mendeley_eis_encoder -> degradation_head_eis_only -> LLI/LAM/CL
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
    Multimodal BaFuse v2: SOH + degradation-mode estimation.

    Args:
        discharge_input_size       : Discharge features per time-step.
        eis_num_frequencies        : NASA EIS feature count.
        physics_num_features       : Physics feature count.
        latent_dim                 : Latent dim for each encoder.
        fusion_method              : "cross_attention" | "weighted" | "concat".
        fusion_dim                 : Fused feature dim.
        physics_encoder_type       : "mlp" | "cnn1d".
        deg_hidden_dim             : Hidden dim for DegradationHead.
        deg_dropout                : Dropout for DegradationHead.
        mendeley_eis_num_features  : Mendeley EIS feature count (default = same as NASA).
        use_discharge/eis/physics  : Ablation flags (zero-mask when False).
    """

    def __init__(
        self,
        discharge_input_size:      int   = 3,
        eis_num_frequencies:       int   = 3,
        physics_num_features:      int   = 4,
        latent_dim:                int   = 64,
        fusion_method:             str   = "cross_attention",
        fusion_dim:                int   = 128,
        physics_encoder_type:      str   = "mlp",
        deg_hidden_dim:            int   = 128,
        deg_dropout:               float = 0.2,
        mendeley_eis_num_features: Optional[int] = None,
        use_discharge:             bool  = True,
        use_eis:                   bool  = True,
        use_physics:               bool  = True,
    ):
        super().__init__()

        self.use_discharge = use_discharge
        self.use_eis       = use_eis
        self.use_physics   = use_physics
        self.latent_dim    = latent_dim
        self.fusion_dim    = fusion_dim

        # ---- NASA encoders ----------------------------------------------
        self.discharge_encoder = DischargeEncoder(
            input_size=discharge_input_size, latent_dim=latent_dim
        )
        self.eis_encoder = EISEncoder(
            num_frequencies=eis_num_frequencies, latent_dim=latent_dim
        )

        # ---- Mendeley EIS encoder (may differ in feature dim) -----------
        _mend_dim = mendeley_eis_num_features or eis_num_frequencies
        self._mendeley_eis_dim = _mend_dim
        if _mend_dim != eis_num_frequencies:
            # Separate encoder when Mendeley has different EIS feature count
            self.mendeley_eis_encoder = EISEncoder(
                num_frequencies=_mend_dim, latent_dim=latent_dim
            )
        else:
            # Share weights when dims are equal
            self.mendeley_eis_encoder = self.eis_encoder

        # ---- Physics encoder --------------------------------------------
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

        # ---- Degradation head (from fused representation) ---------------
        self.degradation_head = DegradationHead(
            input_dim=fusion_dim,
            hidden_dim=deg_hidden_dim,
            dropout=deg_dropout,
        )

        # ---- Degradation head for EIS-only Mendeley path ----------------
        self.degradation_head_eis_only = DegradationHead(
            input_dim=latent_dim,
            hidden_dim=deg_hidden_dim // 2,
            dropout=deg_dropout,
        )

    # -----------------------------------------------------------------------
    # Encode helpers
    # -----------------------------------------------------------------------

    def _encode(
        self,
        discharge: torch.Tensor,
        eis:       torch.Tensor,
        physics:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode all modalities; zero-mask inactive ones for ablation."""
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
    # Main forward (NASA path: SOH + optional degradation from fused rep)
    # -----------------------------------------------------------------------

    def forward(
        self,
        discharge:   torch.Tensor,
        eis:         torch.Tensor,
        physics:     torch.Tensor,
        predict_deg: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full multimodal forward (NASA path).

        Args:
            discharge    : (B, seq_len, F) or (B, F)
            eis          : (B, nasa_eis_F)   -- NASA EIS features
            physics      : (B, phys_F)
            predict_deg  : Also run degradation head on fused rep (research mode).
        """
        d_latent, e_latent, p_latent = self._encode(discharge, eis, physics)
        fused, fusion_info = self.fusion(d_latent, e_latent, p_latent)
        soh_pred = self.soh_head(fused)

        out = {
            "soh_pred":         soh_pred,
            "discharge_latent": d_latent,
            "eis_latent":       e_latent,
            "physics_latent":   p_latent,
            "fused":            fused,
            **fusion_info,
        }

        if predict_deg:
            deg_out = self.degradation_head(fused)
            out.update(deg_out)

        return out

    # -----------------------------------------------------------------------
    # Mendeley / EIS-only path (Stage-1 pre-training)
    # -----------------------------------------------------------------------

    def forward_eis_only(self, eis: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Mendeley EIS path: eis -> mendeley_eis_encoder -> degradation_head_eis_only.

        Uses mendeley_eis_encoder which handles Mendeley's EIS feature dim.
        NASA samples do NOT have LLI/LAM/CL labels; do not call this on NASA batches.

        Args:
            eis: (B, mendeley_eis_features)
        """
        e_latent = self.mendeley_eis_encoder(eis)
        deg_out  = self.degradation_head_eis_only(e_latent)
        return {"eis_latent": e_latent, **deg_out}

    # -----------------------------------------------------------------------
    # Ablation flags
    # -----------------------------------------------------------------------

    def set_modality_flags(
        self,
        use_discharge: Optional[bool] = None,
        use_eis:       Optional[bool] = None,
        use_physics:   Optional[bool] = None,
    ):
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
        Encoder + fusion + SOH head weights are transferred.
        Degradation head is freshly initialized.
        """
        import logging
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        cfg  = ckpt.get("config", {}).get("model", {})
        init_kwargs = {
            "discharge_input_size": cfg.get("discharge_input_size", 3),
            "eis_num_frequencies":  cfg.get("eis_num_frequencies",  3),
            "physics_num_features": cfg.get("physics_num_features", 4),
            "latent_dim":           cfg.get("latent_dim",            64),
            "fusion_method":        cfg.get("fusion_method",         "cross_attention"),
            "fusion_dim":           cfg.get("fusion_dim",            128),
            "physics_encoder_type": cfg.get("physics_encoder_type",  "mlp"),
        }
        init_kwargs.update(kwargs)
        model = cls(**init_kwargs)

        v1_state = ckpt.get("model_state_dict", ckpt)
        v2_state = model.state_dict()
        transferred, skipped = 0, 0
        for key, val in v1_state.items():
            # Direct match
            if key in v2_state and v2_state[key].shape == val.shape:
                v2_state[key] = val
                transferred  += 1
            # Map Stage-1 eis_encoder -> mendeley_eis_encoder in v2
            # (Stage-1 trained on 8D Mendeley EIS; v2 mendeley_eis_encoder has same dims)
            elif key.startswith("eis_encoder."):
                mapped_key = key.replace("eis_encoder.", "mendeley_eis_encoder.", 1)
                if mapped_key in v2_state and v2_state[mapped_key].shape == val.shape:
                    v2_state[mapped_key] = val
                    transferred += 1
                else:
                    skipped += 1
            else:
                skipped += 1
        model.load_state_dict(v2_state)
        logging.getLogger(__name__).info(
            f"BaFuseV1->V2 transfer: {transferred} params loaded, "
            f"{skipped} skipped (new/shape-mismatch)"
        )
        return model
