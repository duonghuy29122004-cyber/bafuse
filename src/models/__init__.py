"""
BaFuse model package.

Public API:
    BaFuse     — v1 model (original, backward compatible)
    BaFuseV2   — v2 model (adds degradation head + modality masking)
    DegradationHead
    DischargeEncoder, EISEncoder, PhysicsEncoder, PhysicsEncoderCNN
    CrossAttentionFusion, WeightedFusion, ConcatFusion
"""

from .bafuse import BaFuse
from .bafuse_v2 import BaFuseV2
from .degradation_head import DegradationHead
from .encoders import (
    DischargeEncoder,
    EISEncoder,
    PhysicsEncoder,
    PhysicsEncoderCNN,
)
from .fusion import CrossAttentionFusion, WeightedFusion, ConcatFusion

__all__ = [
    "BaFuse",
    "BaFuseV2",
    "DegradationHead",
    "DischargeEncoder",
    "EISEncoder",
    "PhysicsEncoder",
    "PhysicsEncoderCNN",
    "CrossAttentionFusion",
    "WeightedFusion",
    "ConcatFusion",
]
