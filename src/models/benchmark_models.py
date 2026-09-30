"""
BaFuse Architecture Benchmark v1 — Competing Models A–G.

All models share the same forward signature:
    forward(discharge, eis, physics) -> {"soh_pred": (B,1), "fused": (B,D)}

This matches BaFuse / BaFuseV2 so they all plug into the same evaluation
loop in scripts/architecture_benchmark.py.

Input shapes (confirmed from benchmark_audit.py):
    discharge : (B, 100, 3)  — z-scored V/I/T time-series
    eis       : (B, 3)        — z-scored [|Z|, Re, Rct]
    physics   : (B, 4)        — [cycle_age_norm, fade_prior, v_droop, imp_rise]
    soh_label : (B,)          — [0, 1]

Models implemented:
    ModelA  — TinyMultimodalMLP    flat stats concat → 3-layer MLP
    ModelB  — SmallMLPFusion       per-modality mini-MLP → concat → MLP
    ModelC  — CNN1DFusion          shallow CNN discharge + EIS + Physics → MLP
    ModelD  — TCNFusion            dilated TCN discharge + EIS + Physics → MLP
    ModelE  — SmallLSTM            1-layer LSTM (64) + EIS + Physics → MLP
    ModelE2 — SmallLSTM128         1-layer LSTM (128) variant
    ModelF  — GatedFusion          learned soft-gate over 3 modality embeddings
    ModelG  — CNNGatedFusion       CNN discharge + EIS + Physics + gated fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helper: small MLP block
# ═══════════════════════════════════════════════════════════════════════════════

def _mlp(in_dim: int, out_dim: int, hidden: int = None, dropout: float = 0.1):
    hidden = hidden or max(in_dim, out_dim) * 2
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
        nn.ReLU(),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A — TinyMultimodalMLP
# ═══════════════════════════════════════════════════════════════════════════════

class ModelA_TinyMLP(nn.Module):
    """
    Flatten all modalities into one feature vector, feed through shallow MLP.

    Discharge: aggregate stats derived from the time-series
        (we pool the (100,3) tensor to mean/max/min/std per channel → 12 stats)
    EIS:     3 features
    Physics: 4 features
    Total flat input: 12 + 3 + 4 = 19D

    Architecture:
        19 -> LN -> 64 -> ReLU -> Dropout(0.1)
            -> 32 -> ReLU
            -> 1
    """

    def __init__(self, eis_dim: int = 3, physics_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        # 4 stats (mean/std/min/max) × 3 channels = 12  + eis + physics
        disc_stats_dim = 12
        flat_in = disc_stats_dim + eis_dim + physics_dim  # 19

        self.net = nn.Sequential(
            nn.LayerNorm(flat_in),
            nn.Linear(flat_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _pool_discharge(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, 4*C) via mean/std/min/max per channel."""
        if x.dim() == 2:          # aggregate fallback (B, C)
            return x.repeat(1, 4)
        m   = x.mean(dim=1)       # (B, C)
        s   = x.std(dim=1)        # (B, C)
        mn  = x.min(dim=1).values # (B, C)
        mx  = x.max(dim=1).values # (B, C)
        return torch.cat([m, s, mn, mx], dim=-1)   # (B, 4C=12)

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        d_feat = self._pool_discharge(discharge)
        fused  = torch.cat([d_feat, eis, physics], dim=-1)
        soh    = self.net(fused)
        return {"soh_pred": soh, "fused": fused}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL B — SmallMLPFusion
# ═══════════════════════════════════════════════════════════════════════════════

class ModelB_SmallMLP(nn.Module):
    """
    Per-modality mini-MLP → concatenate 96D → fusion MLP → SOH.

    Discharge stats → 32D
    EIS            → 32D
    Physics        → 32D
    Concat 96D → 64 → 32 → 1
    """

    def __init__(self, eis_dim: int = 3, physics_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        disc_stats_dim = 12

        self.d_enc = _mlp(disc_stats_dim, 32, dropout=dropout)
        self.e_enc = _mlp(eis_dim,        32, dropout=dropout)
        self.p_enc = _mlp(physics_dim,    32, dropout=dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _pool_discharge(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.repeat(1, 4)
        return torch.cat([x.mean(1), x.std(1), x.min(1).values, x.max(1).values], -1)

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        d = self.d_enc(self._pool_discharge(discharge))
        e = self.e_enc(eis)
        p = self.p_enc(physics)
        fused = torch.cat([d, e, p], dim=-1)
        return {"soh_pred": self.head(fused), "fused": fused}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL C — CNN1DFusion
# ═══════════════════════════════════════════════════════════════════════════════

class ModelC_CNN1D(nn.Module):
    """
    Shallow 1D-CNN on discharge time-series → 64D.
    EIS MLP → 32D.  Physics MLP → 32D.
    Concat 128D → MLP → SOH.
    """

    def __init__(self, disc_channels: int = 3, eis_dim: int = 3,
                 physics_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(disc_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.e_enc = _mlp(eis_dim,     32, dropout=dropout)
        self.p_enc = _mlp(physics_dim, 32, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        if discharge.dim() == 2:
            x = discharge.unsqueeze(1)
        else:
            x = discharge.transpose(1, 2)     # (B, 3, T)
        d = self.cnn(x).squeeze(-1)           # (B, 64)
        e = self.e_enc(eis)
        p = self.p_enc(physics)
        fused = torch.cat([d, e, p], dim=-1)
        return {"soh_pred": self.head(fused), "fused": fused}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL D — TCNFusion  (dilated causal residual blocks)
# ═══════════════════════════════════════════════════════════════════════════════

class _TCNBlock(nn.Module):
    """One dilated causal residual block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) * dilation     # causal padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.conv1.padding[0] if isinstance(self.conv1.padding, tuple) else self.conv1.padding
        r = F.relu(self.bn1(self.conv1(x)[..., :-pad] if pad else self.conv1(x)))
        r = self.drop(r)
        r = F.relu(self.bn2(self.conv2(r)[..., :-pad] if pad else self.conv2(r)))
        return F.relu(r + self.skip(x)[..., :r.shape[-1]])


class ModelD_TCN(nn.Module):
    """
    3 dilated-residual TCN blocks (dilations 1,2,4).
    Discharge channels 3→32→32→64. Global avg-pool → 64D.
    EIS→32D, Physics→32D. Concat 128→64→1.
    """

    def __init__(self, disc_channels: int = 3, eis_dim: int = 3,
                 physics_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.tcn = nn.Sequential(
            _TCNBlock(disc_channels, 32, dilation=1, dropout=dropout),
            _TCNBlock(32, 32,           dilation=2, dropout=dropout),
            _TCNBlock(32, 64,           dilation=4, dropout=dropout),
        )
        self.e_enc = _mlp(eis_dim,     32, dropout=dropout)
        self.p_enc = _mlp(physics_dim, 32, dropout=dropout)
        self.head  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        if discharge.dim() == 2:
            x = discharge.unsqueeze(1)
        else:
            x = discharge.transpose(1, 2)     # (B, 3, T)
        d = self.tcn(x).mean(dim=-1)          # global avg pool → (B, 64)
        e = self.e_enc(eis)
        p = self.p_enc(physics)
        fused = torch.cat([d, e, p], dim=-1)
        return {"soh_pred": self.head(fused), "fused": fused}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL E — SmallLSTM  (single-layer, hidden=64 and hidden=128 variants)
# ═══════════════════════════════════════════════════════════════════════════════

class ModelE_SmallLSTM(nn.Module):
    """
    1-layer LSTM (hidden=H) → final hidden → H-dim.
    EIS→32D, Physics→32D. Concat (H+64)D → MLP → SOH.
    """

    def __init__(self, disc_channels: int = 3, hidden: int = 64,
                 eis_dim: int = 3, physics_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.lstm  = nn.LSTM(disc_channels, hidden, num_layers=1,
                             batch_first=True, dropout=0.0)
        self.e_enc = _mlp(eis_dim,     32, dropout=dropout)
        self.p_enc = _mlp(physics_dim, 32, dropout=dropout)
        fused_in   = hidden + 32 + 32
        self.head  = nn.Sequential(
            nn.Linear(fused_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        if discharge.dim() == 2:
            x = discharge.unsqueeze(1)
        else:
            x = discharge
        _, (h_n, _) = self.lstm(x)
        d = h_n[-1]                            # (B, hidden)
        e = self.e_enc(eis)
        p = self.p_enc(physics)
        fused = torch.cat([d, e, p], dim=-1)
        return {"soh_pred": self.head(fused), "fused": fused}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL F — GatedFusion  (soft-gate over 3 modality embeddings)
# ═══════════════════════════════════════════════════════════════════════════════

class ModelF_GatedFusion(nn.Module):
    """
    Each modality → 64D embedding.
    Gating: g = softmax(Linear(concat_3_embeddings)) → 3 scalars.
    z_fused = g0*z_d + g1*z_e + g2*z_p  (64D weighted sum).
    SOH head: 64 → 32 → 1.

    Uses CNN1D for discharge (better than stat-pooling for temporal signal).
    """

    def __init__(self, disc_channels: int = 3, eis_dim: int = 3,
                 physics_dim: int = 4, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Discharge: shallow CNN → 64D
        self.d_enc = nn.Sequential(
            nn.Conv1d(disc_channels, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, embed_dim, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.e_enc = nn.Sequential(
            nn.Linear(eis_dim, 32), nn.ReLU(),
            nn.Linear(32, embed_dim), nn.ReLU(),
        )
        self.p_enc = nn.Sequential(
            nn.Linear(physics_dim, 32), nn.ReLU(),
            nn.Linear(32, embed_dim), nn.ReLU(),
        )
        # Gate: takes concat of 3 embeddings → 3 gate weights
        self.gate = nn.Linear(embed_dim * 3, 3)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        if discharge.dim() == 2:
            x = discharge.unsqueeze(1)
        else:
            x = discharge.transpose(1, 2)
        z_d = self.d_enc(x).squeeze(-1)      # (B, 64)
        z_e = self.e_enc(eis)                # (B, 64)
        z_p = self.p_enc(physics)            # (B, 64)

        cat_all = torch.cat([z_d, z_e, z_p], dim=-1)  # (B, 192)
        g = torch.softmax(self.gate(cat_all), dim=-1)  # (B, 3)

        fused = (g[:, 0:1] * z_d +
                 g[:, 1:2] * z_e +
                 g[:, 2:3] * z_p)            # (B, 64)

        return {
            "soh_pred": self.head(fused),
            "fused":    fused,
            "gates":    g,                    # record for analysis
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL G — CNNGatedFusion
# ═══════════════════════════════════════════════════════════════════════════════

class ModelG_CNNGated(nn.Module):
    """
    CNN1D discharge + EIS MLP + Physics MLP → gated fusion → SOH.
    Identical gate mechanism to ModelF, but uses a dedicated
    EIS/Physics encoder projection to match discharge embed_dim.
    """

    def __init__(self, disc_channels: int = 3, eis_dim: int = 3,
                 physics_dim: int = 4, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.d_enc = nn.Sequential(
            nn.Conv1d(disc_channels, 32, 5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, embed_dim, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.e_enc = _mlp(eis_dim,     embed_dim, hidden=64, dropout=dropout)
        self.p_enc = _mlp(physics_dim, embed_dim, hidden=64, dropout=dropout)

        self.gate = nn.Linear(embed_dim * 3, 3)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        if discharge.dim() == 2:
            x = discharge.unsqueeze(1)
        else:
            x = discharge.transpose(1, 2)
        z_d = self.d_enc(x).squeeze(-1)
        z_e = self.e_enc(eis)
        z_p = self.p_enc(physics)

        g = torch.softmax(
            self.gate(torch.cat([z_d, z_e, z_p], -1)), dim=-1
        )
        fused = g[:, 0:1]*z_d + g[:, 1:2]*z_e + g[:, 2:3]*z_p

        return {
            "soh_pred": self.head(fused),
            "fused":    fused,
            "gates":    g,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Registry: name → (class, constructor kwargs)
# ═══════════════════════════════════════════════════════════════════════════════

def get_model(name: str, d_in: int = 3, e_in: int = 3, p_in: int = 4) -> nn.Module:
    """
    Instantiate a benchmark model by name.

    Names:
        "A"   TinyMultimodalMLP
        "B"   SmallMLPFusion
        "C"   CNN1DFusion
        "D"   TCNFusion
        "E"   SmallLSTM-64
        "E2"  SmallLSTM-128
        "F"   GatedFusion
        "G"   CNNGatedFusion
    """
    registry = {
        "A":  lambda: ModelA_TinyMLP(eis_dim=e_in, physics_dim=p_in),
        "B":  lambda: ModelB_SmallMLP(eis_dim=e_in, physics_dim=p_in),
        "C":  lambda: ModelC_CNN1D(disc_channels=d_in, eis_dim=e_in, physics_dim=p_in),
        "D":  lambda: ModelD_TCN(disc_channels=d_in,   eis_dim=e_in, physics_dim=p_in),
        "E":  lambda: ModelE_SmallLSTM(disc_channels=d_in, hidden=64,  eis_dim=e_in, physics_dim=p_in),
        "E2": lambda: ModelE_SmallLSTM(disc_channels=d_in, hidden=128, eis_dim=e_in, physics_dim=p_in),
        "F":  lambda: ModelF_GatedFusion(disc_channels=d_in, eis_dim=e_in, physics_dim=p_in),
        "G":  lambda: ModelG_CNNGated(disc_channels=d_in,   eis_dim=e_in, physics_dim=p_in),
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Available: {list(registry)}")
    return registry[name]()


def all_model_param_counts(d_in=3, e_in=3, p_in=4) -> dict:
    return {n: count_params(get_model(n, d_in, e_in, p_in))
            for n in ["A", "B", "C", "D", "E", "E2", "F", "G"]}


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED MODELS — v2.2
# ═══════════════════════════════════════════════════════════════════════════════
#
# EIS NOTE: NASA EIS is represented by 3 summary scalars [|Z|, Re, Rct].
#   These are NOT a frequency spectrum. CNN/TCN on EIS is a "3-feature CNN
#   control" experiment — do NOT interpret as learning impedance spectrum.
#
# PHYSICS NOTE: Physics features are 4 engineered scalars with no natural
#   temporal axis. CNN/TCN on physics treats them as a short ordered sequence;
#   feature ordering is artificial.

def _small_cnn1d(in_ch: int, out_ch: int = 32, dropout: float = 0.1) -> nn.Module:
    """
    Minimal CNN1D for short feature vectors (EIS=3, Physics=4).
    Input: (B, in_ch, L) where L is the feature dimension treated as length.
    Output after AdaptiveAvgPool: (B, out_ch).
    """
    mid = max(in_ch, 16)
    return nn.Sequential(
        nn.Conv1d(1, mid, kernel_size=2, padding=1),
        nn.ReLU(),
        nn.Conv1d(mid, out_ch, kernel_size=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )


def _small_tcn(out_ch: int = 32, dropout: float = 0.1) -> nn.Module:
    """
    Minimal TCN for short feature vectors (1 channel → out_ch).
    dilations 1, 2 to cover the short sequence.
    """
    return nn.Sequential(
        _TCNBlock(1, 16, kernel=2, dilation=1, dropout=dropout),
        _TCNBlock(16, out_ch, kernel=2, dilation=2, dropout=dropout),
    )


# ── MODEL C2 — CNN1D Full (CNN for all three modalities) ─────────────────────

class ModelC2_CNN1DFull(nn.Module):
    """
    C2 — CNN1DFusion Full.

    Discharge -> CNN1D(3,32,64) -> 64D
    EIS       -> CNN1D control (1,16,32) on 3-feature scalar -> 32D
                 [3-feature EIS CNN control; NOT full EIS spectrum]
    Physics   -> CNN1D control (1,16,32) on 4-feature scalar -> 32D
                 [feature ordering is artificial]
    Concat 128D -> 64 -> 1

    vs C:  EIS/Physics change from MLP → CNN
    """

    def __init__(self, disc_channels: int = 3, eis_dim: int = 3,
                 physics_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        # Discharge: same as Model C
        self.d_cnn = nn.Sequential(
            nn.Conv1d(disc_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # EIS: 3-feature CNN control — (B,3) → (B,1,3) → CNN → 32D
        self.e_cnn = _small_cnn1d(eis_dim,     out_ch=32)
        # Physics: 4-feature CNN control
        self.p_cnn = _small_cnn1d(physics_dim, out_ch=32)

        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _run_cnn(self, module, x):
        # x: (B, L) → (B, 1, L) → module → (B, C, 1) → (B, C)
        return module(x.unsqueeze(1)).squeeze(-1)

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        x = discharge.transpose(1, 2) if discharge.dim() == 3 else discharge.unsqueeze(1)
        d = self.d_cnn(x).squeeze(-1)
        e = self._run_cnn(self.e_cnn, eis)
        p = self._run_cnn(self.p_cnn, physics)
        fused = torch.cat([d, e, p], dim=-1)
        return {"soh_pred": self.head(fused), "fused": fused}


# ── MODEL D2 — TCN Full ───────────────────────────────────────────────────────

class ModelD2_TCNFull(nn.Module):
    """
    D2 — TCNFusion Full.

    Discharge -> TCN(3→32→32→64) dilations 1,2,4 -> 64D
    EIS       -> TCN control (1→16→32) dilations 1,2 on 3-feature -> 32D
                 [3-feature EIS TCN control; NOT full EIS spectrum]
    Physics   -> TCN control (1→16→32) dilations 1,2 on 4-feature -> 32D
                 [feature ordering is artificial]
    Concat 128D -> 64 -> 1

    vs D: EIS/Physics change from MLP → TCN
    """

    def __init__(self, disc_channels: int = 3, eis_dim: int = 3,
                 physics_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        # Discharge: same as Model D
        self.d_tcn = nn.Sequential(
            _TCNBlock(disc_channels, 32, dilation=1, dropout=dropout),
            _TCNBlock(32, 32,           dilation=2, dropout=dropout),
            _TCNBlock(32, 64,           dilation=4, dropout=dropout),
        )
        # EIS: 3-feature TCN control
        self.e_tcn = _small_tcn(out_ch=32, dropout=dropout)
        # Physics: 4-feature TCN control
        self.p_tcn = _small_tcn(out_ch=32, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _run_tcn(self, module, x):
        # x: (B, L) → (B, 1, L) → TCN → (B, C, L') → global avg pool → (B, C)
        return module(x.unsqueeze(1)).mean(dim=-1)

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        x = discharge.transpose(1, 2) if discharge.dim() == 3 else discharge.unsqueeze(1)
        d = self.d_tcn(x).mean(dim=-1)
        e = self._run_tcn(self.e_tcn, eis)
        p = self._run_tcn(self.p_tcn, physics)
        fused = torch.cat([d, e, p], dim=-1)
        return {"soh_pred": self.head(fused), "fused": fused}


# ── MODEL E3 — CNN → LSTM 64 ─────────────────────────────────────────────────

class ModelE3_CNNLSTM64(nn.Module):
    """
    E3 — CNN-LSTM64 Fusion.

    CNN extracts local features across the temporal axis, then LSTM
    captures sequential dependencies on the feature map.

    Discharge: CNN(3→32→64, no pool) → LSTM(64,64) → final h → 64D
    EIS:    MLP → 32D
    Physics: MLP → 32D
    Concat 128D → 64 → 1

    vs E: adds CNN preprocessing stage before LSTM
    """

    def __init__(self, disc_channels: int = 3, eis_dim: int = 3,
                 physics_dim: int = 4, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        # CNN without final pooling — keeps temporal dim for LSTM
        self.cnn = nn.Sequential(
            nn.Conv1d(disc_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            # NO AdaptiveAvgPool — keep temporal dimension
        )
        self.lstm  = nn.LSTM(64, hidden, num_layers=1, batch_first=True)
        self.e_enc = _mlp(eis_dim,     32, dropout=dropout)
        self.p_enc = _mlp(physics_dim, 32, dropout=dropout)
        fused_in   = hidden + 32 + 32
        self.head  = nn.Sequential(
            nn.Linear(fused_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        x = discharge.transpose(1, 2) if discharge.dim() == 3 else discharge.unsqueeze(1)
        # x: (B, 3, T) → CNN → (B, 64, T) → transpose → (B, T, 64)
        c = self.cnn(x).transpose(1, 2)
        _, (h_n, _) = self.lstm(c)
        d = h_n[-1]                     # (B, hidden)
        e = self.e_enc(eis)
        p = self.p_enc(physics)
        fused = torch.cat([d, e, p], dim=-1)
        return {"soh_pred": self.head(fused), "fused": fused}


# ── MODEL E4 — CNN → LSTM 128 ────────────────────────────────────────────────

class ModelE4_CNNLSTM128(nn.Module):
    """
    E4 — CNN-LSTM128 Fusion.

    Same as E3 with LSTM hidden=128.
    vs E2: adds CNN preprocessing before the 128-unit LSTM.
    vs E3: increases LSTM capacity from 64 to 128.
    """

    def __init__(self, disc_channels: int = 3, eis_dim: int = 3,
                 physics_dim: int = 4, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(disc_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.lstm  = nn.LSTM(64, hidden, num_layers=1, batch_first=True)
        self.e_enc = _mlp(eis_dim,     32, dropout=dropout)
        self.p_enc = _mlp(physics_dim, 32, dropout=dropout)
        fused_in   = hidden + 32 + 32
        self.head  = nn.Sequential(
            nn.Linear(fused_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        x = discharge.transpose(1, 2) if discharge.dim() == 3 else discharge.unsqueeze(1)
        c = self.cnn(x).transpose(1, 2)    # (B, T, 64)
        _, (h_n, _) = self.lstm(c)
        d = h_n[-1]
        e = self.e_enc(eis)
        p = self.p_enc(physics)
        fused = torch.cat([d, e, p], dim=-1)
        return {"soh_pred": self.head(fused), "fused": fused}


# ── MODEL E5 — Full Hybrid CNN-LSTM ──────────────────────────────────────────

class ModelE5_FullHybrid(nn.Module):
    """
    E5 — Full Hybrid CNN-LSTM.

    Discharge: CNN → LSTM64 → 64D
    EIS:    3-feature CNN control → 32D
    Physics: 4-feature CNN control → 32D
    Concat 128D → 64 → 1

    vs E3: replaces EIS/Physics MLP with CNN (same as C2/D2 EIS/Physics branch)
    vs C2: adds LSTM after discharge CNN instead of just pooling

    EIS/Physics CNN caveat: same as C2 — 3-feature / 4-feature control, not
    true spectrum CNN.
    """

    def __init__(self, disc_channels: int = 3, eis_dim: int = 3,
                 physics_dim: int = 4, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        # Discharge: CNN → LSTM
        self.d_cnn = nn.Sequential(
            nn.Conv1d(disc_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden, num_layers=1, batch_first=True)
        # EIS + Physics: 3-feature / 4-feature CNN control
        self.e_cnn = _small_cnn1d(eis_dim,     out_ch=32)
        self.p_cnn = _small_cnn1d(physics_dim, out_ch=32)

        fused_in = hidden + 32 + 32
        self.head = nn.Sequential(
            nn.Linear(fused_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _run_cnn(self, module, x):
        return module(x.unsqueeze(1)).squeeze(-1)

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        x = discharge.transpose(1, 2) if discharge.dim() == 3 else discharge.unsqueeze(1)
        c = self.d_cnn(x).transpose(1, 2)
        _, (h_n, _) = self.lstm(c)
        d = h_n[-1]
        e = self._run_cnn(self.e_cnn, eis)
        p = self._run_cnn(self.p_cnn, physics)
        fused = torch.cat([d, e, p], dim=-1)
        return {"soh_pred": self.head(fused), "fused": fused}


# ── MODEL G2 — Full CNN Gated Fusion ─────────────────────────────────────────

class ModelG2_FullCNNGated(nn.Module):
    """
    G2 — Full CNN Gated Fusion.

    CNN for all three modalities (vs G which uses MLP for EIS/Physics).
    Then same gated fusion as G/F.

    Discharge: CNN1D → 64D  (same as G)
    EIS:    3-feature CNN control → project to 64D
    Physics: 4-feature CNN control → project to 64D
    Gate:   softmax(Linear(192D)) → 3 weights
    Fused:  weighted sum → 64D
    Head:   64 → 32 → 1

    EIS/Physics CNN caveat: 3-feature / 4-feature control experiment.
    Gate values logged for analysis; do NOT interpret as causal importance.
    """

    def __init__(self, disc_channels: int = 3, eis_dim: int = 3,
                 physics_dim: int = 4, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Discharge CNN
        self.d_enc = nn.Sequential(
            nn.Conv1d(disc_channels, 32, 5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, embed_dim, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # EIS: 3-feature CNN → 32D → project to embed_dim
        self.e_cnn  = _small_cnn1d(eis_dim,     out_ch=32)
        self.e_proj = nn.Linear(32, embed_dim)
        # Physics: 4-feature CNN → 32D → project to embed_dim
        self.p_cnn  = _small_cnn1d(physics_dim, out_ch=32)
        self.p_proj = nn.Linear(32, embed_dim)

        self.gate = nn.Linear(embed_dim * 3, 3)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def _run_cnn_proj(self, cnn, proj, x):
        return F.relu(proj(cnn(x.unsqueeze(1)).squeeze(-1)))

    def forward(self, discharge, eis, physics, **kw) -> Dict[str, torch.Tensor]:
        x = discharge.transpose(1, 2) if discharge.dim() == 3 else discharge.unsqueeze(1)
        z_d = self.d_enc(x).squeeze(-1)
        z_e = self._run_cnn_proj(self.e_cnn, self.e_proj, eis)
        z_p = self._run_cnn_proj(self.p_cnn, self.p_proj, physics)

        g = torch.softmax(
            self.gate(torch.cat([z_d, z_e, z_p], -1)), dim=-1
        )
        fused = g[:, 0:1]*z_d + g[:, 1:2]*z_e + g[:, 2:3]*z_p

        return {
            "soh_pred": self.head(fused),
            "fused":    fused,
            "gates":    g,
        }


# ── Update registry ───────────────────────────────────────────────────────────

def get_model(name: str, d_in: int = 3, e_in: int = 3, p_in: int = 4) -> nn.Module:
    """
    Instantiate a benchmark model by name.  Includes v2.2 extended models.
    """
    registry = {
        # v1 models (unchanged)
        "A":  lambda: ModelA_TinyMLP(eis_dim=e_in, physics_dim=p_in),
        "B":  lambda: ModelB_SmallMLP(eis_dim=e_in, physics_dim=p_in),
        "C":  lambda: ModelC_CNN1D(disc_channels=d_in, eis_dim=e_in, physics_dim=p_in),
        "D":  lambda: ModelD_TCN(disc_channels=d_in,   eis_dim=e_in, physics_dim=p_in),
        "E":  lambda: ModelE_SmallLSTM(disc_channels=d_in, hidden=64,  eis_dim=e_in, physics_dim=p_in),
        "E2": lambda: ModelE_SmallLSTM(disc_channels=d_in, hidden=128, eis_dim=e_in, physics_dim=p_in),
        "F":  lambda: ModelF_GatedFusion(disc_channels=d_in, eis_dim=e_in, physics_dim=p_in),
        "G":  lambda: ModelG_CNNGated(disc_channels=d_in,   eis_dim=e_in, physics_dim=p_in),
        # v2.2 extended models
        "C2": lambda: ModelC2_CNN1DFull(disc_channels=d_in, eis_dim=e_in, physics_dim=p_in),
        "D2": lambda: ModelD2_TCNFull(disc_channels=d_in,   eis_dim=e_in, physics_dim=p_in),
        "E3": lambda: ModelE3_CNNLSTM64(disc_channels=d_in, eis_dim=e_in, physics_dim=p_in),
        "E4": lambda: ModelE4_CNNLSTM128(disc_channels=d_in,eis_dim=e_in, physics_dim=p_in),
        "E5": lambda: ModelE5_FullHybrid(disc_channels=d_in, eis_dim=e_in, physics_dim=p_in),
        "G2": lambda: ModelG2_FullCNNGated(disc_channels=d_in,eis_dim=e_in, physics_dim=p_in),
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(registry)}")
    return registry[name]()


def all_model_param_counts(d_in=3, e_in=3, p_in=4) -> dict:
    all_names = ["A","B","C","D","E","E2","F","G","C2","D2","E3","E4","E5","G2"]
    return {n: count_params(get_model(n, d_in, e_in, p_in)) for n in all_names}
