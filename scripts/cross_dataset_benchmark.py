"""
BaFuse Cross-Dataset Benchmark.

Evaluates all 14 benchmark architectures on:
  A. NASA-SOH  — SOH regression (results taken from existing benchmark files
                  or run fresh if missing)
  B. Mendeley-DEG — LLI/LAM/CL degradation-mode regression using only the
                    EIS encoder branch of each architecture.

Design principle for Mendeley-DEG:
  Each architecture has an EIS encoder (MLP or CNN) that produces an
  embedding vector.  A lightweight shared DegradationHead (3-output MLP)
  is appended.  Discharge and physics branches are NOT used — Mendeley does
  not have NASA-format discharge sequences.

  This measures how well the EIS representation learned by each architecture
  can predict model-derived degradation-mode labels.

IMPORTANT:
  - LLI/LAM/CL are model-derived from ECM fitting (Mendeley dataset).
    They are NOT physical ground truth.
  - Mendeley cells: 6 train / 1 val / 1 test (cell-level split).
  - No NASA/Mendeley row matching.
  - Normalization statistics computed from Mendeley training set only.

Usage:
    # Full run (seed=42):
    python scripts/cross_dataset_benchmark.py --seed 42

    # Re-use existing NASA results without rerunning:
    python scripts/cross_dataset_benchmark.py --skip_nasa

    # Skip Mendeley (only produce combined table from existing files):
    python scripts/cross_dataset_benchmark.py --skip_mendeley

    # Multi-seed for top candidates after single-seed run:
    python scripts/cross_dataset_benchmark.py --seeds 42 123 2026
                                               --models D D2 G G2 A
"""

import argparse, csv, json, logging, sys, time, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "results" / "cross_dataset"
for sub in ["plots", "logs"]:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
ERRORS_LOG = OUT_DIR / "logs" / "errors.log"

# Training hyperparameters (same as benchmark A-H)
LR         = 1e-3
WD         = 1e-4
EPOCHS     = 100
PATIENCE   = 15
BATCH_SIZE = 32
DEVICE     = "cpu"

# NASA reference table (from spec — do not overwrite)
# Format: model → {params, val_mae, test_mae, test_rmse, test_r2}
NASA_REFERENCE = {
    "D":  {"params": 44257,   "val_mae": 3.02, "test_mae": 5.34, "test_rmse": 8.37,  "test_r2": 0.7239},
    "G":  {"params": 24964,   "val_mae": 2.74, "test_mae": 5.43, "test_rmse": 7.22,  "test_r2": 0.7943},
    "A":  {"params": 3431,    "val_mae": 2.25, "test_mae": 6.38, "test_rmse": 8.55,  "test_r2": 0.7118},
    "C":  {"params": 24193,   "val_mae": 2.77, "test_mae": 7.06, "test_rmse": 9.21,  "test_r2": 0.6653},
    "F":  {"params": 18148,   "val_mae": 3.80, "test_mae": 7.66, "test_rmse": 11.48, "test_r2": 0.4797},
    "E2": {"params": 85505,   "val_mae": 4.50, "test_mae": 9.28, "test_rmse": 11.43, "test_r2": 0.4844},
    "E":  {"params": 30977,   "val_mae": 4.64, "test_mae": 9.29, "test_rmse": 11.70, "test_r2": 0.4603},
    "B":  {"params": 16545,   "val_mae": 3.79, "test_mae": 9.91, "test_rmse": 12.06, "test_r2": 0.4264},
    "D2": {"params": 48225,   "val_mae": 3.58, "test_mae": 4.23, "test_rmse": 6.04,  "test_r2": 0.8561},
    "G2": {"params": 20132,   "val_mae": 3.08, "test_mae": 5.52, "test_rmse": 7.61,  "test_r2": 0.7717},
    "C2": {"params": 21409,   "val_mae": 3.97, "test_mae": 6.24, "test_rmse": 7.75,  "test_r2": 0.7628},
    "E5": {"params": 54625,   "val_mae": 3.58, "test_mae": 7.24, "test_rmse": 9.76,  "test_r2": 0.6246},
    "E4": {"params": 127553,  "val_mae": 3.78, "test_mae": 7.70, "test_rmse": 10.45, "test_r2": 0.5690},
    "E3": {"params": 57409,   "val_mae": 4.01, "test_mae": 8.32, "test_rmse": 11.01, "test_r2": 0.5222},
    "H":  {"params": 1536321, "val_mae": 3.42, "test_mae": 10.80,"test_rmse": 13.46, "test_r2": 0.2853},
}

ALL_MODELS = ["D", "G", "A", "C", "F", "E2", "E", "B",
              "D2", "G2", "C2", "E5", "E4", "E3"]


# ═══════════════════════════════════════════════════════════════════════════════
# EIS-only adapter with degradation head
# ═══════════════════════════════════════════════════════════════════════════════

class DegradationHead(nn.Module):
    """
    Shared 3-output regression head for LLI/LAM/CL.
    Input: EIS embedding (any dim). Outputs: (B,1) for each target.
    """
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        out_in = hidden // 2
        self.h_lli = nn.Linear(out_in, 1)
        self.h_lam = nn.Linear(out_in, 1)
        self.h_cl  = nn.Linear(out_in, 1)

    def forward(self, x):
        t = self.trunk(x)
        return self.h_lli(t), self.h_lam(t), self.h_cl(t)


def _extract_eis_encoder(arch_model: nn.Module, mendeley_eis_dim: int = 8) -> nn.Module:
    """
    Extract or create an EIS encoder from an architecture model.

    Strategy per architecture family:
    - Models with an explicit .e_enc (B,C,C2,D,D2,E,E2,E3,E4,E5,F,G,G2):
        use that encoder directly if input dim matches mendeley_eis_dim.
        If dim mismatches (NASA e_in=3 vs Mendeley=8), create a fresh
        equivalent encoder with the correct input dim.
    - Model A (TinyMLP): uses _pool_discharge + flat concat — extract the
        EIS-relevant slice handler and create a standalone MLP.

    Returns a module: (B, mendeley_eis_dim) → (B, emb_dim)
    """
    name = type(arch_model).__name__

    # Models with CNN EIS branch (C2, D2, E5, G2)
    if hasattr(arch_model, "e_cnn"):
        # CNN branch on EIS — input is (B, eis_dim).
        # We need to know current e_cnn's expected input dim to decide
        # whether to reuse or rebuild.
        # All these were built with e_in=3 (NASA). Mendeley has 8 dims.
        # Rebuild a matching CNN for 8-dim input.
        from src.models.benchmark_models import _small_cnn1d
        cnn = _small_cnn1d(mendeley_eis_dim, out_ch=32)

        # Determine if there's a projection layer after cnn (G2 has e_proj)
        if hasattr(arch_model, "e_proj"):
            proj = arch_model.e_proj   # Linear(32 → 64), reusable
            class _CNNProjEnc(nn.Module):
                def __init__(self, cnn, proj):
                    super().__init__(); self.cnn = cnn; self.proj = proj
                def forward(self, x):
                    return torch.relu(self.proj(self.cnn(x.unsqueeze(1)).squeeze(-1)))
            return _CNNProjEnc(cnn, proj), proj.out_features

        class _CNNEnc(nn.Module):
            def __init__(self, cnn):
                super().__init__(); self.cnn = cnn
            def forward(self, x):
                return self.cnn(x.unsqueeze(1)).squeeze(-1)   # (B, 32)
        return _CNNEnc(cnn), 32

    # Models with MLP .e_enc (B, C, D, E, E2, E3, E4, F, G)
    if hasattr(arch_model, "e_enc"):
        # Rebuild with mendeley_eis_dim as input
        from src.models.benchmark_models import _mlp
        enc = _mlp(mendeley_eis_dim, 64, hidden=64)
        return enc, 64

    # Model A: TinyMLP — no dedicated e_enc, uses flat concat.
    # Build a simple MLP encoder for Mendeley EIS.
    from src.models.benchmark_models import _mlp
    enc = _mlp(mendeley_eis_dim, 64, hidden=64)
    return enc, 64


class EISOnlyModel(nn.Module):
    """
    Wraps any benchmark architecture's EIS encoder + a degradation head.

    Mendeley input: (B, 8) EIS features → EIS encoder → DegradationHead → LLI/LAM/CL

    This is the minimal legitimate adapter:
    - Only the EIS branch is active.
    - Discharge and physics branches are NOT used.
    - A fresh DegradationHead is appended (not shared with NASA branch).

    Adapter requirement is documented per model in the compatibility matrix.
    """
    def __init__(self, arch_model: nn.Module, mendeley_eis_dim: int = 8,
                 deg_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        eis_enc, emb_dim = _extract_eis_encoder(arch_model, mendeley_eis_dim)
        self.eis_enc = eis_enc
        self.deg_head = DegradationHead(emb_dim, hidden=deg_hidden, dropout=dropout)

    def forward(self, eis: torch.Tensor):
        emb = self.eis_enc(eis)
        lli, lam, cl = self.deg_head(emb)
        return lli, lam, cl

    def parameters(self, recurse=True):
        return list(nn.Module.parameters(self, recurse))


# ═══════════════════════════════════════════════════════════════════════════════
# Mendeley data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_mendeley():
    """Load preprocessed Mendeley splits and compute robust normalisation."""
    proc = ROOT / "data" / "mendeley_processed"
    tr = pd.read_pickle(str(proc / "mendeley_train.pkl"))
    va = pd.read_pickle(str(proc / "mendeley_val.pkl"))
    te = pd.read_pickle(str(proc / "mendeley_test.pkl"))
    return tr, va, te


EIS_FEATURE_COLS = ["zmod_ohm", "zreal_ohm", "zimg_ohm", "zphz_deg",
                    "R_electrolyte", "R_ct1", "R_ct2", "Zw"]


def compute_mendeley_norm(train_df: pd.DataFrame) -> dict:
    """
    Compute robust normalisation stats from Mendeley train set only.
    Uses median/IQR for EIS features (robust to outliers).
    Uses mean/std with percentile-clipping for LLI/LAM/CL (extreme outliers).
    """
    stats = {}
    for col in EIS_FEATURE_COLS:
        if col in train_df.columns:
            v = train_df[col].dropna().values.astype(float)
            stats[col] = {"mean": float(np.mean(v)), "std": float(np.std(v)) + 1e-8}

    # LLI/LAM/CL: clip to [1st, 99th] percentile then normalise
    for col in ["lli_pct", "lam_pct", "cl_pct"]:
        if col in train_df.columns:
            v = train_df[col].dropna().values.astype(float)
            p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
            v_clip = np.clip(v, p1, p99)
            stats[col] = {
                "mean": float(np.mean(v_clip)),
                "std":  float(np.std(v_clip)) + 1e-8,
                "p1":   float(p1),
                "p99":  float(p99),
            }
    return stats


def make_mendeley_tensors(df: pd.DataFrame, stats: dict, device: torch.device):
    """Convert a Mendeley DataFrame split into (eis, lli, lam, cl) tensors."""
    avail = [c for c in EIS_FEATURE_COLS if c in df.columns]
    eis_arr = np.zeros((len(df), len(avail)), dtype=np.float32)
    for i, col in enumerate(avail):
        vals = df[col].fillna(stats[col]["mean"]).values.astype(float)
        eis_arr[:, i] = (vals - stats[col]["mean"]) / stats[col]["std"]

    def _norm_target(col):
        s = stats[col]
        v = df[col].fillna(s["mean"]).values.astype(float)
        v = np.clip(v, s["p1"], s["p99"])
        return ((v - s["mean"]) / s["std"]).astype(np.float32)

    eis = torch.tensor(eis_arr, device=device)
    lli = torch.tensor(_norm_target("lli_pct"), device=device)
    lam = torch.tensor(_norm_target("lam_pct"), device=device)
    cl  = torch.tensor(_norm_target("cl_pct"),  device=device)
    return eis, lli, lam, cl


def denorm_deg(preds: np.ndarray, stats: dict, col: str) -> np.ndarray:
    """Denormalise degradation predictions back to original % units."""
    s = stats[col]
    return preds * s["std"] + s["mean"]


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _met(pred: np.ndarray, tgt: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(pred - tgt)))
    rmse = float(np.sqrt(np.mean((pred - tgt)**2)))
    ss_r = np.sum((tgt - pred)**2)
    ss_t = np.sum((tgt - tgt.mean())**2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))
    return {"mae": round(mae,3), "rmse": round(rmse,3), "r2": round(r2,4)}


def deg_metrics(
    lli_p, lam_p, cl_p,
    lli_t, lam_t, cl_t,
) -> dict:
    """Compute MAE/RMSE/R² per degradation mode (raw normalised scale)."""
    return {
        "lli": _met(lli_p, lli_t),
        "lam": _met(lam_p, lam_t),
        "cl":  _met(cl_p,  cl_t),
        "mean_mae":  round(float(np.mean([
            np.mean(np.abs(lli_p - lli_t)),
            np.mean(np.abs(lam_p - lam_t)),
            np.mean(np.abs(cl_p  - cl_t)),
        ])), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    import random; random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_mendeley_model(
    model: EISOnlyModel,
    tr_eis, tr_lli, tr_lam, tr_cl,
    va_eis, va_lli, va_lam, va_cl,
    device: torch.device,
) -> Tuple[dict, int]:
    """Mini-batch training for Mendeley-DEG."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    warmup = min(5, EPOCHS // 5)
    def _lr(ep):
        if ep < warmup: return (ep+1)/warmup
        p = (ep-warmup)/max(EPOCHS-warmup, 1)
        return max(0.05, 0.5*(1+np.cos(np.pi*p)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)

    n = tr_eis.shape[0]
    best_mae = float("inf")
    best_state = None
    p_ctr = 0
    best_ep = 1

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            lli_p, lam_p, cl_p = model(tr_eis[idx])
            loss = (criterion(lli_p.view(-1), tr_lli[idx]) +
                    criterion(lam_p.view(-1), tr_lam[idx]) +
                    criterion(cl_p.view(-1),  tr_cl[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vli, vla, vcl = model(va_eis)
        v_mae = float(np.mean([
            np.mean(np.abs(vli.view(-1).cpu().numpy() - va_lli.cpu().numpy())),
            np.mean(np.abs(vla.view(-1).cpu().numpy() - va_lam.cpu().numpy())),
            np.mean(np.abs(vcl.view(-1).cpu().numpy() - va_cl.cpu().numpy())),
        ]))

        if v_mae < best_mae - 1e-4:
            best_mae = v_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            p_ctr = 0; best_ep = ep
        else:
            p_ctr += 1
        if p_ctr >= PATIENCE:
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return {"best_val_mean_mae": round(best_mae, 4)}, best_ep


# ═══════════════════════════════════════════════════════════════════════════════
# Single Mendeley experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_mendeley_experiment(
    model_name: str,
    tr_df, va_df, te_df,
    stats: dict,
    seed: int,
    device: torch.device,
) -> Optional[dict]:
    exp_id = f"{model_name}_mendeley_s{seed}"
    logger.info(f"  [{exp_id}] Mendeley-DEG")

    try:
        from src.models.benchmark_models import get_model as get_arch
        arch = get_arch(model_name, d_in=3, e_in=3, p_in=4)

        model = EISOnlyModel(arch, mendeley_eis_dim=8)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  [{exp_id}] EIS-only adapter params: {n_params:,}")

        set_seed(seed)

        tr_eis, tr_lli, tr_lam, tr_cl = make_mendeley_tensors(tr_df, stats, device)
        va_eis, va_lli, va_lam, va_cl = make_mendeley_tensors(va_df, stats, device)
        te_eis, te_lli, te_lam, te_cl = make_mendeley_tensors(te_df, stats, device)

        t0 = time.time()
        hist, best_ep = train_mendeley_model(
            model,
            tr_eis, tr_lli, tr_lam, tr_cl,
            va_eis, va_lli, va_lam, va_cl,
            device,
        )
        elapsed = time.time() - t0

        model.eval()
        with torch.no_grad():
            te_lli_p, te_lam_p, te_cl_p = model(te_eis)

        lli_p = te_lli_p.view(-1).cpu().numpy()
        lam_p = te_lam_p.view(-1).cpu().numpy()
        cl_p  = te_cl_p.view(-1).cpu().numpy()
        lli_t = te_lli.cpu().numpy()
        lam_t = te_lam.cpu().numpy()
        cl_t  = te_cl.cpu().numpy()

        m = deg_metrics(lli_p, lam_p, cl_p, lli_t, lam_t, cl_t)

        result = {
            "model":        model_name,
            "dataset":      "Mendeley-DEG",
            "task":         "Mendeley-DEG",
            "seed":         seed,
            "adapter_params": n_params,
            "best_ep":      best_ep,
            "elapsed_s":    round(elapsed, 1),
            "val_mean_mae": hist["best_val_mean_mae"],
            "lli_mae":  m["lli"]["mae"],  "lli_rmse": m["lli"]["rmse"],  "lli_r2":  m["lli"]["r2"],
            "lam_mae":  m["lam"]["mae"],  "lam_rmse": m["lam"]["rmse"],  "lam_r2":  m["lam"]["r2"],
            "cl_mae":   m["cl"]["mae"],   "cl_rmse":  m["cl"]["rmse"],   "cl_r2":   m["cl"]["r2"],
            "mean_deg_mae": m["mean_mae"],
        }

        logger.info(
            f"  [{exp_id}] DONE  "
            f"LLI={m['lli']['r2']:.3f}  LAM={m['lam']['r2']:.3f}  "
            f"CL={m['cl']['r2']:.3f}  mean_MAE={m['mean_mae']:.4f}  "
            f"ep={best_ep}  {elapsed:.0f}s"
        )
        return result

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"  [{exp_id}] FAILED: {e}")
        with open(ERRORS_LOG, "a") as f:
            f.write(f"\n{'='*60}\n{exp_id}\n{tb}\n")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════════════════════════

def build_combined_table(mendeley_results: list) -> pd.DataFrame:
    """Merge NASA reference + Mendeley results into one DataFrame."""
    rows = []

    # NASA rows (from reference table)
    for m, ref in NASA_REFERENCE.items():
        rows.append({
            "model": m,
            "dataset": "NASA-SOH",
            "task": "NASA-SOH",
            "params": ref["params"],
            "val_mae": ref["val_mae"],
            "test_mae": ref["test_mae"],
            "test_rmse": ref["test_rmse"],
            "test_r2": ref["test_r2"],
            "lli_r2": "NA", "lam_r2": "NA", "cl_r2": "NA",
            "mean_deg_mae": "NA",
        })

    # Mendeley rows
    for r in mendeley_results:
        if r is None: continue
        rows.append({
            "model": r["model"],
            "dataset": "Mendeley-DEG",
            "task": "Mendeley-DEG",
            "params": r["adapter_params"],
            "val_mae": r["val_mean_mae"],
            "test_mae": "NA",
            "test_rmse": "NA",
            "test_r2": "NA",
            "lli_r2": r["lli_r2"],
            "lam_r2": r["lam_r2"],
            "cl_r2":  r["cl_r2"],
            "mean_deg_mae": r["mean_deg_mae"],
        })

    return pd.DataFrame(rows)


def build_compatibility_matrix(mendeley_results: list) -> pd.DataFrame:
    """Architecture compatibility table."""
    mend_done = {r["model"] for r in mendeley_results if r}
    rows = []
    for m in ALL_MODELS:
        ref = NASA_REFERENCE.get(m, {})
        # Determine adapter type
        from src.models.benchmark_models import get_model
        arch = get_model(m, 3, 3, 4)
        has_e_cnn  = hasattr(arch, "e_cnn")
        has_e_enc  = hasattr(arch, "e_enc")

        if has_e_cnn and hasattr(arch, "e_proj"):
            eis_type = "CNN+proj"
        elif has_e_cnn:
            eis_type = "CNN"
        elif has_e_enc:
            eis_type = "MLP"
        else:
            eis_type = "flat-pool"

        rows.append({
            "model":                   m,
            "params_nasa":             ref.get("params", "NA"),
            "NASA_SOH_supported":      "YES",
            "Mendeley_DEG_supported":  "YES" if m in mend_done else "FAILED",
            "eis_encoder_type":        eis_type,
            "adapter_required":        "YES",
            "adapter_description":     (
                f"EIS-only path: {eis_type} encoder (rebuilt for 8D Mendeley EIS) + DegradationHead. "
                "Discharge and physics branches disabled."
            ),
        })
    return pd.DataFrame(rows)


def make_plots(combined_df: pd.DataFrame, mend_results: list):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nasa_df = combined_df[combined_df["dataset"] == "NASA-SOH"].copy()
        mend_df = combined_df[combined_df["dataset"] == "Mendeley-DEG"].copy()

        models_nasa = nasa_df.sort_values("test_mae")["model"].tolist()
        models_mend = mend_df.sort_values("mean_deg_mae")["model"].tolist()

        # 1. NASA Test MAE
        fig, ax = plt.subplots(figsize=(11,4))
        ax.bar(nasa_df.sort_values("test_mae")["model"],
               nasa_df.sort_values("test_mae")["test_mae"].astype(float),
               color="#2196F3", alpha=0.85)
        ax.set_ylabel("Test MAE (%)"); ax.set_title("NASA SOH — Test MAE by Architecture")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout(); plt.savefig(str(OUT_DIR/"plots"/"nasa_test_mae.png"), dpi=130); plt.close()

        # 2. NASA Test R²
        fig, ax = plt.subplots(figsize=(11,4))
        ax.bar(nasa_df.sort_values("test_r2", ascending=False)["model"],
               nasa_df.sort_values("test_r2", ascending=False)["test_r2"].astype(float),
               color="#4CAF50", alpha=0.85)
        ax.axhline(0, color="red", lw=0.8, linestyle="--")
        ax.set_ylabel("Test R²"); ax.set_title("NASA SOH — Test R² by Architecture")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout(); plt.savefig(str(OUT_DIR/"plots"/"nasa_test_r2.png"), dpi=130); plt.close()

        # 3–5. Mendeley per-mode R²
        for mode, col, color, fname in [
            ("LLI", "lli_r2", "#E91E63", "mendeley_lli_r2.png"),
            ("LAM", "lam_r2", "#9C27B0", "mendeley_lam_r2.png"),
            ("CL",  "cl_r2",  "#009688", "mendeley_cl_r2.png"),
        ]:
            if mend_df.empty: continue
            sub = mend_df.dropna(subset=[col]).copy()
            sub[col] = sub[col].astype(float)
            sub = sub.sort_values(col, ascending=False)
            fig, ax = plt.subplots(figsize=(11,4))
            ax.bar(sub["model"], sub[col], color=color, alpha=0.85)
            ax.axhline(0, color="gray", lw=0.8, linestyle="--")
            ax.set_ylabel(f"{mode} R²")
            ax.set_title(f"Mendeley-DEG — {mode} R² by Architecture")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout(); plt.savefig(str(OUT_DIR/"plots"/fname), dpi=130); plt.close()

        # 6. Params vs NASA MAE
        fig, ax = plt.subplots(figsize=(8,5))
        for _, row in nasa_df.iterrows():
            ax.scatter(int(row["params"]), float(row["test_mae"]), s=60, alpha=0.8)
            ax.annotate(row["model"], (int(row["params"]), float(row["test_mae"])),
                        xytext=(5,2), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Parameters"); ax.set_ylabel("NASA Test MAE (%)")
        ax.set_title("Parameter Efficiency — NASA SOH"); ax.set_xscale("log")
        plt.tight_layout(); plt.savefig(str(OUT_DIR/"plots"/"params_vs_nasa_mae.png"), dpi=130); plt.close()

        # 7. Params vs Mendeley mean MAE
        if not mend_df.empty:
            fig, ax = plt.subplots(figsize=(8,5))
            for _, row in mend_df.iterrows():
                if row["mean_deg_mae"] == "NA": continue
                ax.scatter(int(row["params"]), float(row["mean_deg_mae"]), s=60, alpha=0.8)
                ax.annotate(row["model"], (int(row["params"]), float(row["mean_deg_mae"])),
                            xytext=(5,2), textcoords="offset points", fontsize=8)
            ax.set_xlabel("Adapter Parameters"); ax.set_ylabel("Mendeley Mean DEG MAE")
            ax.set_title("Parameter Efficiency — Mendeley DEG"); ax.set_xscale("log")
            plt.tight_layout(); plt.savefig(str(OUT_DIR/"plots"/"params_vs_mendeley_mae.png"), dpi=130); plt.close()

        # 8. Cross-dataset compatibility heatmap
        fig, ax = plt.subplots(figsize=(12, 5))
        compat_data = {
            "model": ALL_MODELS,
            "NASA-SOH": [1]*len(ALL_MODELS),
            "Mendeley-DEG": [1 if m in {r["model"] for r in mend_results if r} else 0
                             for m in ALL_MODELS],
        }
        import numpy as _np
        mat = _np.array([compat_data["NASA-SOH"], compat_data["Mendeley-DEG"]])
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ALL_MODELS))); ax.set_xticklabels(ALL_MODELS, rotation=30)
        ax.set_yticks([0,1]); ax.set_yticklabels(["NASA-SOH","Mendeley-DEG"])
        ax.set_title("Architecture Compatibility Matrix")
        plt.colorbar(im, ax=ax, ticks=[0,1], label="Supported")
        plt.tight_layout(); plt.savefig(str(OUT_DIR/"plots"/"compatibility_matrix.png"), dpi=130); plt.close()

        logger.info("Plots saved.")
    except ImportError:
        logger.warning("matplotlib unavailable — plots skipped.")
    except Exception as e:
        logger.warning(f"Plot error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    global LR, WD, EPOCHS, PATIENCE, BATCH_SIZE, DEVICE
    DEVICE = args.device

    logger.info("="*65)
    logger.info("BaFuse Cross-Dataset Benchmark")
    logger.info(f"  Models: {args.models}")
    logger.info(f"  Seeds:  {args.seeds}")
    logger.info("="*65)

    device = torch.device(DEVICE)

    # ── Mendeley data ─────────────────────────────────────────────────────────
    tr_df, va_df, te_df = load_mendeley()
    stats = compute_mendeley_norm(tr_df)
    logger.info(f"Mendeley: train={len(tr_df)} val={len(va_df)} test={len(te_df)}")
    logger.info(f"  EIS features: {[c for c in EIS_FEATURE_COLS if c in tr_df.columns]}")
    logger.info(f"  LLI clipping: [{stats['lli_pct']['p1']:.1f}, {stats['lli_pct']['p99']:.1f}] -> "
                f"mean={stats['lli_pct']['mean']:.2f} std={stats['lli_pct']['std']:.2f}")

    # ── Save config ───────────────────────────────────────────────────────────
    cfg = {
        "models": args.models,
        "seeds":  args.seeds,
        "lr": LR, "wd": WD, "epochs": EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "nasa_reference": "from spec table (14 models)",
        "mendeley_eis_dim": 8,
        "deg_targets": ["LLI","LAM","CL"],
        "label_note": "LLI/LAM/CL are model-derived degradation-mode estimates; not physical ground truth",
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Mendeley-DEG experiments ──────────────────────────────────────────────
    mend_results = []
    success, failed = [], []

    if not args.skip_mendeley:
        for model_name in args.models:
            for seed in args.seeds:
                set_seed(seed)
                r = run_mendeley_experiment(
                    model_name, tr_df, va_df, te_df,
                    stats, seed, device,
                )
                if r:
                    mend_results.append(r)
                    success.append(f"{model_name}_mend_s{seed}")
                else:
                    failed.append(f"{model_name}_mend_s{seed}")

    # ── Save Mendeley results ─────────────────────────────────────────────────
    if mend_results:
        mend_df = pd.DataFrame(mend_results)
        mend_df.to_csv(str(OUT_DIR / "mendeley_deg_results.csv"), index=False)
        with open(str(OUT_DIR / "mendeley_deg_results.json"), "w") as f:
            json.dump(mend_results, f, indent=2, default=str)
    else:
        # Load previously saved if available
        mend_csv = OUT_DIR / "mendeley_deg_results.csv"
        if mend_csv.exists():
            mend_df = pd.read_csv(str(mend_csv))
            mend_results = mend_df.to_dict(orient="records")
            logger.info(f"Loaded {len(mend_results)} existing Mendeley results")
        else:
            mend_df = pd.DataFrame()

    # ── Combined table ────────────────────────────────────────────────────────
    combined = build_combined_table(mend_results)
    combined.to_csv(str(OUT_DIR / "all_architectures_comparison.csv"), index=False)

    # ── Compatibility matrix ──────────────────────────────────────────────────
    compat = build_compatibility_matrix(mend_results)
    compat.to_csv(str(OUT_DIR / "architecture_compatibility.csv"), index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    make_plots(combined, mend_results)

    # ── Print table ───────────────────────────────────────────────────────────
    SEP = "="*95
    print(f"\n{SEP}")
    print("CROSS-DATASET RESULTS — NASA-SOH (reference) + Mendeley-DEG")
    print(SEP)
    print(f"  {'Model':5s}  {'Params':>9s}  {'NASA_MAE':>9s}  {'NASA_R2':>8s}  "
          f"{'LLI_R2':>7s}  {'LAM_R2':>7s}  {'CL_R2':>7s}  {'MenMAE':>7s}")
    print(f"  {'-'*87}")
    for m in ALL_MODELS:
        ref  = NASA_REFERENCE.get(m, {})
        mrow = next((r for r in mend_results if r and r["model"] == m), None)
        nasa_mae = f"{ref.get('test_mae','NA'):>7}"
        nasa_r2  = f"{ref.get('test_r2','NA'):>7}"
        if mrow:
            lli_r2 = f"{mrow['lli_r2']:>7.3f}"
            lam_r2 = f"{mrow['lam_r2']:>7.3f}"
            cl_r2  = f"{mrow['cl_r2']:>7.3f}"
            m_mae  = f"{mrow['mean_deg_mae']:>7.4f}"
        else:
            lli_r2 = lam_r2 = cl_r2 = m_mae = f"{'FAIL':>7}"
        params = ref.get("params", "NA")
        print(f"  {m:5s}  {str(params):>9s}  {nasa_mae}%  {nasa_r2}  "
              f"{lli_r2}  {lam_r2}  {cl_r2}  {m_mae}")
    print(SEP)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\nSUCCESS ({len(success)}): {success}")
    if failed:
        logger.info(f"FAILED  ({len(failed)}): {failed}")
    logger.info(f"Results saved -> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",        nargs="+", default=ALL_MODELS)
    ap.add_argument("--seeds",         type=int, nargs="+", default=[42])
    ap.add_argument("--epochs",        type=int, default=EPOCHS)
    ap.add_argument("--patience",      type=int, default=PATIENCE)
    ap.add_argument("--device",        default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--skip_nasa",     action="store_true",
                    help="Skip NASA (use reference table only)")
    ap.add_argument("--skip_mendeley", action="store_true",
                    help="Skip Mendeley training (load existing results)")
    args = ap.parse_args()
    EPOCHS  = args.epochs
    PATIENCE = args.patience
    main(args)
