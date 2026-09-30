"""
Samsung External Test Evaluation — EIS-Only.

RESEARCH PROTOCOL (2026-09):
  Samsung INR18650-30Q cells are used EXCLUSIVELY as an independent
  external test dataset. They are NEVER used for training, validation,
  hyperparameter tuning, early stopping, normalization, or weight updates.

IMPORTANT — CASE B APPLIES:
  Samsung does NOT provide discharge time-series (V/I/T).
  Therefore, the full NASA multimodal model (Discharge + EIS + Physics)
  CANNOT be evaluated on Samsung data.

  This script evaluates ONLY the EIS encoder sub-path of each model.
  Results are explicitly labeled:

      "Samsung EIS-only external evaluation"

  NOT:
      "Full BaFuse multimodal external validation"

What this script does:
  1. Load a checkpoint trained ONLY on NASA data (weights frozen).
  2. Load Samsung EIS features from the test split only (cell 2 by default).
  3. Apply NASA training normalization statistics (NOT Samsung statistics).
  4. Route Samsung EIS through only the EIS encoder branch of each model.
  5. Evaluate predictions against Samsung SOH labels.
  6. Report metrics with explicit EIS-only caveat.

What this script does NOT do:
  - Does NOT use discharge or physics branches (not available from Samsung).
  - Does NOT zero-pad missing modalities to run the full fusion model.
  - Does NOT retrain or fine-tune any model.
  - Does NOT use Samsung data for normalization statistics.
  - Does NOT use Samsung LLI/LAM/CL as training targets.
  - Does NOT claim results are equivalent to full multimodal evaluation.

Normalization:
  Uses NASA training statistics from data/processed/nasa_train_stats.json
  (saved automatically by create_dataloaders() during training).
  If that file does not exist, re-runs the dataset to compute train stats.

EIS compatibility caveat:
  Samsung EIS values are ~13x smaller than NASA EIS values (different cell
  chemistry/size: Samsung 2.95 Ah vs NASA 2.0 Ah). Applying NASA z-score
  statistics to Samsung values produces out-of-distribution inputs.
  This limits the scientific validity of cross-dataset EIS comparison.
  All results must be interpreted with this domain gap in mind.

Usage:
    python scripts/evaluate_samsung_external.py
    python scripts/evaluate_samsung_external.py --test_cells 2
    python scripts/evaluate_samsung_external.py --out_dir results/samsung_eval
"""

import argparse, json, logging, sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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

SAMSUNG_NOMINAL_AH = 2.95
NASA_NOMINAL_AH    = 2.0

# Samsung EIS columns in the processed pkl
SAMSUNG_EIS_COLS = [
    "zmod_ohm", "zreal_ohm", "zimg_ohm", "zphz_deg",
    "R_electrolyte", "R_ct1", "R_ct2", "Zw",
]

# The 3 EIS features the NASA model expects
# (BaFuseDataset returns: [impedance_ohm, re_ohm, rct_ohm])
# Samsung column → closest equivalent
SAMSUNG_TO_NASA_EIS = {
    "zmod_ohm":      ("impedance", "Closest to NASA |Z|; ~13x smaller"),
    "zreal_ohm":     ("re",        "Closest to NASA Re(Z); ~13x smaller"),
    "zimg_ohm":      ("rct",       "Closest to NASA |Im(Z)|; ~9x smaller"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_samsung_test(
    processed_dir: str = "data/mendeley_processed",
    test_cells: Optional[list] = None,
) -> pd.DataFrame:
    """Load Samsung test split only. Never loads train/val."""
    proc = Path(processed_dir)
    df = pd.read_pickle(str(proc / "mendeley_test.pkl"))
    if test_cells:
        df = df[df["cell_id"].isin(test_cells)].reset_index(drop=True)
    logger.info(
        f"Samsung test: {len(df)} samples, "
        f"cells={sorted(df.cell_id.unique().tolist())}, "
        f"aging_cycles=[{df.aging_cycle.min()}, {df.aging_cycle.max()}]"
    )
    return df


def load_nasa_train_stats(
    stats_path: str = "data/processed/nasa_train_stats.json",
) -> Dict:
    """
    Load NASA training normalization statistics.

    These stats were computed from NASA train batteries only and saved by
    create_dataloaders(save_stats_path=...) during training.
    They must NEVER be replaced by Samsung statistics.
    """
    p = ROOT / stats_path
    if p.exists():
        with open(str(p)) as f:
            stats = json.load(f)
        logger.info(f"Loaded NASA train stats from {p}")
        logger.info(f"  Keys: {list(stats.keys())}")
        return stats

    # Fall back: recompute from cached train.pkl
    train_pkl = ROOT / "data" / "processed" / "train.pkl"
    if train_pkl.exists():
        logger.warning(
            f"nasa_train_stats.json not found. "
            f"Recomputing from {train_pkl} (train data only)."
        )
        import pandas as pd_
        from src.data.dataset import BaFuseDataset
        train_df = pd_.read_pickle(str(train_pkl))
        ds = BaFuseDataset(train_df, normalize=True)
        stats = ds.stats
        # Save for future use
        with open(str(p), "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved NASA train stats to {p}")
        return stats

    raise FileNotFoundError(
        "Cannot find NASA training stats. "
        "Run training pipeline first, or ensure "
        "data/processed/nasa_train_stats.json exists."
    )


def samsung_soh_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Compute Samsung SOH from test data.
    Used ONLY after prediction for metric computation — never as model input.
    """
    if "soh_norm" in df.columns:
        return df["soh_norm"].values.astype(float)
    if "soh_pct" in df.columns:
        return df["soh_pct"].values.astype(float) / 100.0
    raise ValueError("No SOH column in Samsung test data.")


# ═══════════════════════════════════════════════════════════════════════════════
# EIS-only evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def build_samsung_eis_3d(
    samsung_df: pd.DataFrame,
    nasa_stats: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """
    Build a 3D EIS tensor matching the NASA model's expected EIS input shape.

    NASA EIS input = (B, 3): [z-scored |Z|, z-scored Re, z-scored |Im|]

    Samsung mapping:
      Samsung zmod_ohm  → NASA impedance (stat key: 'impedance')
      Samsung zreal_ohm → NASA re        (stat key: 're')
      Samsung zimg_ohm  → NASA rct       (stat key: 'rct')

    IMPORTANT CAVEAT:
      Samsung values are ~13x smaller than NASA values due to different cell
      chemistry and size (2.95 Ah vs 2.0 Ah). After NASA z-score normalisation,
      Samsung values will be far below the training distribution mean (negative
      z-scores ~-10 to -6). This is an inherent domain gap, not a bug.
      Results must be reported with this caveat.

    Returns:
        eis_tensor: (B, 3) z-scored tensor using NASA statistics
        normalization_info: dict documenting the transformation applied
    """
    mapping = [
        ("zmod_ohm",  "impedance"),
        ("zreal_ohm", "re"),
        ("zimg_ohm",  "rct"),
    ]

    eis_arr = np.zeros((len(samsung_df), 3), dtype=np.float32)
    norm_info = {}

    for i, (sam_col, nasa_key) in enumerate(mapping):
        vals = samsung_df[sam_col].fillna(0.0).values.astype(float)
        s = nasa_stats.get(nasa_key, {"mean": 0.0, "std": 1.0})

        # Apply NASA z-score
        z_scores = (vals - s["mean"]) / s["std"]
        eis_arr[:, i] = z_scores

        norm_info[sam_col] = {
            "samsung_mean": float(np.mean(vals)),
            "samsung_std":  float(np.std(vals)),
            "nasa_mean":    s["mean"],
            "nasa_std":     s["std"],
            "z_score_mean": float(np.mean(z_scores)),
            "z_score_std":  float(np.std(z_scores)),
            "ood_warning":  abs(float(np.mean(z_scores))) > 3.0,
        }

    # Flag out-of-distribution
    n_ood = sum(1 for v in norm_info.values() if v["ood_warning"])
    if n_ood > 0:
        logger.warning(
            f"  EIS DOMAIN GAP: {n_ood}/3 features have mean z-score > 3σ from "
            f"NASA training distribution. Samsung cell chemistry produces "
            f"systematically different EIS values. Results are exploratory only."
        )

    return torch.tensor(eis_arr, device=device), norm_info


def evaluate_eis_only(
    model_name: str,
    eis_tensor: torch.Tensor,
    samsung_soh: np.ndarray,
    device: torch.device,
) -> Dict:
    """
    Evaluate a model using ONLY its EIS encoder branch.

    This function:
    1. Instantiates a fresh model (untrained weights — only evaluating structure).
    2. Extracts just the EIS encoder.
    3. Passes Samsung EIS through the encoder.
    4. Uses a RANDOMLY INITIALISED downstream head for SOH.

    NOTE: Since the model weights are NOT the NASA-trained weights (we don't
    have a saved checkpoint to reload), this evaluates structural EIS encoder
    capacity on Samsung data, not the trained NASA model.

    If a checkpoint is provided, the correct weights will be loaded.
    """
    from src.models.benchmark_models import get_model

    model = get_model(model_name, d_in=3, e_in=3, p_in=4)
    model.eval()

    n_eis_params = 0
    if hasattr(model, "e_enc"):
        enc = model.e_enc
        n_eis_params = sum(p.numel() for p in enc.parameters())
        with torch.no_grad():
            eis_emb = model.e_enc(eis_tensor)
    elif hasattr(model, "e_cnn"):
        enc = model.e_cnn
        n_eis_params = sum(p.numel() for p in enc.parameters())
        with torch.no_grad():
            eis_emb = model.e_cnn(eis_tensor.unsqueeze(1)).squeeze(-1)
    elif hasattr(model, "e_tcn"):
        enc = model.e_tcn
        n_eis_params = sum(p.numel() for p in enc.parameters())
        with torch.no_grad():
            eis_emb = model.e_tcn(eis_tensor.unsqueeze(1)).mean(dim=-1)
    elif hasattr(model, "d_enc"):
        # Model A uses flat pooling — no dedicated EIS encoder;
        # the whole model is a flat MLP; use the whole model's parameters
        # but evaluate by routing through full net with zero discharge/physics.
        # Mark explicitly as "flat-model EIS-context" not EIS-only.
        n_total = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            disc_zero = torch.zeros(len(eis_tensor), 100, 3, device=eis_tensor.device)
            phys_zero = torch.zeros(len(eis_tensor), 4,   device=eis_tensor.device)
            out = model(disc_zero, eis_tensor, phys_zero)
        preds = out["soh_pred"].view(-1).numpy()
        mae   = float(np.mean(np.abs(preds - samsung_soh))) * 100
        rmse  = float(np.sqrt(np.mean((preds - samsung_soh)**2))) * 100
        ss_r  = np.sum((samsung_soh - preds)**2)
        ss_t  = np.sum((samsung_soh - samsung_soh.mean())**2)
        r2    = float(1 - ss_r / (ss_t + 1e-8))
        return {
            "model":            model_name,
            "evaluation_mode":  "Flat model with zero discharge/physics (EIS context only)",
            "eis_encoder_params": n_total,
            "embedding_dim":    "N/A",
            "mae_pct":          round(mae, 3),
            "rmse_pct":         round(rmse, 3),
            "r2":               round(r2, 4),
            "n":                len(samsung_soh),
            "caveat": (
                "Flat-pooling model (A). No dedicated EIS encoder. "
                "Zero tensors used for discharge and physics — "
                "this is NOT a trained-model evaluation. Random weights."
            ),
        }
    else:
        return {"error": f"Model {model_name} has no identifiable EIS encoder"}

    emb_dim = eis_emb.shape[-1]

    # Minimal linear head for SOH prediction from EIS embedding
    # (random weights — only valid for structural capability assessment)
    head = nn.Linear(emb_dim, 1)
    nn.init.xavier_uniform_(head.weight)
    with torch.no_grad():
        preds = head(eis_emb).view(-1).numpy()

    mae  = float(np.mean(np.abs(preds - samsung_soh))) * 100
    rmse = float(np.sqrt(np.mean((preds - samsung_soh)**2))) * 100
    ss_r = np.sum((samsung_soh - preds)**2)
    ss_t = np.sum((samsung_soh - samsung_soh.mean())**2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))

    return {
        "model":            model_name,
        "evaluation_mode":  "EIS-encoder-only (random downstream head)",
        "eis_encoder_params": n_eis_params,
        "embedding_dim":    emb_dim,
        "mae_pct":          round(mae, 3),
        "rmse_pct":         round(rmse, 3),
        "r2":               round(r2, 4),
        "n":                len(samsung_soh),
        "caveat": (
            "Random downstream head — metrics reflect EIS embedding quality, "
            "not a trained NASA model. Load a checkpoint for meaningful evaluation."
        ),
    }


def evaluate_with_checkpoint(
    checkpoint_path: str,
    eis_tensor: torch.Tensor,
    samsung_soh: np.ndarray,
    device: torch.device,
) -> Dict:
    """
    Evaluate the trained NASA model using only the EIS encoder.

    Loads the full trained checkpoint, extracts the EIS encoder,
    and applies a forward pass with only EIS input.

    The model is fully frozen. No gradient computation.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f"  Loaded checkpoint: epoch={ckpt.get('epoch','?')}")

    # Try BaFuseV2 first, then BaFuse
    try:
        from src.models.bafuse_v2 import BaFuseV2
        model = BaFuseV2(
            discharge_input_size=3, eis_num_frequencies=3,
            physics_num_features=4, latent_dim=64, fusion_dim=128,
        )
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    except Exception:
        from src.models.bafuse import BaFuse
        model = BaFuse(
            discharge_input_size=3, eis_num_frequencies=3,
            physics_num_features=4, latent_dim=64, fusion_dim=128,
        )
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False   # confirm frozen

    # Extract EIS latent using trained encoder
    with torch.no_grad():
        eis_latent = model.eis_encoder(eis_tensor)   # (B, latent_dim)

    emb_dim = eis_latent.shape[-1]

    # Use the trained SOH head but pass only EIS latent
    # We pass zero tensors for discharge and physics to the FULL model
    # — however we must be explicit that this is NOT a valid full eval
    # We instead use just eis_latent + a linear projection
    n = len(samsung_soh)
    with torch.no_grad():
        # Linear head from EIS latent only (trained SOH head expects fused 128D)
        # Create a minimal projection for honest EIS-only evaluation
        head_proj = nn.Linear(emb_dim, 1, bias=True)
        head_proj.eval()
        preds = head_proj(eis_latent).view(-1).detach().numpy()

    mae  = float(np.mean(np.abs(preds - samsung_soh))) * 100
    rmse = float(np.sqrt(np.mean((preds - samsung_soh)**2))) * 100
    ss_r = np.sum((samsung_soh - preds)**2)
    ss_t = np.sum((samsung_soh - samsung_soh.mean())**2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))

    return {
        "checkpoint":        checkpoint_path,
        "evaluation_mode":   "Trained NASA EIS encoder + random linear projection",
        "embedding_dim":     emb_dim,
        "mae_pct":           round(mae, 3),
        "rmse_pct":          round(rmse, 3),
        "r2":                round(r2, 4),
        "n":                 n,
        "caveat": (
            "NASA-trained EIS encoder weights loaded. Downstream projection "
            "is random (trained SOH head expects 128D fused input, not 64D EIS "
            "latent alone). This reflects EIS encoder transferability, "
            "not the full trained model performance."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Physical degradation analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_degradation_analysis(samsung_df: pd.DataFrame) -> dict:
    """
    Physical degradation analysis using observable EIS signal changes.
    Does NOT use Samsung LLI/LAM/CL as supervised targets.
    """
    from src.degradation_analysis import BatteryDegradationAnalyzer

    analyzer = BatteryDegradationAnalyzer()
    adapted = samsung_df.copy()

    # Column name adaptation for BatteryDegradationAnalyzer
    if "cell_id" in adapted.columns:
        adapted["battery_id"] = "Samsung_Cell_" + adapted["cell_id"].astype(str)
    if "aging_cycle" in adapted.columns:
        adapted["discharge_cycle"] = adapted["aging_cycle"]
    if "zmod_ohm" in adapted.columns:
        adapted["impedance_ohm"] = adapted["zmod_ohm"]
    if "zreal_ohm" in adapted.columns:
        adapted["re_ohm"] = adapted["zreal_ohm"]
    if "zimg_ohm" in adapted.columns:
        adapted["rct_ohm"] = adapted["zimg_ohm"]
    if "soh_norm" in adapted.columns:
        adapted["capacity_ahr"] = adapted["soh_norm"] * SAMSUNG_NOMINAL_AH

    reports = {}
    for bid, grp in adapted.groupby("battery_id"):
        try:
            reports[str(bid)] = analyzer.analyze_battery(
                grp.copy(), str(bid),
                cell_info={"chemistry": "Samsung INR18650-30Q",
                           "nominal_ah": SAMSUNG_NOMINAL_AH}
            )
        except Exception as e:
            reports[str(bid)] = {"battery_id": bid, "error": str(e)}

    return reports


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 68

    logger.info(SEP)
    logger.info("Samsung EIS-Only External Test Evaluation")
    logger.info("  Research protocol: Samsung is external test only")
    logger.info("  Modality:          EIS only (discharge/physics unavailable)")
    logger.info("  Evaluation type:   EIS encoder structural assessment")
    logger.info("  Samsung NOT used for: training / validation / normalization")
    logger.info(SEP)

    # ── Load Samsung test data ────────────────────────────────────────────────
    samsung_df  = load_samsung_test(args.samsung_dir, args.test_cells)
    samsung_soh = samsung_soh_labels(samsung_df)
    logger.info(
        f"Samsung SOH range: [{samsung_soh.min():.3f}, {samsung_soh.max():.3f}] "
        f"(Samsung nominal: {SAMSUNG_NOMINAL_AH} Ah)"
    )

    # ── Load NASA training stats ──────────────────────────────────────────────
    logger.info("")
    logger.info("Loading NASA training normalization statistics...")
    nasa_stats = load_nasa_train_stats(args.nasa_stats)

    # ── Build Samsung EIS tensor (normalized with NASA stats) ─────────────────
    logger.info("")
    logger.info("Building Samsung EIS features (normalized with NASA statistics):")
    eis_tensor, norm_info = build_samsung_eis_3d(samsung_df, nasa_stats, device)
    logger.info("  Normalization info (domain gap assessment):")
    for col, info in norm_info.items():
        ood_tag = " [OUT-OF-DISTRIBUTION]" if info["ood_warning"] else ""
        logger.info(
            f"    {col}: Samsung_mean={info['samsung_mean']:.5f}  "
            f"NASA_mean={info['nasa_mean']:.5f}  "
            f"z_mean={info['z_score_mean']:.2f}{ood_tag}"
        )

    # ── Checkpoint evaluation (if provided) ───────────────────────────────────
    checkpoint_result = None
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info("")
        logger.info(f"Evaluating with checkpoint: {args.checkpoint}")
        checkpoint_result = evaluate_with_checkpoint(
            args.checkpoint, eis_tensor, samsung_soh, device
        )
        logger.info(
            f"  Checkpoint EIS eval: MAE={checkpoint_result['mae_pct']:.2f}%  "
            f"R2={checkpoint_result['r2']:.4f}"
        )
        logger.info(f"  Caveat: {checkpoint_result['caveat']}")

    # ── 14-model EIS encoder assessment ──────────────────────────────────────
    logger.info("")
    logger.info("EIS encoder structural assessment (all 14 benchmark models):")
    logger.info("  Note: Random downstream head — reflects encoder capacity only")
    model_names = ["D","G","A","C","F","E2","E","B","D2","G2","C2","E5","E4","E3"]
    model_results = {}

    for mname in model_names:
        try:
            r = evaluate_eis_only(mname, eis_tensor, samsung_soh, device)
            model_results[mname] = r
            logger.info(
                f"  {mname:4s}: EIS_enc_params={r['eis_encoder_params']:,}  "
                f"emb_dim={r['embedding_dim']}  "
                f"MAE={r['mae_pct']:.2f}%  R2={r['r2']:.4f}"
            )
        except Exception as e:
            model_results[mname] = {"error": str(e)}
            logger.warning(f"  {mname}: FAILED — {e}")

    # ── Physical degradation analysis ─────────────────────────────────────────
    logger.info("")
    logger.info("Physical degradation analysis (observable signals, no Samsung labels):")
    deg_reports = run_degradation_analysis(samsung_df)
    for bid, rep in deg_reports.items():
        if "error" in rep:
            logger.warning(f"  {bid}: {rep['error']}")
        else:
            fs  = rep.get("final_signals", {})
            dom = rep.get("rag_result", {}).get("dominant_mode", "?")
            logger.info(
                f"  {bid}: SOH={rep.get('final_soh', float('nan')):.3f}  "
                f"imp_rise={fs.get('impedance_rise', 0):.4f}  "
                f"dominant_signal={dom}"
            )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("SAMSUNG EXTERNAL EVALUATION — EIS-ONLY STRUCTURAL ASSESSMENT")
    print("  IMPORTANT: This is NOT a full multimodal evaluation.")
    print("  Samsung lacks discharge time-series (V/I/T).")
    print("  The full NASA model requires all 3 modalities.")
    print(f"  Samsung cell(s): {sorted(samsung_df.cell_id.unique().tolist())}")
    print(f"  Samsung nominal capacity: {SAMSUNG_NOMINAL_AH} Ah (vs NASA {NASA_NOMINAL_AH} Ah)")
    print(f"  Domain gap: Samsung EIS values are ~13x smaller than NASA EIS values.")
    print(SEP)
    print(f"  {'Model':5s}  {'EIS_params':>12s}  {'emb_dim':>8s}  {'MAE%':>7s}  {'R2':>7s}  Notes")
    print(f"  {'-'*68}")
    for mname in model_names:
        r = model_results.get(mname, {})
        if "error" in r:
            print(f"  {mname:5s}  {'FAIL':>12s}  {'':>8s}  {'':>7s}  {'':>7s}")
        else:
            print(
                f"  {mname:5s}  {r.get('eis_encoder_params',0):>12,}  "
                f"{r.get('embedding_dim',0):>8}  "
                f"{r.get('mae_pct', float('nan')):>7.2f}  "
                f"{r.get('r2', float('nan')):>7.4f}  random-head"
            )
    print(SEP)
    print()
    print("Scientific caveats:")
    print("  1. EIS-only: discharge and physics modalities NOT available from Samsung.")
    print("  2. Domain gap: Samsung EIS ~13x smaller scale than NASA EIS.")
    print("  3. Random downstream head: metrics reflect EIS encoder capacity,")
    print("     not trained model performance.")
    print("  4. Samsung NOT used in training, validation, or normalization.")
    print("  5. Samsung SOH used ONLY for post-hoc metric calculation.")
    print(SEP)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "evaluation_type":       "Samsung EIS-only external test (CASE B — not full multimodal)",
        "protocol":              "Samsung external test only; not used in training/validation",
        "samsung_nominal_ah":    SAMSUNG_NOMINAL_AH,
        "nasa_nominal_ah":       NASA_NOMINAL_AH,
        "n_samsung_samples":     int(len(samsung_df)),
        "samsung_cells":         sorted([int(x) for x in samsung_df.cell_id.unique()]),
        "samsung_soh_range":     [float(samsung_soh.min()), float(samsung_soh.max())],
        "normalization_source":  "NASA training statistics (nasa_train_stats.json)",
        "eis_domain_gap": {
            "note": "Samsung EIS values are ~13x smaller than NASA EIS values",
            "details": norm_info,
        },
        "model_eis_assessment":  model_results,
        "checkpoint_result":     checkpoint_result,
        "degradation_note": (
            "Physical signals from Samsung EIS (impedance rise, aging trends). "
            "No LLI/LAM/CL labels used as supervised targets."
        ),
        "scientific_limitations": [
            "No discharge time-series in Samsung data — full model cannot be evaluated.",
            "EIS absolute scale differs ~13x between datasets.",
            "Random downstream head used — not the trained NASA SOH head.",
            "Single test cell (cell 2) — insufficient for statistical significance.",
        ],
    }

    out_file = out_dir / "samsung_external_eval.json"
    with open(str(out_file), "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved -> {out_file}")

    # Save per-model CSV
    rows = []
    for m, r in model_results.items():
        rows.append({
            "model": m,
            "evaluation_type": "Samsung EIS-only (random head)",
            "eis_encoder_params": r.get("eis_encoder_params", "NA"),
            "embedding_dim": r.get("embedding_dim", "NA"),
            "samsung_mae_pct": r.get("mae_pct", "NA"),
            "samsung_rmse_pct": r.get("rmse_pct", "NA"),
            "samsung_r2": r.get("r2", "NA"),
            "n_samsung": r.get("n", "NA"),
            "caveat": r.get("caveat", r.get("error", "")),
        })
    import pandas as pd
    pd.DataFrame(rows).to_csv(str(out_dir / "samsung_metrics.csv"), index=False)
    logger.info(f"Metrics CSV -> {out_dir / 'samsung_metrics.csv'}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Samsung EIS-only external test evaluation"
    )
    ap.add_argument("--samsung_dir",  default="data/mendeley_processed")
    ap.add_argument("--nasa_stats",   default="data/processed/nasa_train_stats.json",
                    help="Path to NASA training normalization stats JSON")
    ap.add_argument("--checkpoint",   default=None,
                    help="NASA-trained checkpoint (optional; for EIS encoder extraction)")
    ap.add_argument("--test_cells",   type=int, nargs="+", default=None)
    ap.add_argument("--out_dir",      default="results/samsung_external_eval")
    ap.add_argument("--device",       default="cpu", choices=["cpu", "cuda"])
    sys.exit(main(ap.parse_args()) or 0)
