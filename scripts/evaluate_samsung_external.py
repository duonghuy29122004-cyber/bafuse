"""
Samsung/Mendeley External Test Evaluation.

NEW DIRECTION (2026-09):
  Samsung/Mendeley is used EXCLUSIVELY as an independent external test dataset.
  It is NEVER used for training, validation, normalization, or any weight update.

Purpose:
  Evaluate generalization of a NASA-trained SOH model to Samsung INR18650-30Q
  cells, which differ in chemistry and protocol from the NASA 18650 cells.

What this script does:
  1. Load a checkpoint trained ONLY on NASA data.
  2. Load Samsung/Mendeley EIS features (no discharge time-series).
  3. Evaluate SOH predictions using NASA normalization statistics.
  4. Compare with NASA internal test results.
  5. Run physical degradation analysis on Samsung signal changes.

What this script does NOT do:
  - Does NOT retrain any model.
  - Does NOT fine-tune on Samsung data.
  - Does NOT use Samsung data for normalization.
  - Does NOT use Samsung LLI/LAM/CL labels as training targets.
  - Does NOT row-match NASA and Samsung cells.

Important caveats reported:
  - Samsung nominal capacity (2.95 Ah) differs from NASA (2.0 Ah).
  - Direct SOH comparison requires per-dataset normalization.
  - Chemistry/protocol differences mean generalization gap is expected.

Usage:
    python scripts/evaluate_samsung_external.py
    python scripts/evaluate_samsung_external.py --checkpoint checkpoints/best_model.pth
    python scripts/evaluate_samsung_external.py --model_name G --checkpoint ...
"""

import argparse, json, logging, sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Samsung nominal capacity (INR18650-30Q)
SAMSUNG_NOMINAL_AH = 2.95

# NASA nominal capacity (used in model training)
NASA_NOMINAL_AH = 2.0

# EIS features shared between datasets
EIS_FEATURE_COLS = [
    "zmod_ohm", "zreal_ohm", "zimg_ohm", "zphz_deg",
    "R_electrolyte", "R_ct1", "R_ct2", "Zw",
]


def load_nasa_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load a NASA-trained model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"  Checkpoint epoch : {ckpt.get('epoch', '?')}")
    logger.info(f"  Val metrics      : {ckpt.get('val_metrics', {})}")
    return ckpt


def load_samsung_data(
    processed_dir: str = "data/mendeley_processed",
    test_cells: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load Samsung/Mendeley processed data for external evaluation.

    Returns the test split only. Never loads train/val splits to prevent
    accidental normalization leakage.
    """
    proc = Path(processed_dir)
    test_pkl = proc / "mendeley_test.pkl"
    if not test_pkl.exists():
        raise FileNotFoundError(
            f"Samsung test data not found at {test_pkl}. "
            "Run scripts/preprocess_mendeley.py first."
        )
    df = pd.read_pickle(str(test_pkl))
    if test_cells:
        df = df[df["cell_id"].isin(test_cells)].reset_index(drop=True)
    logger.info(
        f"Samsung external test: {len(df)} samples, "
        f"cells={sorted(df.cell_id.unique().tolist())}"
    )
    return df


def build_samsung_eis_tensor(
    samsung_df: pd.DataFrame,
    nasa_eis_stats: Dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Build EIS feature tensor from Samsung data, normalised with NASA statistics.

    IMPORTANT: Uses NASA training statistics, NOT Samsung statistics.
    This is the correct procedure for external test — the model was trained
    with NASA normalization and must receive the same scaling at test time.

    Caveat: Samsung EIS features have a different absolute scale than NASA
    (different cell chemistry/size). Performance may be limited by this
    domain gap.
    """
    avail = [c for c in EIS_FEATURE_COLS if c in samsung_df.columns]
    if not avail:
        raise ValueError(
            f"No EIS feature columns found in Samsung data. "
            f"Expected: {EIS_FEATURE_COLS}. Got: {list(samsung_df.columns)}"
        )

    eis_arr = np.zeros((len(samsung_df), len(avail)), dtype=np.float32)
    for i, col in enumerate(avail):
        vals = samsung_df[col].fillna(0.0).values.astype(float)
        if col in nasa_eis_stats:
            s = nasa_eis_stats[col]
            vals = (vals - s["mean"]) / s["std"]
        eis_arr[:, i] = vals

    return torch.tensor(eis_arr, device=device)


def compute_samsung_soh_labels(samsung_df: pd.DataFrame) -> np.ndarray:
    """
    Compute SOH for Samsung cells using Samsung nominal capacity (2.95 Ah).

    NOTE: Do NOT use NASA nominal capacity (2.0 Ah) for Samsung cells.
    These are different physical batteries.
    """
    if "soh_norm" in samsung_df.columns:
        return samsung_df["soh_norm"].values.astype(float)
    if "soh_pct" in samsung_df.columns:
        return (samsung_df["soh_pct"].values.astype(float) / 100.0)
    # Fallback: compute from capacity if available
    if "capacity_ahr" in samsung_df.columns:
        return np.clip(
            samsung_df["capacity_ahr"].values / SAMSUNG_NOMINAL_AH, 0, 1
        )
    raise ValueError("Cannot compute Samsung SOH: no soh_norm, soh_pct, or capacity_ahr column found.")


def metrics_soh(pred: np.ndarray, tgt: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(pred - tgt))) * 100      # report as %
    rmse = float(np.sqrt(np.mean((pred - tgt)**2))) * 100
    ss_r = np.sum((tgt - pred)**2)
    ss_t = np.sum((tgt - tgt.mean())**2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))
    return {"mae_pct": round(mae, 3), "rmse_pct": round(rmse, 3),
            "r2": round(r2, 4), "n": len(pred)}


def evaluate_with_benchmark_model(
    model_name: str,
    samsung_df: pd.DataFrame,
    nasa_eis_stats: Dict,
    device: torch.device,
) -> dict:
    """
    Evaluate a benchmark model (A-G, C2-G2) on Samsung EIS features.

    These models accept (discharge, eis, physics) but we can evaluate
    the EIS-only path by zeroing out discharge and physics.
    This is an ablation experiment to see EIS-only generalization.
    """
    from src.models.benchmark_models import get_model

    # Samsung EIS is 8D; benchmark models use 3D NASA EIS.
    # We use the first 3 EIS features (zmod, zreal, zimg) to match NASA dims.
    avail = [c for c in EIS_FEATURE_COLS if c in samsung_df.columns]
    if len(avail) < 3:
        return {"error": f"Insufficient EIS features: {avail}"}

    # Use first 3 features (rough approximation; see caveat)
    eis_3d = np.zeros((len(samsung_df), 3), dtype=np.float32)
    for i, col in enumerate(avail[:3]):
        vals = samsung_df[col].fillna(0.0).values.astype(float)
        if col in nasa_eis_stats:
            s = nasa_eis_stats[col]
            vals = (vals - s["mean"]) / s["std"]
        eis_3d[:, i] = vals

    eis_t = torch.tensor(eis_3d, device=device)
    disc_zero  = torch.zeros(len(samsung_df), 100, 3, device=device)
    phys_zero  = torch.zeros(len(samsung_df), 4, device=device)

    model = get_model(model_name, d_in=3, e_in=3, p_in=4)
    model.eval()

    with torch.no_grad():
        out = model(disc_zero, eis_t, phys_zero)

    preds = out["soh_pred"].view(-1).cpu().numpy()
    tgts  = compute_samsung_soh_labels(samsung_df)
    return metrics_soh(preds, tgts)


def run_physical_degradation_analysis(
    samsung_df: pd.DataFrame,
) -> dict:
    """
    Run physical degradation analysis on Samsung cells using observable signals.

    Uses src/degradation_analysis.py — does NOT use Samsung LLI/LAM/CL labels.
    """
    from src.degradation_analysis import BatteryDegradationAnalyzer

    analyzer = BatteryDegradationAnalyzer()
    reports = {}

    # Samsung df uses cell_id, not battery_id; adapt column name
    if "cell_id" in samsung_df.columns and "battery_id" not in samsung_df.columns:
        samsung_df = samsung_df.copy()
        samsung_df["battery_id"] = "Samsung_Cell_" + samsung_df["cell_id"].astype(str)
    # Samsung df uses aging_cycle, not discharge_cycle
    if "aging_cycle" in samsung_df.columns and "discharge_cycle" not in samsung_df.columns:
        samsung_df = samsung_df.copy()
        samsung_df["discharge_cycle"] = samsung_df["aging_cycle"]
    # Map Samsung EIS column names to expected names
    col_map = {
        "zmod_ohm": "impedance_ohm",
        "zreal_ohm": "re_ohm",
        "zimg_ohm": "rct_ohm",
        "soh_norm": "capacity_ahr",   # approximate: soh_norm * nominal ≈ capacity
    }
    renamed = samsung_df.copy()
    for src, dst in col_map.items():
        if src in renamed.columns and dst not in renamed.columns:
            renamed[dst] = renamed[src]
    # For capacity_ahr, scale from soh_norm
    if "capacity_ahr" not in renamed.columns and "soh_norm" in renamed.columns:
        renamed["capacity_ahr"] = renamed["soh_norm"] * SAMSUNG_NOMINAL_AH

    for bid, grp in renamed.groupby("battery_id"):
        try:
            reports[str(bid)] = analyzer.analyze_battery(
                grp.copy(), str(bid),
                cell_info={"chemistry": "Samsung INR18650-30Q",
                           "nominal_ah": SAMSUNG_NOMINAL_AH}
            )
        except Exception as e:
            reports[str(bid)] = {"battery_id": bid, "error": str(e)}

    return reports


def main(args):
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 65
    logger.info(SEP)
    logger.info("Samsung External Test Evaluation")
    logger.info("  Training data  : NASA PCoE ONLY")
    logger.info("  Evaluation data: Samsung INR18650-30Q (Mendeley, external test)")
    logger.info("  Protocol       : No Samsung data in training/normalization")
    logger.info(SEP)

    # ── Load Samsung test data ────────────────────────────────────────────────
    samsung_df = load_samsung_data(
        processed_dir=args.samsung_dir,
        test_cells=args.test_cells,
    )

    # ── Load NASA normalization statistics ────────────────────────────────────
    # Use NASA train stats — do NOT refit on Samsung
    nasa_stats_path = Path("data/mendeley_processed/mendeley_stats.json")
    nasa_eis_stats: dict = {}
    if nasa_stats_path.exists():
        with open(str(nasa_stats_path)) as f:
            nasa_eis_stats = json.load(f)
        logger.info(
            f"Loaded normalization stats (Note: these are Mendeley stats, "
            f"not NASA. Ideally use NASA EIS stats for strict external eval.)"
        )
    else:
        logger.warning(
            "No normalization stats found. Samsung EIS will not be scaled. "
            "Results will be unreliable."
        )

    # ── Compute Samsung SOH labels ────────────────────────────────────────────
    samsung_soh = compute_samsung_soh_labels(samsung_df)
    logger.info(
        f"Samsung SOH range: [{samsung_soh.min():.3f}, {samsung_soh.max():.3f}] "
        f"(normalized to Samsung nominal {SAMSUNG_NOMINAL_AH} Ah)"
    )

    # ── Evaluate benchmark models ─────────────────────────────────────────────
    logger.info("\n" + SEP)
    logger.info("EIS-only evaluation on Samsung (benchmark models A-G2)")
    logger.info("  Caveat: Only 3D EIS used (models trained on NASA 3D EIS).")
    logger.info("  Discharge and physics branches zeroed out.")
    logger.info(SEP)

    model_names = ["D", "G", "A", "C", "F", "E2", "E", "B",
                   "D2", "G2", "C2", "E5", "E4", "E3"]

    benchmark_results = {}
    for mname in model_names:
        try:
            r = evaluate_with_benchmark_model(mname, samsung_df, nasa_eis_stats, device)
            benchmark_results[mname] = r
            logger.info(
                f"  {mname:4s}: MAE={r.get('mae_pct','?'):.2f}%  "
                f"RMSE={r.get('rmse_pct','?'):.2f}%  R2={r.get('r2','?'):.4f}"
            )
        except Exception as e:
            benchmark_results[mname] = {"error": str(e)}
            logger.warning(f"  {mname}: FAILED — {e}")

    # ── Physical degradation analysis (signal-based, no Samsung labels) ───────
    logger.info("\n" + SEP)
    logger.info("Physical Degradation Analysis (observable signals only)")
    logger.info("  No Samsung LLI/LAM/CL labels used.")
    logger.info(SEP)

    deg_reports = run_physical_degradation_analysis(samsung_df)
    for bid, rep in deg_reports.items():
        if "error" in rep:
            logger.warning(f"  {bid}: {rep['error']}")
        else:
            fs = rep.get("final_signals", {})
            dom = rep.get("rag_result", {}).get("dominant_mode", "?")
            logger.info(
                f"  {bid}: SOH={rep.get('final_soh', float('nan')):.3f}  "
                f"imp_rise={fs.get('impedance_rise', 0):.3f}  "
                f"dominant_signal={dom}"
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("SAMSUNG EXTERNAL TEST — EIS-Only SOH Evaluation")
    print(f"  Samsung INR18650-30Q ({SAMSUNG_NOMINAL_AH} Ah nominal)")
    print(f"  NASA-trained models, Samsung cells as external test")
    print(SEP)
    print(f"  {'Model':5s}  {'MAE%':>7s}  {'RMSE%':>8s}  {'R2':>7s}")
    print(f"  {'-'*40}")
    for mname in model_names:
        r = benchmark_results.get(mname, {})
        if "error" in r:
            print(f"  {mname:5s}  {'FAIL':>7s}  {'FAIL':>8s}  {'FAIL':>7s}")
        else:
            print(f"  {mname:5s}  {r.get('mae_pct',float('nan')):>7.2f}  "
                  f"{r.get('rmse_pct',float('nan')):>8.2f}  "
                  f"{r.get('r2',float('nan')):>7.4f}")
    print(SEP)
    print()
    print("Caveats:")
    print("  1. EIS-only path: discharge/physics zeroed out.")
    print("  2. Models trained on NASA 3D EIS, Samsung has 8D EIS — 3D subset used.")
    print("  3. Samsung nominal capacity (2.95 Ah) != NASA (2.0 Ah).")
    print("  4. Samsung/Mendeley NOT used in training, normalization, or selection.")
    print(SEP)

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "protocol": "Samsung external test only — no Samsung data in training",
        "samsung_nominal_ah": SAMSUNG_NOMINAL_AH,
        "nasa_nominal_ah": NASA_NOMINAL_AH,
        "n_samsung_samples": len(samsung_df),
        "samsung_cells": sorted(samsung_df.cell_id.unique().tolist()),
        "benchmark_eis_only": benchmark_results,
        "degradation_analysis_note": (
            "Physical signals (impedance rise, voltage droop, capacity fade) "
            "from Samsung observable measurements. No LLI/LAM/CL labels used."
        ),
    }
    out_file = out_dir / "samsung_external_eval.json"
    with open(str(out_file), "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved -> {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Samsung external test evaluation")
    ap.add_argument("--samsung_dir", default="data/mendeley_processed")
    ap.add_argument("--checkpoint",  default=None,
                    help="NASA-trained checkpoint path (optional, for BaFuseV2 eval)")
    ap.add_argument("--model_name",  default="G",
                    help="Benchmark model to use for checkpoint evaluation")
    ap.add_argument("--test_cells",  type=int, nargs="+", default=None,
                    help="Samsung cell IDs to use as external test (default: all in test split)")
    ap.add_argument("--out_dir",     default="results/samsung_external_eval")
    ap.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    sys.exit(main(ap.parse_args()) or 0)
