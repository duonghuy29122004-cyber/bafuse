"""
BaFuse end-to-end pipeline: data -> train -> evaluate.

Usage:
    python run_pipeline.py                         # full run (CPU)
    python run_pipeline.py --device cuda           # use GPU
    python run_pipeline.py --data_dir "5. BatteryDataSet" --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Paths ----
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main(args):
    from src.data.parse_mat import parse_all_mat_files
    from src.data.pairing import pair_discharge_eis, validate_pairs
    from src.data.split import split_by_battery, get_split_statistics
    from src.data.dataset import create_dataloaders
    from src.models.bafuse import BaFuse
    from src.losses import SoHPredictionLoss
    from src.train import train
    from src.evaluate import evaluate, analyze_modality_contribution

    # =========================================================
    # 1. PARSE
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 1 -- Parse .mat files")
    logger.info("=" * 60)
    discharge_df, eis_df = parse_all_mat_files(args.data_dir)
    logger.info(f"  discharge records : {len(discharge_df):,}")
    logger.info(f"  EIS records       : {len(eis_df):,}")

    if discharge_df.empty:
        logger.error("No discharge data found. Check --data_dir path.")
        return 1

    # Save raw
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    discharge_df.to_pickle(out / "discharge_raw.pkl")
    eis_df.to_pickle(out / "eis_raw.pkl")

    # =========================================================
    # 2. PAIR
    # =========================================================
    logger.info("\nSTEP 2 -- Pair discharge <-> EIS")
    paired_df = pair_discharge_eis(discharge_df, eis_df, max_cycle_gap=2)
    logger.info(f"  pairs created     : {len(paired_df):,}")

    if paired_df.empty:
        logger.error("Pairing produced no results. Check data.")
        return 1

    stats = validate_pairs(paired_df)
    logger.info(f"  batteries         : {stats.get('num_batteries', '?')}")
    logger.info(f"  capacity range    : {stats.get('capacity_range', '?')}")
    paired_df.to_pickle(out / "paired.pkl")

    # =========================================================
    # 3. SPLIT
    # =========================================================
    logger.info("\nSTEP 3 -- Battery-level split (60/20/20)")
    train_df, val_df, test_df = split_by_battery(
        paired_df,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=42,
        stratify_by_soh=True,
    )
    logger.info(f"  train / val / test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    train_df.to_pickle(out / "train.pkl")
    val_df.to_pickle(out / "val.pkl")
    test_df.to_pickle(out / "test.pkl")

    # =========================================================
    # 4. DATALOADERS
    # =========================================================
    logger.info("\nSTEP 4 -- Build DataLoaders")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df,
        val_df,
        test_df,
        discharge_data_df=discharge_df,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Inspect a sample to detect actual feature shapes
    sample = next(iter(train_loader))
    d_shape = sample["discharge"].shape   # (B, seq_len, F) or (B, F)
    e_shape = sample["eis"].shape         # (B, eis_F)
    p_shape = sample["physics"].shape     # (B, phys_F)
    logger.info(f"  discharge shape   : {tuple(d_shape)}")
    logger.info(f"  eis shape         : {tuple(e_shape)}")
    logger.info(f"  physics shape     : {tuple(p_shape)}")

    # Derive input sizes from actual data
    discharge_input_size = d_shape[-1]       # last dim = features
    eis_num_features = e_shape[-1]
    physics_num_features = p_shape[-1]

    # =========================================================
    # 5. BUILD MODEL
    # =========================================================
    logger.info("\nSTEP 5 -- Build BaFuse model")
    device_str = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)

    model = BaFuse(
        discharge_input_size=discharge_input_size,
        eis_num_frequencies=eis_num_features,
        physics_num_features=physics_num_features,
        latent_dim=64,
        fusion_method="cross_attention",
        fusion_dim=128,
        physics_encoder_type=args.physics_encoder,   # Task 3
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  trainable params  : {num_params:,}")

    # =========================================================
    # 6. TRAIN
    # =========================================================
    logger.info("\nSTEP 6 -- Training")
    # Override config with CLI args by patching config.yaml temporarily
    import yaml
    cfg_path = str(ROOT / "configs" / "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Patch cfg to match real shapes and CLI flags
    cfg["model"]["discharge_input_size"] = int(discharge_input_size)
    cfg["model"]["eis_num_frequencies"] = int(eis_num_features)
    cfg["model"]["physics_num_features"] = int(physics_num_features)
    cfg["training"]["num_epochs"] = args.epochs
    cfg["training"]["early_stopping"]["patience"] = args.patience
    cfg["device"] = device_str
    cfg["model"]["physics_encoder_type"] = args.physics_encoder  # Task 3

    tmp_cfg = str(out / "_runtime_config.yaml")
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f)

    trained_model, history = train(
        config_path=tmp_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device_str,
    )
    # BUG FIX: log best-checkpoint metrics, not last-epoch metrics
    logger.info(f"  best val MAE      : {history['best_val_mae']*100:.2f}%")
    logger.info(f"  best val RMSE     : {history['best_val_rmse']*100:.2f}%")
    logger.info(f"  best val R^2      : {history['best_val_r2']:.4f}")

    # =========================================================
    # 7. EVALUATE
    # =========================================================
    logger.info("\nSTEP 7 -- Test set evaluation")
    results = evaluate(trained_model, test_loader, device)
    ov = results["overall"]
    logger.info(f"  Test MAE          : {ov['mae']*100:.2f}%")
    logger.info(f"  Test RMSE         : {ov['rmse']*100:.2f}%")
    logger.info(f"  Test R^2          : {ov['r2']:.4f}")
    logger.info(f"  Test MAPE         : {ov['mape']:.2f}%")

    # Per-battery
    logger.info("  Per-battery MAE:")
    for bid, m in results["per_battery"].items():
        logger.info(f"    {bid}: MAE={m['mae']*100:.2f}%  R^2={m['r2']:.4f}")

    # =========================================================
    # 8. MODALITY CONTRIBUTIONS
    # =========================================================
    logger.info("\nSTEP 8 -- Modality contribution analysis")
    contrib = analyze_modality_contribution(trained_model, test_loader, device)
    logger.info(f"  Discharge : {contrib['discharge']*100:.1f}%")
    logger.info(f"  EIS       : {contrib['eis']*100:.1f}%")
    logger.info(f"  Physics   : {contrib['physics']*100:.1f}%")

    # =========================================================
    # 9. SAVE RESULTS
    # =========================================================
    import json
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    summary = {
        "overall_metrics": ov,
        "per_battery_metrics": results["per_battery"],
        "modality_contributions": {
            k: float(v) for k, v in contrib.items()
        },
        "training_epochs": len(history["train_loss"]),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n  Results saved -> {results_dir / 'summary.json'}")

    logger.info("\n" + "=" * 60)
    logger.info("[OK]  Pipeline complete!")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BaFuse end-to-end pipeline")
    p.add_argument("--data_dir", default="5. BatteryDataSet", help="Battery dataset directory")
    p.add_argument("--output_dir", default="data/processed", help="Output directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--physics_encoder", default="mlp", choices=["mlp", "cnn1d"],
                   help="Physics encoder type: mlp (default) or cnn1d (Task 3 experiment)")
    args = p.parse_args()
    sys.exit(main(args))

