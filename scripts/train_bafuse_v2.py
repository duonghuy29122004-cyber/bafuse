"""
BaFuse v2 — Stage-2/3 Training Script.

Implements all three training modes:

    nasa_only  (Stage-3 equivalent when no Stage-1 pre-training was done):
        NASA discharge + EIS + physics → SOH
        Equivalent to original BaFuse training. Backward compatible.

    staged (recommended):
        Loads Stage-1 EIS encoder weights from experiments/degradation/best_degradation.pth
        Then trains full BaFuseV2 on NASA for SOH.
        Optionally runs joint Mendeley degradation loss simultaneously.

    joint:
        Trains both SOH (NASA) and degradation (Mendeley) from scratch
        simultaneously, without Stage-1 pre-training.

Training outputs:
    checkpoints/best_model.pth
    results/summary_v2.json
    results/plots/

Usage:
    # NASA-only (backward-compatible with run_pipeline.py):
    python scripts/train_bafuse_v2.py --training_mode nasa_only

    # Staged (after running train_degradation.py):
    python scripts/train_bafuse_v2.py --training_mode staged

    # Joint (both tasks simultaneously):
    python scripts/train_bafuse_v2.py --training_mode joint

    # With GPU:
    python scripts/train_bafuse_v2.py --training_mode staged --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args):
    import yaml
    import torch
    import numpy as np
    import pandas as pd

    from src.data.parse_mat import parse_all_mat_files
    from src.data.pairing import pair_discharge_eis
    from src.data.split import split_by_battery
    from src.data.dataset import create_dataloaders
    from src.data.mendeley_dataset import (
        build_mendeley_dataframe,
        split_mendeley_by_cell,
        create_mendeley_dataloaders,
    )
    from src.models.bafuse_v2 import BaFuseV2
    from src.training.multitask_trainer import MultiTaskTrainer
    from src.evaluate import evaluate, analyze_modality_contribution
    from src.visualization import generate_all_plots
    from evaluation.metrics import compute_metrics, compute_per_battery_metrics

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = args.device or cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — using CPU")
        device_str = "cpu"

    training_mode = args.training_mode or cfg.get("training", {}).get("training_mode", "nasa_only")
    logger.info(f"Training mode: {training_mode}")

    t_cfg  = cfg.get("training", {})
    m_cfg  = cfg.get("model",    {})
    proc   = cfg.get("processing", {})
    out_cfg = cfg.get("output", {})

    if args.epochs:
        t_cfg["num_epochs"] = args.epochs

    # ── Step 1: NASA data ────────────────────────────────────────────────────
    proc_dir = Path(cfg["data"].get("processed_dir", "data/processed"))
    if (proc_dir / "train.pkl").exists() and not args.reprocess:
        logger.info(f"Loading NASA splits from {proc_dir}")
        train_df = pd.read_pickle(str(proc_dir / "train.pkl"))
        val_df   = pd.read_pickle(str(proc_dir / "val.pkl"))
        test_df  = pd.read_pickle(str(proc_dir / "test.pkl"))
        disc_df  = pd.read_pickle(str(proc_dir / "discharge_raw.pkl")) \
                   if (proc_dir / "discharge_raw.pkl").exists() else None
    else:
        logger.info("Parsing NASA .mat files...")
        disc_df, eis_df = parse_all_mat_files(cfg["data"]["raw_dir"])
        paired_df = pair_discharge_eis(disc_df, eis_df, max_cycle_gap=2)
        train_df, val_df, test_df = split_by_battery(
            paired_df,
            train_ratio=proc.get("train_ratio", 0.6),
            val_ratio=proc.get("val_ratio", 0.2),
            test_ratio=proc.get("test_ratio", 0.2),
            random_state=seed,
            stratify_by_soh=True,
        )
        proc_dir.mkdir(parents=True, exist_ok=True)
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            df.to_pickle(str(proc_dir / f"{name}.pkl"))
        disc_df.to_pickle(str(proc_dir / "discharge_raw.pkl"))

    nasa_train, nasa_val, nasa_test = create_dataloaders(
        train_df, val_df, test_df,
        discharge_data_df=disc_df,
        batch_size=t_cfg.get("batch_size", 32),
        num_workers=0,
    )

    sample  = next(iter(nasa_train))
    d_shape = tuple(sample["discharge"].shape)
    e_shape = tuple(sample["eis"].shape)
    p_shape = tuple(sample["physics"].shape)
    logger.info(f"NASA shapes: discharge={d_shape} eis={e_shape} physics={p_shape}")

    # ── Step 2: Mendeley data (if needed) ────────────────────────────────────
    mend_train = mend_val = None
    mend_stats = {}

    if training_mode in ("staged", "joint"):
        mend_proc = Path(cfg["data"].get("mendeley_processed_dir", "data/mendeley_processed"))
        if (mend_proc / "mendeley_train.pkl").exists() and not args.reprocess:
            logger.info(f"Loading Mendeley splits from {mend_proc}")
            m_train_df = pd.read_pickle(str(mend_proc / "mendeley_train.pkl"))
            m_val_df   = pd.read_pickle(str(mend_proc / "mendeley_val.pkl"))
            m_test_df  = pd.read_pickle(str(mend_proc / "mendeley_test.pkl"))
        else:
            mend_dir = cfg["data"].get("mendeley_dir", "")
            m_df = build_mendeley_dataframe(
                mendeley_root=mend_dir,
                soc_filter=proc.get("mendeley_soc_filter", 100),
            )
            m_train_df, m_val_df, m_test_df = split_mendeley_by_cell(
                m_df,
                train_cells=proc.get("mendeley_train_cells"),
                val_cells=proc.get("mendeley_val_cells"),
                test_cells=proc.get("mendeley_test_cells"),
                random_state=seed,
            )
        mend_train, mend_val, _, mend_stats = create_mendeley_dataloaders(
            m_train_df, m_val_df, m_test_df,
            batch_size=t_cfg.get("batch_size", 16),
            num_workers=0,
        )
        mend_eis_dim = next(iter(mend_train))["eis"].shape[-1]
    else:
        mend_eis_dim = e_shape[-1]

    # ── Step 3: Build model ──────────────────────────────────────────────────
    ckpt_dir = Path(out_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t_cfg_full = dict(cfg.get("training", {}))
    t_cfg_full["training_mode"]   = training_mode
    t_cfg_full["lambda_soh"]      = cfg.get("loss", {}).get("lambda_soh", 1.0)
    t_cfg_full["lambda_deg"]      = cfg.get("loss", {}).get("lambda_deg", 0.5)
    t_cfg_full["warmup_epochs"]   = cfg.get("training", {}).get("scheduler", {}).get("warmup_epochs", 5)

    cfg_for_trainer = dict(cfg)
    cfg_for_trainer["training"] = t_cfg_full
    cfg_for_trainer["output"]   = {"checkpoint_dir": str(ckpt_dir)}

    if training_mode == "staged":
        stage1_ckpt = Path(
            cfg.get("degradation_training", {}).get("checkpoint_dir", "experiments/degradation")
        ) / "best_degradation.pth"
        if stage1_ckpt.exists():
            logger.info(f"Loading Stage-1 EIS weights from {stage1_ckpt}")
            model = BaFuseV2.from_bafuse_v1_checkpoint(
                str(stage1_ckpt),
                map_location=device_str,
                discharge_input_size=d_shape[-1],
                eis_num_frequencies=max(e_shape[-1], mend_eis_dim),
                physics_num_features=p_shape[-1],
                latent_dim=m_cfg.get("latent_dim", 64),
                fusion_dim=m_cfg.get("fusion_dim", 128),
            )
        else:
            logger.warning(
                f"Stage-1 checkpoint not found at {stage1_ckpt}. "
                "Initialising BaFuseV2 from scratch."
            )
            model = BaFuseV2(
                discharge_input_size=d_shape[-1],
                eis_num_frequencies=e_shape[-1],
                physics_num_features=p_shape[-1],
                latent_dim=m_cfg.get("latent_dim", 64),
                fusion_dim=m_cfg.get("fusion_dim", 128),
            )
    else:
        model = BaFuseV2(
            discharge_input_size=d_shape[-1],
            eis_num_frequencies=e_shape[-1],
            physics_num_features=p_shape[-1],
            latent_dim=m_cfg.get("latent_dim", 64),
            fusion_method=m_cfg.get("fusion_method", "cross_attention"),
            fusion_dim=m_cfg.get("fusion_dim", 128),
            physics_encoder_type=m_cfg.get("physics_encoder_type", "mlp"),
            deg_hidden_dim=m_cfg.get("deg_hidden_dim", 128),
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"BaFuseV2 trainable params: {n_params:,}")

    # ── Step 4: Train ────────────────────────────────────────────────────────
    trainer = MultiTaskTrainer(model, config=cfg_for_trainer, device=device_str)
    history = trainer.fit(
        nasa_train_loader=nasa_train,
        nasa_val_loader=nasa_val,
        mendeley_train_loader=mend_train,
        mendeley_val_loader=mend_val,
    )

    # ── Step 5: Evaluate ─────────────────────────────────────────────────────
    device = torch.device(device_str)
    nasa_preds = trainer.predict_soh(nasa_test)
    overall    = compute_metrics(nasa_preds["predictions"], nasa_preds["targets"])
    per_bat    = compute_per_battery_metrics(
        nasa_preds["predictions"], nasa_preds["targets"], nasa_preds["battery_ids"]
    )

    logger.info("=" * 60)
    logger.info("Test Results")
    logger.info(f"  MAE  : {overall['mae']:.3f}%")
    logger.info(f"  RMSE : {overall['rmse']:.3f}%")
    logger.info(f"  R²   : {overall['r2']:.4f}")
    logger.info("  Per-battery MAE:")
    for bid, m in per_bat.items():
        logger.info(f"    {bid}: MAE={m['mae']:.3f}%  R²={m['r2']:.4f}")

    # Degradation evaluation (if Mendeley was used)
    deg_results = None
    if mend_val is not None:
        mend_test_loader = mend_val  # use val as proxy for quick eval
        deg_results = trainer.predict_deg(mend_test_loader)
        from evaluation.metrics import compute_degradation_metrics
        deg_met = compute_degradation_metrics(
            deg_results["lli_pred"], deg_results["lli_target"],
            deg_results["lam_pred"], deg_results["lam_target"],
            deg_results["cl_pred"],  deg_results["cl_target"],
        )
        logger.info("  Degradation metrics (model-derived estimates):")
        for k, v in deg_met.items():
            logger.info(f"    {k}: {v:.4f}")

    # ── Step 6: Save results ─────────────────────────────────────────────────
    results_dir = Path(out_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "training_mode":   training_mode,
        "overall_metrics": overall,
        "per_battery":     per_bat,
        "best_val_mae":    history.get("best_val_mae", float("nan")),
    }
    with open(results_dir / "summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ── Step 7: Plots ─────────────────────────────────────────────────────────
    plots_dir = Path(out_cfg.get("plots_dir", "results/plots"))
    try:
        generate_all_plots(
            soh_preds=nasa_preds["predictions"],
            soh_targets=nasa_preds["targets"],
            battery_ids=nasa_preds["battery_ids"],
            history=history,
            ablation_df=None,
            out_dir=str(plots_dir),
            deg_results=deg_results,
        )
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    logger.info(f"Results saved → {results_dir}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BaFuse v2 training")
    p.add_argument("--config",         default="configs/config_bafuse_v2.yaml")
    p.add_argument("--training_mode",  default=None,
                   choices=["nasa_only", "staged", "joint"])
    p.add_argument("--epochs",         type=int, default=None)
    p.add_argument("--device",         default=None, choices=["cpu", "cuda"])
    p.add_argument("--reprocess",      action="store_true")
    sys.exit(main(p.parse_args()))
