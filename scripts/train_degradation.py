"""
Stage-1: Train EIS encoder + DegradationHead on Mendeley dataset.

This is the first stage of the BaFuse v2 staged training strategy:

    Stage 1 (this script):
        Mendeley EIS → EIS encoder → DegradationHead → LLI/LAM/CL loss
        Only EIS encoder + eis-only degradation head are trained.
        All other BaFuseV2 parameters are frozen.

    Stage 2 (train_bafuse_v2.py --training_mode staged):
        Load Stage-1 EIS weights into BaFuseV2, then train full model
        on NASA data for SOH + optionally joint Mendeley for degradation.

IMPORTANT:
  LLI/LAM/CL labels are model-derived from ECM fitting — NOT physical ground truth.

Prerequisites:
    python scripts/preprocess_mendeley.py   (run first)

Output:
    experiments/degradation/
        best_degradation.pth   — best Stage-1 checkpoint
        history.json           — training history
        metrics.json           — final metrics

Usage:
    python scripts/train_degradation.py
    python scripts/train_degradation.py --config configs/config_bafuse_v2.yaml
    python scripts/train_degradation.py --epochs 30 --device cuda
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
    import pandas as pd

    from src.data.mendeley_dataset import (
        build_mendeley_dataframe,
        split_mendeley_by_cell,
        create_mendeley_dataloaders,
    )
    from src.models.bafuse_v2 import BaFuseV2
    from src.training.degradation_trainer import DegradationTrainer

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed       = int(cfg.get("seed", 42))
    device_str = args.device or cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        device_str = "cpu"

    torch.manual_seed(seed)

    # ── Load or build Mendeley data ──────────────────────────────────────────
    proc_cfg  = cfg.get("processing", {})
    mend_proc = Path(cfg["data"].get("mendeley_processed_dir", "data/mendeley_processed"))

    if (mend_proc / "mendeley_train.pkl").exists() and not args.reprocess:
        logger.info(f"Loading pre-processed Mendeley data from {mend_proc}")
        train_df = pd.read_pickle(str(mend_proc / "mendeley_train.pkl"))
        val_df   = pd.read_pickle(str(mend_proc / "mendeley_val.pkl"))
        test_df  = pd.read_pickle(str(mend_proc / "mendeley_test.pkl"))
    else:
        logger.info("Building Mendeley DataFrame from raw files...")
        mend_dir = cfg["data"].get(
            "mendeley_dir",
            "Lithium-ion cells EIS dataset with fitting and deg/"
            "Lithium-ion cells EIS dataset with fitting and deg",
        )
        df = build_mendeley_dataframe(
            mendeley_root=mend_dir,
            soc_filter=proc_cfg.get("mendeley_soc_filter", 100),
        )
        train_cells = proc_cfg.get("mendeley_train_cells")
        val_cells   = proc_cfg.get("mendeley_val_cells")
        test_cells  = proc_cfg.get("mendeley_test_cells")
        train_df, val_df, test_df = split_mendeley_by_cell(
            df,
            train_cells=train_cells,
            val_cells=val_cells,
            test_cells=test_cells,
            random_state=seed,
        )

    # ── DataLoaders ──────────────────────────────────────────────────────────
    t_cfg = cfg.get("training", {})
    train_loader, val_loader, test_loader, train_stats = create_mendeley_dataloaders(
        train_df, val_df, test_df,
        batch_size=t_cfg.get("batch_size", 16),
        num_workers=0,
    )

    # Detect EIS feature count from first batch
    sample      = next(iter(train_loader))
    eis_dim     = sample["eis"].shape[-1]
    logger.info(f"Mendeley EIS dim: {eis_dim}")

    # ── Build BaFuseV2 model ─────────────────────────────────────────────────
    m_cfg  = cfg.get("model", {})
    model  = BaFuseV2(
        discharge_input_size=m_cfg.get("discharge_input_size", 3),
        eis_num_frequencies=eis_dim,
        physics_num_features=m_cfg.get("physics_num_features", 4),
        latent_dim=m_cfg.get("latent_dim", 64),
        fusion_method=m_cfg.get("fusion_method", "cross_attention"),
        fusion_dim=m_cfg.get("fusion_dim", 128),
        deg_hidden_dim=m_cfg.get("deg_hidden_dim", 128),
        deg_dropout=m_cfg.get("deg_dropout", 0.2),
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"BaFuseV2 total params: {n_params:,}")

    # ── Stage-1 trainer ──────────────────────────────────────────────────────
    deg_cfg = cfg.get("degradation_training", {})
    if args.epochs:
        deg_cfg["num_epochs"] = args.epochs

    trainer = DegradationTrainer(model, deg_config=deg_cfg, device=device_str)
    history = trainer.fit(train_loader, val_loader)

    # ── Save history ─────────────────────────────────────────────────────────
    out_dir = Path(deg_cfg.get("checkpoint_dir", "experiments/degradation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
                   for k, vals in history.items()}, f, indent=2)

    logger.info("=" * 60)
    logger.info("Stage-1 training complete.")
    logger.info(f"  Best checkpoint → {out_dir / 'best_degradation.pth'}")
    logger.info(f"  Best val loss   : {history.get('best_val_loss', '?'):.4f}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage-1: Degradation pre-training")
    p.add_argument("--config",     default="configs/config_bafuse_v2.yaml")
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--device",     default=None, choices=["cpu", "cuda"])
    p.add_argument("--reprocess",  action="store_true",
                   help="Re-parse Mendeley raw files even if pkl exists")
    sys.exit(main(p.parse_args()))
