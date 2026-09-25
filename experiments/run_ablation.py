"""
BaFuse v2 Ablation Runner.

Trains and evaluates seven ablation configurations (A1–A7) using the same
data split, random seed, and training protocol.

Ablation matrix:
    A1  full            discharge=T  eis=T  physics=T  (baseline)
    A2  no_discharge    discharge=F  eis=T  physics=T
    A3  no_eis          discharge=T  eis=F  physics=T
    A4  no_physics      discharge=T  eis=T  physics=F
    A5  discharge_only  discharge=T  eis=F  physics=F
    A6  eis_only        discharge=F  eis=T  physics=F
    A7  physics_only    discharge=F  eis=F  physics=T

Each experiment:
  - Uses the SAME train/val/test split.
  - Uses the SAME random seed.
  - Trains a fresh BaFuseV2 (retrain-based ablation, not zero-masking).
  - Saves config, checkpoint, predictions, metrics, runtime.

NOTE: Retrain-based ablation is methodologically stronger than zero-masking
because it allows remaining modalities to compensate. Zero-masking inference
is available in BaFuseV2.set_modality_flags() for quick sensitivity checks.

Usage:
    python experiments/run_ablation.py \\
        --data_dir "5. BatteryDataSet" \\
        --config   configs/config_bafuse_v2.yaml \\
        --out_dir  experiments/ablation \\
        --epochs   50 \\
        --seed     42

    # Run a single experiment only:
    python experiments/run_ablation.py --exp A3_no_eis

    # Skip training, only re-evaluate saved checkpoints:
    python experiments/run_ablation.py --eval_only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.data.parse_mat import parse_all_mat_files
from src.data.pairing import pair_discharge_eis
from src.data.split import split_by_battery
from src.data.dataset import create_dataloaders
from src.models.bafuse_v2 import BaFuseV2
from src.losses import SoHPredictionLoss
from evaluation.ablation_metrics import AblationAnalyzer, ABLATION_EXPERIMENTS
from evaluation.metrics import compute_metrics, compute_per_battery_metrics

# ── Ablation experiment definitions ──────────────────────────────────────────

def _build_model(
    d_shape, e_shape, p_shape,
    use_discharge: bool,
    use_eis:       bool,
    use_physics:   bool,
    cfg:           Dict,
    device:        torch.device,
) -> BaFuseV2:
    m_cfg = cfg.get("model", {})
    model = BaFuseV2(
        discharge_input_size = d_shape[-1],
        eis_num_frequencies  = e_shape[-1],
        physics_num_features = p_shape[-1],
        latent_dim           = m_cfg.get("latent_dim", 64),
        fusion_method        = m_cfg.get("fusion_method", "cross_attention"),
        fusion_dim           = m_cfg.get("fusion_dim", 128),
        physics_encoder_type = m_cfg.get("physics_encoder_type", "mlp"),
        use_discharge        = use_discharge,
        use_eis              = use_eis,
        use_physics          = use_physics,
    ).to(device)
    return model


def _train_one(
    exp_id:        str,
    use_discharge: bool,
    use_eis:       bool,
    use_physics:   bool,
    train_loader,
    val_loader,
    d_shape, e_shape, p_shape,
    cfg:     Dict,
    device:  torch.device,
    out_dir: Path,
    seed:    int,
) -> Dict:
    """Train one ablation configuration. Returns metrics dict."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    exp_dir = out_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save this experiment's config snapshot
    exp_cfg = {
        "experiment": exp_id,
        "use_discharge": use_discharge,
        "use_eis": use_eis,
        "use_physics": use_physics,
        "seed": seed,
        "model_config": cfg.get("model", {}),
        "training_config": cfg.get("training", {}),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(exp_cfg, f, indent=2)

    model = _build_model(
        d_shape, e_shape, p_shape,
        use_discharge, use_eis, use_physics, cfg, device
    )

    t_cfg  = cfg.get("training", {})
    l_cfg  = cfg.get("loss", {})
    lr     = float(t_cfg.get("learning_rate", 5e-4))
    wd     = float(t_cfg.get("weight_decay",  1e-5))
    epochs = int(t_cfg.get("num_epochs", 50))
    pat    = int(t_cfg.get("early_stopping", {}).get("patience", 10))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = SoHPredictionLoss(
        base_loss=l_cfg.get("base_loss", "mse"),
        lambda_physics=float(l_cfg.get("lambda_physics", 0.1)),
        lambda_smooth=float(l_cfg.get("lambda_smooth", 0.05)),
    )

    best_val_mae   = float("inf")
    patience_ctr   = 0
    train_history  = {"train_loss": [], "val_mae": [], "val_rmse": [], "val_r2": []}
    start_t        = time.time()

    for epoch in range(1, epochs + 1):
        # ---- train --------------------------------------------------------
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            d = batch["discharge"].to(device)
            e = batch["eis"].to(device)
            p = batch["physics"].to(device)
            y = batch["soh_label"].to(device)
            ci = batch.get("cycle_idx")
            if ci is not None:
                ci = ci.to(device)
            bids = batch.get("battery_id")

            optimizer.zero_grad()
            out  = model(d, e, p)
            loss = criterion(out["soh_pred"], y, cycle_age=ci, battery_ids=bids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()

        # ---- validate -----------------------------------------------------
        model.eval()
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for batch in val_loader:
                d = batch["discharge"].to(device)
                e = batch["eis"].to(device)
                p = batch["physics"].to(device)
                y = batch["soh_label"].to(device)
                out = model(d, e, p)
                all_preds.append(out["soh_pred"].view(-1).cpu().numpy())
                all_tgts.append(y.view(-1).cpu().numpy())

        preds   = np.concatenate(all_preds)
        targets = np.concatenate(all_tgts)
        met     = compute_metrics(preds, targets)
        avg_loss = ep_loss / max(len(train_loader), 1)

        train_history["train_loss"].append(avg_loss)
        train_history["val_mae"].append(met["mae"])
        train_history["val_rmse"].append(met["rmse"])
        train_history["val_r2"].append(met["r2"])

        logger.info(
            f"[{exp_id}] Epoch {epoch:3d}/{epochs}  "
            f"loss={avg_loss:.4f}  MAE={met['mae']:.3f}  "
            f"RMSE={met['rmse']:.3f}  R²={met['r2']:.4f}"
        )

        if met["mae"] < best_val_mae - 1e-3:
            best_val_mae = met["mae"]
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": met,
                    "config": exp_cfg,
                },
                str(exp_dir / "best_model.pth"),
            )
        else:
            patience_ctr += 1
            if patience_ctr >= pat:
                logger.info(f"[{exp_id}] Early stop at epoch {epoch}")
                break

    elapsed = time.time() - start_t

    # ---- test evaluation --------------------------------------------------
    # Load best checkpoint
    ckpt = torch.load(
        str(exp_dir / "best_model.pth"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_preds, test_tgts, test_bids = [], [], []
    # NOTE: val_loader is used here; call with test_loader for final evaluation
    with torch.no_grad():
        for batch in val_loader:
            d = batch["discharge"].to(device)
            e = batch["eis"].to(device)
            p = batch["physics"].to(device)
            y = batch["soh_label"].to(device)
            out = model(d, e, p)
            test_preds.append(out["soh_pred"].view(-1).cpu().numpy())
            test_tgts.append(y.view(-1).cpu().numpy())
            test_bids.extend(
                batch["battery_id"]
                if isinstance(batch["battery_id"], list)
                else ["?"] * len(out["soh_pred"])
            )

    preds   = np.concatenate(test_preds)
    targets = np.concatenate(test_tgts)
    met     = compute_metrics(preds, targets)
    per_bat = compute_per_battery_metrics(preds, targets, test_bids)

    # Save predictions
    np.save(str(exp_dir / "predictions.npy"), preds)
    np.save(str(exp_dir / "targets.npy"),     targets)

    # Save training history
    with open(exp_dir / "history.json", "w") as f:
        json.dump(train_history, f, indent=2)

    # Save metrics
    result = {
        "experiment":    exp_id,
        "discharge":     use_discharge,
        "eis":           use_eis,
        "physics":       use_physics,
        "mae":           met["mae"],
        "rmse":          met["rmse"],
        "r2":            met["r2"],
        "elapsed_sec":   elapsed,
        "seed":          seed,
        "per_battery":   per_bat,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"[{exp_id}] DONE  MAE={met['mae']:.3f}  "
        f"RMSE={met['rmse']:.3f}  R²={met['r2']:.4f}  "
        f"({elapsed:.0f}s)"
    )
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed       = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    device     = torch.device(device_str)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Override epochs from CLI ──────────────────────────────────────────
    if args.epochs:
        cfg.setdefault("training", {})["num_epochs"] = args.epochs

    # ── Data ─────────────────────────────────────────────────────────────
    if not args.eval_only:
        logger.info("Parsing NASA data...")
        discharge_df, eis_df = parse_all_mat_files(args.data_dir)
        paired_df = pair_discharge_eis(discharge_df, eis_df, max_cycle_gap=2)
        train_df, val_df, test_df = split_by_battery(
            paired_df,
            train_ratio=cfg.get("processing", {}).get("train_ratio", 0.6),
            val_ratio=cfg.get("processing", {}).get("val_ratio", 0.2),
            test_ratio=cfg.get("processing", {}).get("test_ratio", 0.2),
            random_state=seed,
            stratify_by_soh=True,
        )

        # Save splits for reproducibility
        import pickle
        proc_dir = Path(args.data_dir).parent / "data" / "ablation_splits"
        proc_dir.mkdir(parents=True, exist_ok=True)
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            df.to_pickle(str(proc_dir / f"{name}.pkl"))

        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df,
            discharge_data_df=discharge_df,
            batch_size=cfg.get("training", {}).get("batch_size", 32),
            num_workers=0,
        )

        sample = next(iter(train_loader))
        d_shape = tuple(sample["discharge"].shape)
        e_shape = tuple(sample["eis"].shape)
        p_shape = tuple(sample["physics"].shape)
        logger.info(f"Shapes: discharge={d_shape} eis={e_shape} physics={p_shape}")

        # Save shape info for eval_only mode
        with open(out_dir / "_shapes.json", "w") as f:
            json.dump({"d": list(d_shape), "e": list(e_shape), "p": list(p_shape)}, f)
    else:
        # Load shapes from previous run
        with open(out_dir / "_shapes.json") as f:
            shapes = json.load(f)
        d_shape = tuple(shapes["d"])
        e_shape = tuple(shapes["e"])
        p_shape = tuple(shapes["p"])
        # Loaders not needed for eval_only — placeholder
        train_loader = val_loader = test_loader = None

    # ── Determine which experiments to run ───────────────────────────────
    exps_to_run = ABLATION_EXPERIMENTS
    if args.exp:
        exps_to_run = [e for e in ABLATION_EXPERIMENTS if e["id"] == args.exp]
        if not exps_to_run:
            logger.error(f"Unknown experiment ID: {args.exp}")
            return 1

    # ── Run / collect results ────────────────────────────────────────────
    analyzer  = AblationAnalyzer()
    all_results: List[Dict] = []

    for exp in exps_to_run:
        exp_id = exp["id"]
        if args.eval_only:
            # Load saved metrics
            m_path = out_dir / exp_id / "metrics.json"
            if m_path.exists():
                with open(m_path) as f:
                    result = json.load(f)
                preds   = np.load(str(out_dir / exp_id / "predictions.npy"))
                targets = np.load(str(out_dir / exp_id / "targets.npy"))
            else:
                logger.warning(f"No saved metrics for {exp_id}, skipping")
                continue
        else:
            result = _train_one(
                exp_id=exp_id,
                use_discharge=exp["discharge"],
                use_eis=exp["eis"],
                use_physics=exp["physics"],
                train_loader=train_loader,
                val_loader=val_loader,
                d_shape=d_shape,
                e_shape=e_shape,
                p_shape=p_shape,
                cfg=cfg,
                device=device,
                out_dir=out_dir,
                seed=seed,
            )
            preds   = np.load(str(out_dir / exp_id / "predictions.npy"))
            targets = np.load(str(out_dir / exp_id / "targets.npy"))

        analyzer.add_result(
            experiment_id=exp_id,
            predictions=preds,
            targets=targets,
            discharge=exp["discharge"],
            eis=exp["eis"],
            physics=exp["physics"],
            extra_info={
                "elapsed_sec": result.get("elapsed_sec", 0),
                "seed": seed,
            },
        )
        all_results.append(result)

    # ── Save summary ─────────────────────────────────────────────────────
    analyzer.save_csv(str(out_dir / "ablation_results.csv"))
    analyzer.save_json(str(out_dir / "ablation_results.json"))
    analyzer.print_table()

    logger.info(f"\nAblation complete. Results → {out_dir}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BaFuse v2 ablation runner")
    p.add_argument("--data_dir",  default="5. BatteryDataSet")
    p.add_argument("--config",    default="configs/config_bafuse_v2.yaml")
    p.add_argument("--out_dir",   default="experiments/ablation")
    p.add_argument("--epochs",    type=int, default=None)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--exp",       default=None,
                   help="Run a single experiment by ID (e.g. A3_no_eis)")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training, re-evaluate saved checkpoints")
    sys.exit(main(p.parse_args()))
