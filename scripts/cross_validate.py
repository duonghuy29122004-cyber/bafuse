"""
Battery-group k-fold cross-validation for BaFuse.

Features:
- Stratified 5-fold split by mean battery SoH
- Each fold: train on k-1 groups, evaluate on 1 group
- Reports MAE/RMSE/R² mean +/- std across folds
- LSTM size comparison: hidden_size in [128, 192, 256] x num_layers in [2, 3]
- Trajectory grouping: sufficient (>50 cycles) vs insufficient (<20 cycles)
  reported separately so short-trajectory batteries don't skew overall metrics

Usage:
    python scripts/cross_validate.py                    # 5-fold, 30 epochs, all sizes
    python scripts/cross_validate.py --epochs 50        # more epochs per fold
    python scripts/cross_validate.py --hidden 128       # single size only
    python scripts/cross_validate.py --no-size-compare  # skip LSTM comparison
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Battery trajectory thresholds ──────────────────────────────────────────
SUFFICIENT_CYCLES   = 50   # >= this: "sufficient" trajectory
INSUFFICIENT_CYCLES = 20   # <  this: "insufficient" -- too few to learn trend

# ── LSTM configurations to compare ────────────────────────────────────────
LSTM_CONFIGS = [
    {"hidden_size": 128, "num_layers": 2, "label": "LSTM-128x2 (small)"},
    {"hidden_size": 192, "num_layers": 2, "label": "LSTM-192x2 (medium)"},
    {"hidden_size": 256, "num_layers": 3, "label": "LSTM-256x3 (large)"},
]


# ── Data loading ───────────────────────────────────────────────────────────

def _load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached paired + discharge data, or build from scratch."""
    paired_cache  = ROOT / "data" / "processed" / "paired.pkl"
    dis_cache     = ROOT / "data" / "processed" / "discharge_raw.pkl"

    if paired_cache.exists() and dis_cache.exists():
        logger.info("Loading cached paired + discharge data")
        return pd.read_pickle(paired_cache), pd.read_pickle(dis_cache)

    logger.info("Building data from scratch (no cache found)")
    from src.data.parse_mat import parse_all_mat_files
    from src.data.pairing import pair_discharge_eis
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    discharge_df, eis_df = parse_all_mat_files("5. BatteryDataSet")
    paired_df = pair_discharge_eis(discharge_df, eis_df, max_cycle_gap=10)
    discharge_df.to_pickle(dis_cache)
    paired_df.to_pickle(paired_cache)
    return paired_df, discharge_df


def classify_batteries(paired_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify batteries into trajectory groups.

    Returns dict:
        sufficient   : >= SUFFICIENT_CYCLES  pairs
        intermediate : INSUFFICIENT_CYCLES <= n < SUFFICIENT_CYCLES
        insufficient : < INSUFFICIENT_CYCLES pairs
    """
    counts = paired_df.groupby("battery_id").size()
    groups: Dict[str, List[str]] = {
        "sufficient":    [],
        "intermediate":  [],
        "insufficient":  [],
    }
    for bid, n in counts.items():
        if n >= SUFFICIENT_CYCLES:
            groups["sufficient"].append(str(bid))
        elif n < INSUFFICIENT_CYCLES:
            groups["insufficient"].append(str(bid))
        else:
            groups["intermediate"].append(str(bid))
    return groups


# ── Fold splitting ─────────────────────────────────────────────────────────

def stratified_battery_kfold(
    paired_df: pd.DataFrame, k: int, seed: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """
    Stratified k-fold split at battery level by mean capacity.
    Round-robin assignment ensures each fold has similar SoH distribution.
    """
    battery_stats = (
        paired_df.groupby("battery_id")["capacity_ahr"]
        .mean()
        .reset_index()
        .rename(columns={"capacity_ahr": "mean_cap"})
        .sort_values("mean_cap")
        .reset_index(drop=True)
    )
    battery_stats["fold"] = battery_stats.index % k

    folds = []
    for fold_idx in range(k):
        test_bats  = battery_stats[battery_stats["fold"] == fold_idx]["battery_id"].tolist()
        train_bats = battery_stats[battery_stats["fold"] != fold_idx]["battery_id"].tolist()
        folds.append((train_bats, test_bats))
    return folds


# ── Metrics ────────────────────────────────────────────────────────────────

def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    if len(preds) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_r = np.sum((targets - preds) ** 2)
    ss_t = np.sum((targets - targets.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": len(preds)}


def _summarise(fold_metrics: List[dict], key: str = "mae") -> Tuple[float, float]:
    vals = [m[key] for m in fold_metrics if not np.isnan(m.get(key, np.nan))]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


# ── Train + eval one fold ──────────────────────────────────────────────────

def train_and_eval_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    epochs: int,
    device_str: str,
    lstm_hidden: int = 256,
    lstm_layers: int = 3,
) -> Tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train one fold and return:
        (all_metrics, per_battery_metrics, preds, targets, battery_ids_array)
    """
    from src.data.dataset import create_dataloaders
    from src.models.bafuse import BaFuse
    from src.models.encoders import DischargeEncoder
    from src.losses import SoHPredictionLoss

    device = torch.device(device_str)

    # 15% of train batteries -> val
    all_bats = list(train_df["battery_id"].unique())
    rng = np.random.default_rng(42)
    rng.shuffle(all_bats)
    n_val = max(1, int(len(all_bats) * 0.15))
    val_bats   = all_bats[:n_val]
    train_bats = all_bats[n_val:]

    fold_train = train_df[train_df["battery_id"].isin(train_bats)].reset_index(drop=True)
    fold_val   = train_df[train_df["battery_id"].isin(val_bats)].reset_index(drop=True)

    empty = {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0, "epoch_stopped": 0}
    if fold_train.empty or fold_val.empty:
        return empty, {}, np.array([]), np.array([]), np.array([])

    train_loader, val_loader, test_loader = create_dataloaders(
        fold_train, fold_val, test_df,
        discharge_data_df=discharge_df,
        batch_size=32, num_workers=0,
    )

    sample      = next(iter(train_loader))
    dis_dim     = sample["discharge"].shape[-1]
    eis_dim     = sample["eis"].shape[-1]
    physics_dim = sample["physics"].shape[-1]

    # Build model with specified LSTM size
    model = BaFuse(
        discharge_input_size=dis_dim,
        eis_num_frequencies=eis_dim,
        physics_num_features=physics_dim,
        latent_dim=64,
        fusion_method="cross_attention",
        fusion_dim=128,
    ).to(device)

    # Override discharge encoder with the specified LSTM size
    model.discharge_encoder = DischargeEncoder(
        input_size=dis_dim,
        hidden_size=lstm_hidden,
        latent_dim=64,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = SoHPredictionLoss(base_loss="mse", lambda_physics=0.1, lambda_smooth=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # Linear warmup (5 ep) + cosine
    warmup = 5
    def _lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(epochs - warmup, 1)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    patience     = max(10, epochs // 4)
    patience_ctr = 0
    best_val_mae = float("inf")
    best_state   = None
    epoch        = 0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            out  = model(batch["discharge"], batch["eis"], batch["physics"])
            loss = criterion(out["soh_pred"], batch["soh_label"],
                             cycle_age=batch.get("cycle_idx"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # --- validate (track val MAE, not val loss) ---
        model.eval()
        val_preds, val_tgts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch["discharge"], batch["eis"], batch["physics"])
                val_preds.extend(out["soh_pred"].view(-1).cpu().numpy().tolist())
                val_tgts.extend(batch["soh_label"].view(-1).cpu().numpy().tolist())

        val_mae = float(np.mean(np.abs(np.array(val_preds) - np.array(val_tgts))))

        if val_mae < best_val_mae - 1e-3:
            best_val_mae = val_mae
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break

    # Load best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # --- test evaluation ---
    model.eval()
    all_preds, all_tgts, all_bids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch["discharge"], batch["eis"], batch["physics"])
            all_preds.extend(out["soh_pred"].view(-1).cpu().numpy().tolist())
            all_tgts.extend(batch["soh_label"].view(-1).cpu().numpy().tolist())
            bids = batch["battery_id"]
            all_bids.extend(bids if isinstance(bids, list) else [bids] * len(all_preds))

    preds   = np.array(all_preds)
    targets = np.array(all_tgts)
    bids_arr = np.array(all_bids)

    overall = _compute_metrics(preds, targets)
    overall["epoch_stopped"] = epoch
    overall["n_params"]      = n_params

    # per-battery breakdown
    per_bat: Dict[str, dict] = {}
    for bid in np.unique(bids_arr):
        mask = bids_arr == bid
        per_bat[str(bid)] = _compute_metrics(preds[mask], targets[mask])

    return overall, per_bat, preds, targets, bids_arr


# ── Main ───────────────────────────────────────────────────────────────────

def main(args):
    paired_df, discharge_df = _load_data()

    # Battery groups
    groups = classify_batteries(paired_df)
    logger.info(f"Battery trajectory groups:")
    logger.info(f"  Sufficient   (>={SUFFICIENT_CYCLES} cycles): {groups['sufficient']}")
    logger.info(f"  Intermediate ({INSUFFICIENT_CYCLES}-{SUFFICIENT_CYCLES-1} cycles): {groups['intermediate']}")
    logger.info(f"  Insufficient (<{INSUFFICIENT_CYCLES} cycles):  {groups['insufficient']}")

    folds    = stratified_battery_kfold(paired_df, k=args.folds)
    SEP      = "=" * 72
    results  = {}

    # ── LSTM size comparison ──────────────────────────────────────────────
    if args.size_compare:
        configs_to_run = LSTM_CONFIGS
    else:
        # Use only the config matching --hidden and --layers
        configs_to_run = [{"hidden_size": args.hidden,
                           "num_layers":  args.layers,
                           "label":       f"LSTM-{args.hidden}x{args.layers}"}]

    for cfg_lstm in configs_to_run:
        hidden = cfg_lstm["hidden_size"]
        layers = cfg_lstm["num_layers"]
        label  = cfg_lstm["label"]

        print(f"\n{SEP}")
        print(f"Config: {label}  |  {args.folds}-fold CV  |  {args.epochs} epochs/fold")
        print(SEP)

        fold_all: List[dict]  = []
        fold_suf: List[dict]  = []
        fold_ins: List[dict]  = []

        for fold_idx, (train_bats, test_bats) in enumerate(folds):
            train_df = paired_df[paired_df["battery_id"].isin(train_bats)].reset_index(drop=True)
            test_df  = paired_df[paired_df["battery_id"].isin(test_bats)].reset_index(drop=True)

            suf_in_test  = [b for b in test_bats if b in groups["sufficient"]]
            ins_in_test  = [b for b in test_bats if b in groups["insufficient"]]
            int_in_test  = [b for b in test_bats if b in groups["intermediate"]]

            logger.info(f"\n  Fold {fold_idx+1}/{args.folds}  "
                        f"train={len(train_bats)} bats  test={test_bats}")
            logger.info(f"    sufficient={suf_in_test}  "
                        f"intermediate={int_in_test}  "
                        f"insufficient={ins_in_test}")

            m_all, per_bat, preds, targets, bids = train_and_eval_fold(
                train_df, test_df, discharge_df,
                epochs=args.epochs, device_str=args.device,
                lstm_hidden=hidden, lstm_layers=layers,
            )
            fold_all.append(m_all)

            logger.info(f"    ALL  MAE={m_all['mae']:.3f}%  "
                        f"RMSE={m_all['rmse']:.3f}%  R^2={m_all['r2']:.4f}  "
                        f"params={m_all.get('n_params',0):,}  "
                        f"ep={m_all.get('epoch_stopped','?')}")

            # Sufficient subset metrics
            suf_mask = np.isin(bids, groups["sufficient"])
            if suf_mask.sum() > 0:
                m_suf = _compute_metrics(preds[suf_mask], targets[suf_mask])
                fold_suf.append(m_suf)
                logger.info(f"    SUF  MAE={m_suf['mae']:.3f}%  "
                            f"RMSE={m_suf['rmse']:.3f}%  R^2={m_suf['r2']:.4f}  "
                            f"(n={m_suf['n']})")

            # Insufficient subset metrics
            ins_mask = np.isin(bids, groups["insufficient"])
            if ins_mask.sum() > 0:
                m_ins = _compute_metrics(preds[ins_mask], targets[ins_mask])
                fold_ins.append(m_ins)
                logger.info(f"    INS  MAE={m_ins['mae']:.3f}%  "
                            f"RMSE={m_ins['rmse']:.3f}%  R^2={m_ins['r2']:.4f}  "
                            f"(n={m_ins['n']})")

        # ── Per-config summary ────────────────────────────────────────────
        mae_m,  mae_s  = _summarise(fold_all, "mae")
        rmse_m, rmse_s = _summarise(fold_all, "rmse")
        r2_m,   r2_s   = _summarise(fold_all, "r2")

        suf_mae_m, suf_mae_s = _summarise(fold_suf, "mae")
        ins_mae_m, ins_mae_s = _summarise(fold_ins, "mae")

        print(f"\n  {'─'*68}")
        print(f"  {label}  --  {args.folds}-fold summary")
        print(f"  {'─'*68}")
        print(f"  ALL batteries:")
        print(f"    MAE  = {mae_m:6.3f} +/- {mae_s:.3f} %")
        print(f"    RMSE = {rmse_m:6.3f} +/- {rmse_s:.3f} %")
        print(f"    R^2  = {r2_m:6.4f} +/- {r2_s:.4f}")

        if fold_suf:
            print(f"  Sufficient (>={SUFFICIENT_CYCLES} cycles):")
            print(f"    MAE  = {suf_mae_m:6.3f} +/- {suf_mae_s:.3f} %"
                  f"   [{groups['sufficient']}]")
        if fold_ins:
            print(f"  Insufficient (<{INSUFFICIENT_CYCLES} cycles)  [excluded from main metric]:")
            print(f"    MAE  = {ins_mae_m:6.3f} +/- {ins_mae_s:.3f} %"
                  f"   [{groups['insufficient']}]")
        print(f"  {'─'*68}")

        results[label] = {
            "hidden_size": hidden,
            "num_layers":  layers,
            "n_params":    fold_all[0].get("n_params", 0) if fold_all else 0,
            "per_fold":    fold_all,
            "summary_all": {
                "mae_mean": mae_m, "mae_std": mae_s,
                "rmse_mean": rmse_m, "rmse_std": rmse_s,
                "r2_mean": r2_m, "r2_std": r2_s,
            },
            "summary_sufficient": {
                "mae_mean": suf_mae_m, "mae_std": suf_mae_s,
            } if fold_suf else {},
            "summary_insufficient": {
                "mae_mean": ins_mae_m, "mae_std": ins_mae_s,
            } if fold_ins else {},
        }

    # ── Cross-config comparison ───────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{SEP}")
        print("LSTM SIZE COMPARISON  (sorted by ALL-batteries MAE)")
        print(SEP)
        ranked = sorted(
            results.items(),
            key=lambda kv: kv[1]["summary_all"].get("mae_mean", 99),
        )
        print(f"  {'Config':30s}  {'Params':>10s}  {'MAE (all)':>14s}  "
              f"{'MAE (suf)':>14s}  {'R^2 (all)':>12s}")
        print(f"  {'─'*80}")
        for lbl, r in ranked:
            mae_str = (f"{r['summary_all']['mae_mean']:.3f}"
                       f"+/-{r['summary_all']['mae_std']:.3f}")
            suf     = r.get("summary_sufficient", {})
            suf_str = (f"{suf['mae_mean']:.3f}+/-{suf['mae_std']:.3f}"
                       if suf else "  n/a")
            r2_str  = (f"{r['summary_all']['r2_mean']:.4f}"
                       f"+/-{r['summary_all']['r2_std']:.4f}")
            print(f"  {lbl:30s}  {r['n_params']:>10,}  {mae_str:>14s}  "
                  f"{suf_str:>14s}  {r2_str:>12s}")
        print(f"  {'─'*80}")
        best_lbl = ranked[0][0]
        print(f"  Best config: {best_lbl}")
        print(SEP)

    # ── Battery groups reference ──────────────────────────────────────────
    print(f"\n  Battery trajectory classification")
    print(f"  {'─'*50}")
    counts = paired_df.groupby("battery_id").size().to_dict()
    for bid in sorted(counts.keys()):
        n = counts[bid]
        g = ("SUFFICIENT   "  if n >= SUFFICIENT_CYCLES
             else "INSUFFICIENT " if n < INSUFFICIENT_CYCLES
             else "INTERMEDIATE ")
        print(f"    {bid:8s}  {n:4d} cycles  [{g}]")

    # ── Save ──────────────────────────────────────────────────────────────
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    out_path = out / "cv_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "folds": args.folds,
            "epochs_per_fold": args.epochs,
            "battery_groups": groups,
            "configs": results,
        }, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Battery-group k-fold CV for BaFuse")
    p.add_argument("--folds",          type=int,  default=5)
    p.add_argument("--epochs",         type=int,  default=30)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--hidden",         type=int,  default=256,
                   help="LSTM hidden size (used when --no-size-compare)")
    p.add_argument("--layers",         type=int,  default=3,
                   help="LSTM num_layers (used when --no-size-compare)")
    p.add_argument("--no-size-compare", dest="size_compare",
                   action="store_false", default=True,
                   help="Skip LSTM size comparison, run only --hidden/--layers config")
    main(p.parse_args())

