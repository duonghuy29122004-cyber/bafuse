"""
CV comparison: 4 configs
  (a) baseline   -- old voltage_droop=fixed 2.7V, cross-battery monotonicity, MLP physics
  (b) +cutoff    -- per-battery cutoff voltage (Task 1)
  (c) +mono_fix  -- within-battery monotonicity (Task 2), stacks on top of (b)
  (d) +cnn1d     -- CNN1D physics encoder (Task 3), stacks on top of (c)

Each config runs 5-fold CV, 40 epochs/fold.
Results saved to results/cv_compare.json
"""
import sys, json, logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EPOCHS   = 40
FOLDS    = 5
DEVICE   = "cpu"
SEED     = 42

SUFFICIENT_MIN = 50


def _load_data():
    cache_p = ROOT / "data" / "processed" / "paired.pkl"
    cache_d = ROOT / "data" / "processed" / "discharge_raw.pkl"
    if cache_p.exists() and cache_d.exists():
        return pd.read_pickle(cache_p), pd.read_pickle(cache_d)
    from src.data.parse_mat import parse_all_mat_files
    from src.data.pairing import pair_discharge_eis
    dis, eis = parse_all_mat_files("5. BatteryDataSet")
    paired   = pair_discharge_eis(dis, eis, max_cycle_gap=10)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    dis.to_pickle(cache_d); paired.to_pickle(cache_p)
    return paired, dis


def stratified_kfold(paired_df, k=5):
    stats = (paired_df.groupby("battery_id")["capacity_ahr"]
             .mean().reset_index()
             .sort_values("capacity_ahr").reset_index(drop=True))
    stats["fold"] = stats.index % k
    folds = []
    for fi in range(k):
        test  = stats[stats["fold"] == fi]["battery_id"].tolist()
        train = stats[stats["fold"] != fi]["battery_id"].tolist()
        folds.append((train, test))
    return folds


def _metrics(pred, tgt):
    mae  = float(np.mean(np.abs(pred - tgt)))
    rmse = float(np.sqrt(np.mean((pred - tgt)**2)))
    ss_r = np.sum((tgt - pred)**2); ss_t = np.sum((tgt - tgt.mean())**2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))
    return dict(mae=mae, rmse=rmse, r2=r2, n=len(pred))


def run_fold(train_df, test_df, dis_df,
             use_per_battery_cutoff: bool,
             within_battery_mono: bool,
             physics_encoder_type: str) -> Tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Train one fold, return (all_metrics, per_battery_metrics, preds, targets, bids)."""
    from src.data.dataset import create_dataloaders
    from src.models.bafuse import BaFuse
    from src.losses import SoHPredictionLoss

    # ── split 15% of train batteries -> val ────────────────────────────────
    bats = list(train_df["battery_id"].unique())
    rng  = np.random.default_rng(SEED)
    rng.shuffle(bats)
    n_val   = max(1, int(len(bats) * 0.15))
    fold_val   = train_df[train_df["battery_id"].isin(bats[:n_val])].reset_index(drop=True)
    fold_train = train_df[train_df["battery_id"].isin(bats[n_val:])].reset_index(drop=True)

    if fold_train.empty or fold_val.empty:
        empty = dict(mae=np.nan, rmse=np.nan, r2=np.nan, n=0, ep=0)
        return empty, {}, np.array([]), np.array([]), np.array([])

    # Temporarily patch dataset to use correct cutoff behaviour
    import src.data.dataset as ds_module
    _orig_flag = getattr(ds_module, "_USE_PER_BATTERY_CUTOFF", True)
    ds_module._USE_PER_BATTERY_CUTOFF = use_per_battery_cutoff

    train_loader, val_loader, test_loader = create_dataloaders(
        fold_train, fold_val, test_df,
        discharge_data_df=dis_df, batch_size=32, num_workers=0)

    ds_module._USE_PER_BATTERY_CUTOFF = _orig_flag  # restore

    sample      = next(iter(train_loader))
    dis_dim     = sample["discharge"].shape[-1]
    eis_dim     = sample["eis"].shape[-1]
    phys_dim    = sample["physics"].shape[-1]
    device      = torch.device(DEVICE)

    model = BaFuse(
        discharge_input_size=dis_dim,
        eis_num_frequencies=eis_dim,
        physics_num_features=phys_dim,
        latent_dim=64, fusion_dim=128,
        physics_encoder_type=physics_encoder_type,
    ).to(device)

    criterion = SoHPredictionLoss(base_loss="mse", lambda_physics=0.1, lambda_smooth=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    warmup = 5

    def _lr(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(EPOCHS - warmup, 1)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr)
    patience = max(10, EPOCHS // 4)
    best_mae = float("inf"); best_state = None; p_ctr = 0; epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            out  = model(batch["discharge"], batch["eis"], batch["physics"])
            bids = batch["battery_id"] if within_battery_mono else None
            loss = criterion(out["soh_pred"], batch["soh_label"],
                             cycle_age=batch.get("cycle_idx"),
                             battery_ids=bids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch["discharge"], batch["eis"], batch["physics"])
                vp.extend(out["soh_pred"].view(-1).cpu().tolist())
                vt.extend(batch["soh_label"].view(-1).cpu().tolist())
        val_mae = float(np.mean(np.abs(np.array(vp) - np.array(vt))))
        if val_mae < best_mae - 1e-3:
            best_mae = val_mae; p_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            p_ctr += 1
        if p_ctr >= patience:
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    preds, targets, bids_out = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch["discharge"], batch["eis"], batch["physics"])
            preds.extend(out["soh_pred"].view(-1).cpu().tolist())
            targets.extend(batch["soh_label"].view(-1).cpu().tolist())
            b = batch["battery_id"]
            bids_out.extend(b if isinstance(b, list) else [b]*len(preds))

    preds   = np.array(preds)
    targets = np.array(targets)
    bids_arr = np.array(bids_out)
    overall = _metrics(preds, targets); overall["ep"] = epoch
    per_bat = {str(bid): _metrics(preds[bids_arr==bid], targets[bids_arr==bid])
               for bid in np.unique(bids_arr)}
    return overall, per_bat, preds, targets, bids_arr


def run_config(name, paired_df, dis_df, folds,
               use_per_battery_cutoff, within_battery_mono, physics_encoder_type):
    logger.info(f"\n{'='*65}")
    logger.info(f"Config: {name}")
    logger.info(f"{'='*65}")

    # identify sufficient batteries
    cnts = paired_df.groupby("battery_id").size()
    suf_bats = set(cnts[cnts >= SUFFICIENT_MIN].index.tolist())

    all_m, suf_m = [], []
    for fi, (train_bats, test_bats) in enumerate(folds):
        train_df = paired_df[paired_df["battery_id"].isin(train_bats)].reset_index(drop=True)
        test_df  = paired_df[paired_df["battery_id"].isin(test_bats)].reset_index(drop=True)

        m, per_bat, preds, targets, bids = run_fold(
            train_df, test_df, dis_df,
            use_per_battery_cutoff, within_battery_mono, physics_encoder_type)
        all_m.append(m)
        logger.info(f"  Fold {fi+1}: MAE={m['mae']:.3f}%  R^2={m['r2']:.4f}  ep={m['ep']}")

        suf_mask = np.isin(bids, list(suf_bats))
        if suf_mask.sum() > 0:
            sm = _metrics(preds[suf_mask], targets[suf_mask])
            suf_m.append(sm)
            logger.info(f"          SUF MAE={sm['mae']:.3f}%  R^2={sm['r2']:.4f}  (n={sm['n']})")

    def _agg(lst, key):
        vals = [x[key] for x in lst if not np.isnan(x[key])]
        return float(np.mean(vals)), float(np.std(vals))

    mae_m, mae_s   = _agg(all_m, "mae")
    rmse_m, rmse_s = _agg(all_m, "rmse")
    r2_m,  r2_s    = _agg(all_m, "r2")
    suf_mae_m, suf_mae_s = _agg(suf_m, "mae") if suf_m else (np.nan, np.nan)

    logger.info(f"\n  ALL  MAE={mae_m:.3f}+/-{mae_s:.3f}%  "
                f"RMSE={rmse_m:.3f}+/-{rmse_s:.3f}%  R^2={r2_m:.4f}+/-{r2_s:.4f}")
    logger.info(f"  SUF  MAE={suf_mae_m:.3f}+/-{suf_mae_s:.3f}%")

    return dict(
        name=name, per_fold=all_m,
        summary=dict(mae_mean=mae_m, mae_std=mae_s,
                     rmse_mean=rmse_m, rmse_std=rmse_s,
                     r2_mean=r2_m, r2_std=r2_s),
        suf_summary=dict(mae_mean=suf_mae_m, mae_std=suf_mae_s),
    )


def main():
    paired_df, dis_df = _load_data()
    folds = stratified_kfold(paired_df, k=FOLDS)

    configs = [
        # (name, per_battery_cutoff, within_battery_mono, physics_encoder_type)
        ("(a) baseline",      False, False, "mlp"),
        ("(b) +cutoff_fix",   True,  False, "mlp"),
        ("(c) +mono_fix",     True,  True,  "mlp"),
        ("(d) +cnn1d_physics",True,  True,  "cnn1d"),
    ]

    results = []
    for name, cutoff, mono, enc in configs:
        r = run_config(name, paired_df, dis_df, folds, cutoff, mono, enc)
        results.append(r)

    # ── Comparison table ──────────────────────────────────────────────────
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("COMPARISON TABLE  (5-fold CV, 40 epochs/fold)")
    print(SEP)
    print(f"  {'Config':28s}  {'MAE (all)':>14s}  {'MAE (suf)':>14s}  {'R^2 (all)':>14s}")
    print(f"  {'-'*68}")
    for r in results:
        s = r["summary"]; ss = r["suf_summary"]
        mae_str = f"{s['mae_mean']:.3f}+/-{s['mae_std']:.3f}%"
        suf_str = (f"{ss['mae_mean']:.3f}+/-{ss['mae_std']:.3f}%"
                   if not np.isnan(ss['mae_mean']) else "  n/a")
        r2_str  = f"{s['r2_mean']:.4f}+/-{s['r2_std']:.4f}"
        print(f"  {r['name']:28s}  {mae_str:>14s}  {suf_str:>14s}  {r2_str:>14s}")
    print(SEP)

    # Best
    best = min(results, key=lambda x: x["summary"]["mae_mean"])
    print(f"\n  Best overall MAE: {best['name']}")

    # Save
    out_path = ROOT / "results" / "cv_compare.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

