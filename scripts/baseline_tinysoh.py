"""
5-fold CV: TinySOH baseline vs BaFuse (best config).

Uses same stratified fold split as cv_compare.py so results are
directly comparable. Both models use the same DataLoaders (discharge
time-series + EIS + physics) and identical training protocol.

Usage:
    python scripts/baseline_tinysoh.py
    python scripts/baseline_tinysoh.py --epochs 50 --folds 5
"""

import argparse, json, logging, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EPOCHS        = 50
FOLDS         = 5
DEVICE        = "cpu"
SEED          = 42
SUFFICIENT_MIN = 50
BATCH_SIZE    = 32


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_data():
    paired_p = ROOT / "data" / "processed" / "paired.pkl"
    dis_p    = ROOT / "data" / "processed" / "discharge_raw.pkl"
    if paired_p.exists() and dis_p.exists():
        logger.info("Loading cached paired + discharge data")
        return pd.read_pickle(paired_p), pd.read_pickle(dis_p)
    logger.info("Building data from scratch")
    from src.data.parse_mat import parse_all_mat_files
    from src.data.pairing import pair_discharge_eis
    dis, eis = parse_all_mat_files(str(ROOT / "5. BatteryDataSet"))
    paired   = pair_discharge_eis(dis, eis, max_cycle_gap=10)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    dis.to_pickle(dis_p); paired.to_pickle(paired_p)
    return paired, dis


def stratified_kfold(df, k=5):
    """Round-robin split by mean capacity — identical to cv_compare.py."""
    stats = (df.groupby("battery_id")["capacity_ahr"]
             .mean().reset_index()
             .sort_values("capacity_ahr").reset_index(drop=True))
    stats["fold"] = stats.index % k
    folds = []
    for fi in range(k):
        test  = stats[stats["fold"] == fi]["battery_id"].tolist()
        train = stats[stats["fold"] != fi]["battery_id"].tolist()
        folds.append((train, test))
    return folds


def _metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    if len(pred) == 0:
        return dict(mae=np.nan, rmse=np.nan, r2=np.nan, n=0)
    # P6-#2 fix: multiply by 100 before computing mae/rmse
    pred_pct = pred * 100.0
    tgt_pct  = tgt  * 100.0
    mae  = float(np.mean(np.abs(pred_pct - tgt_pct)))
    rmse = float(np.sqrt(np.mean((pred_pct - tgt_pct) ** 2)))
    ss_r = np.sum((tgt - pred) ** 2)
    ss_t = np.sum((tgt - tgt.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))
    return dict(mae=mae, rmse=rmse, r2=r2, n=len(pred))


def _agg(lst, key):
    vals = [x[key] for x in lst if not np.isnan(x.get(key, np.nan))]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, np.nan)


# ── single fold ───────────────────────────────────────────────────────────────

def run_fold(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    dis_df:   pd.DataFrame,
    model_class,
    model_kwargs: dict,
    epochs: int,
    lr: float = 5e-4,
    lambda_physics: float = 0.1,
    lambda_smooth:  float = 0.05,
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train model_class(**model_kwargs) on one fold.
    Returns (metrics, preds, targets, battery_ids).
    """
    from src.data.dataset import create_dataloaders
    from src.losses import SoHPredictionLoss

    # 15 % of train batteries -> val (same ratio as cv_compare.py)
    all_bats = list(train_df["battery_id"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_bats)
    n_val = max(1, int(len(all_bats) * 0.15))
    fold_val   = train_df[train_df["battery_id"].isin(all_bats[:n_val])].reset_index(drop=True)
    fold_train = train_df[train_df["battery_id"].isin(all_bats[n_val:])].reset_index(drop=True)

    if fold_train.empty or fold_val.empty:
        return dict(mae=np.nan, rmse=np.nan, r2=np.nan, n=0, ep=0), \
               np.array([]), np.array([]), np.array([])

    train_loader, val_loader, test_loader = create_dataloaders(
        fold_train, fold_val, test_df,
        discharge_data_df=dis_df,
        batch_size=BATCH_SIZE, num_workers=0,
    )

    device = torch.device(DEVICE)

    # Build model from sample shapes
    sample   = next(iter(train_loader))
    d_shape  = sample["discharge"].shape     # (B, 100, 3)
    e_shape  = sample["eis"].shape           # (B, 3)
    p_shape  = sample["physics"].shape       # (B, 4)

    # Inject shapes into model kwargs
    kw = dict(model_kwargs)
    kw.setdefault("discharge_channels", d_shape[-1])
    kw.setdefault("eis_dim",            e_shape[-1])
    kw.setdefault("physics_dim",        p_shape[-1])

    model = model_class(**kw).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = SoHPredictionLoss(base_loss="mse",
                                  lambda_physics=lambda_physics,
                                  lambda_smooth=lambda_smooth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup = 5
    def _lr(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(epochs - warmup, 1)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr)

    patience = max(10, epochs // 4)
    best_mae = float("inf")
    best_state = None
    p_ctr = 0
    last_ep = 0

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            out  = model(batch["discharge"], batch["eis"], batch["physics"])
            loss = criterion(out["soh_pred"], batch["soh_label"],
                             cycle_age=batch.get("cycle_idx"),
                             battery_ids=batch.get("battery_id"))
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
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            p_ctr = 0
        else:
            p_ctr += 1
        last_ep = ep
        if ep % 10 == 0 or ep == 1:
            logger.info(f"    ep {ep:3d}/{epochs}  val_mae={val_mae*100:.2f}%  best={best_mae*100:.2f}%")
        if p_ctr >= patience:
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    preds, targets, bids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch["discharge"], batch["eis"], batch["physics"])
            preds.extend(out["soh_pred"].view(-1).cpu().tolist())
            targets.extend(batch["soh_label"].view(-1).cpu().tolist())
            b = batch["battery_id"]
            bids.extend(b if isinstance(b, list) else [b] * len(preds))

    preds   = np.array(preds)
    targets = np.array(targets)
    bids    = np.array(bids)
    m = _metrics(preds, targets)
    m["ep"] = last_ep
    m["n_params"] = n_params
    return m, preds, targets, bids


# ── run one model config across all folds ─────────────────────────────────────

def run_config(name, model_class, model_kwargs, paired_df, dis_df, folds, epochs,
               lr=5e-4, lambda_physics=0.1, lambda_smooth=0.05):
    logger.info(f"\n{'='*65}\nModel: {name}\n{'='*65}")

    cnts    = paired_df.groupby("battery_id").size()
    suf_set = set(cnts[cnts >= SUFFICIENT_MIN].index)

    fold_m, suf_m = [], []

    for fi, (train_bats, test_bats) in enumerate(folds):
        train_df = paired_df[paired_df["battery_id"].isin(train_bats)].reset_index(drop=True)
        test_df  = paired_df[paired_df["battery_id"].isin(test_bats)].reset_index(drop=True)

        m, preds, targets, bids = run_fold(
            train_df, test_df, dis_df, model_class, model_kwargs, epochs,
            lr=lr, lambda_physics=lambda_physics, lambda_smooth=lambda_smooth,
        )
        fold_m.append(m)
        logger.info(
            f"  Fold {fi+1}  MAE={m['mae']:.2f}%  RMSE={m['rmse']:.2f}%  "
            f"R2={m['r2']:.4f}  ep={m['ep']}  params={m.get('n_params', '?'):,}"
        )

        suf_mask = np.isin(bids, list(suf_set))
        if suf_mask.sum() > 0:
            sm = _metrics(preds[suf_mask], targets[suf_mask])
            suf_m.append(sm)
            logger.info(f"         [suf] MAE={sm['mae']:.2f}%  R2={sm['r2']:.4f}  n={sm['n']}")

    mae_m,  mae_s  = _agg(fold_m, "mae")
    rmse_m, rmse_s = _agg(fold_m, "rmse")
    r2_m,   r2_s   = _agg(fold_m, "r2")
    sm_m,   sm_s   = _agg(suf_m,  "mae") if suf_m else (np.nan, np.nan)

    logger.info(
        f"\n  ALL   MAE={mae_m:.2f}+/-{mae_s:.2f}%  "
        f"RMSE={rmse_m:.2f}+/-{rmse_s:.2f}%  R2={r2_m:.4f}+/-{r2_s:.4f}"
    )
    logger.info(f"  SUF   MAE={sm_m:.2f}+/-{sm_s:.2f}%")

    return dict(
        name=name,
        per_fold=fold_m,
        summary=dict(mae_mean=mae_m, mae_std=mae_s,
                     rmse_mean=rmse_m, rmse_std=rmse_s,
                     r2_mean=r2_m, r2_std=r2_s),
        suf_summary=dict(mae_mean=sm_m, mae_std=sm_s),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    paired_df, dis_df = _load_data()
    folds = stratified_kfold(paired_df, k=args.folds)

    # ── Mean predictor baseline (no training needed) ──────────────────────
    logger.info("\n" + "="*65)
    logger.info("Baseline: constant mean predictor")
    logger.info("="*65)

    mean_results = []
    for fi, (train_bats, test_bats) in enumerate(folds):
        train_df = paired_df[paired_df["battery_id"].isin(train_bats)]
        test_df  = paired_df[paired_df["battery_id"].isin(test_bats)]
        from src.data.dataset import get_nominal_capacity
        train_soh = (train_df["capacity_ahr"] / train_df["battery_id"]
                     .map(get_nominal_capacity)).clip(0, 1).values
        test_soh  = (test_df["capacity_ahr"] / test_df["battery_id"]
                     .map(get_nominal_capacity)).clip(0, 1).values
        mean_pred = np.full(len(test_soh), train_soh.mean())
        m = _metrics(mean_pred, test_soh)
        mean_results.append(m)
        logger.info(f"  Fold {fi+1}  MAE={m['mae']:.2f}%  R2={m['r2']:.4f}")

    mean_mae_m, mean_mae_s = _agg(mean_results, "mae")
    mean_r2_m,  mean_r2_s  = _agg(mean_results, "r2")
    logger.info(f"  ALL  MAE={mean_mae_m:.2f}+/-{mean_mae_s:.2f}%  R2={mean_r2_m:.4f}+/-{mean_r2_s:.4f}")

    # ── TinySOH ──────────────────────────────────────────────────────────
    # TinySOH is small (~7.6k params): use lower regularization and
    # higher lr to avoid divergence. No monotonicity loss (too noisy for
    # a small model without the LSTM's sequential context).
    from src.models.tiny_soh import TinySOH
    tiny_result = run_config(
        name="TinySOH (CNN1D+MLP, ~15k)",
        model_class=TinySOH,
        model_kwargs=dict(conv_channels=32, hidden_dim=64, dropout=0.2),
        paired_df=paired_df,
        dis_df=dis_df,
        folds=folds,
        epochs=args.epochs,
        lr=1e-3,
        lambda_physics=0.05,
        lambda_smooth=0.0,
    )

    # ── BaFuse (best config c: cutoff + mono, MLP physics) ───────────────
    from src.models.bafuse import BaFuse
    import src.data.dataset as ds_module

    class BaFuseWrapper(torch.nn.Module):
        """Thin wrapper so BaFuse accepts same kwargs as TinySOH."""
        def __init__(self, discharge_channels, eis_dim, physics_dim,
                     latent_dim=64, fusion_dim=128):
            super().__init__()
            self._model = BaFuse(
                discharge_input_size=discharge_channels,
                eis_num_frequencies=eis_dim,
                physics_num_features=physics_dim,
                latent_dim=latent_dim,
                fusion_dim=fusion_dim,
                fusion_method="cross_attention",
                physics_encoder_type="mlp",
            )
        def forward(self, discharge, eis, physics, **kw):
            return self._model(discharge, eis, physics)
        def parameters(self, recurse=True):
            return self._model.parameters(recurse)
        def state_dict(self, **kw):
            return self._model.state_dict(**kw)
        def load_state_dict(self, d, **kw):
            return self._model.load_state_dict(d, **kw)
        def train(self, mode=True):
            self._model.train(mode); return self
        def eval(self):
            self._model.eval(); return self

    # enable per-battery cutoff + within-battery monotonicity
    _orig = getattr(ds_module, "_USE_PER_BATTERY_CUTOFF", True)
    ds_module._USE_PER_BATTERY_CUTOFF = True

    bafuse_result = run_config(
        name="BaFuse (LSTM+CrossAttn, ~1.5M)",
        model_class=BaFuseWrapper,
        model_kwargs=dict(latent_dim=64, fusion_dim=128),
        paired_df=paired_df,
        dis_df=dis_df,
        folds=folds,
        epochs=args.epochs,
        lr=5e-4,
        lambda_physics=0.1,
        lambda_smooth=0.05,
    )
    ds_module._USE_PER_BATTERY_CUTOFF = _orig

    # ── Comparison table ──────────────────────────────────────────────────
    SEP = "=" * 74
    print(f"\n{SEP}")
    print(f"COMPARISON TABLE  ({args.folds}-fold CV, {args.epochs} epochs/fold)")
    print(SEP)
    print(f"  {'Model':36s}  {'MAE (all)':>15s}  {'MAE (suf)':>15s}  {'R2 (all)':>12s}")
    print(f"  {'-'*70}")

    # mean predictor row
    print(f"  {'MeanPredictor (no training)':36s}  "
          f"{mean_mae_m:6.2f}+/-{mean_mae_s:.2f}%  "
          f"{'n/a':>15s}  "
          f"{mean_r2_m:+.4f}")

    for r in [tiny_result, bafuse_result]:
        s  = r["summary"]
        ss = r["suf_summary"]
        mae_str = f"{s['mae_mean']:.2f}+/-{s['mae_std']:.2f}%"
        suf_str = (f"{ss['mae_mean']:.2f}+/-{ss['mae_std']:.2f}%"
                   if not np.isnan(ss["mae_mean"]) else "n/a")
        r2_str  = f"{s['r2_mean']:+.4f}"
        print(f"  {r['name']:36s}  {mae_str:>15s}  {suf_str:>15s}  {r2_str:>12s}")

    print(SEP)

    # delta row: BaFuse - TinySOH
    t_mae = tiny_result["summary"]["mae_mean"]
    b_mae = bafuse_result["summary"]["mae_mean"]
    t_r2  = tiny_result["summary"]["r2_mean"]
    b_r2  = bafuse_result["summary"]["r2_mean"]
    print(f"\n  BaFuse vs TinySOH:  ΔMAE = {b_mae - t_mae:+.2f} pp  "
          f"ΔR2 = {b_r2 - t_r2:+.4f}")
    winner = "BaFuse" if b_mae < t_mae else "TinySOH"
    print(f"  Lower MAE: {winner}")
    print(SEP)

    # save
    out = {
        "mean_predictor": dict(
            summary=dict(mae_mean=mean_mae_m, mae_std=mean_mae_s,
                         r2_mean=mean_r2_m, r2_std=mean_r2_s)),
        "tinysoh": tiny_result,
        "bafuse":  bafuse_result,
    }
    p = ROOT / "results" / "baseline_comparison.json"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"\nSaved -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--folds",  type=int, default=FOLDS)
    ap.add_argument("--device", default=DEVICE, choices=["cpu", "cuda"])
    args = ap.parse_args()
    DEVICE = args.device
    main(args)
