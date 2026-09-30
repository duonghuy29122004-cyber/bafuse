"""
Leak-free baselines on the same NASA battery-level split as BaFuse v2.

Replaces the constant-mean "baseline" and the leaky sklearn pipeline
(capacity_fade used the SOH label).

Models
------
  mean              : train-mean SOH (reference only)
  ridge             : linear, scaled
  hist_gb           : HistGradientBoosting (primary tabular baseline)
  random_forest     : RF
  tiny_mlp          : sklearn MLP (64, 32)
  tiny_soh          : Conv1d + MLP on discharge curves (~15k params)

Usage
-----
  python scripts/train_baselines.py
  python scripts/train_baselines.py --skip_tiny_soh
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / (ss_t + 1e-8))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mae_pct": mae * 100.0,
        "rmse_pct": rmse * 100.0,
    }


def _per_battery(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    tmp = df.copy()
    tmp["_y"] = y_true
    tmp["_p"] = y_pred
    for bid, grp in tmp.groupby("battery_id"):
        out[str(bid)] = _metrics(grp["_y"].values, grp["_p"].values)
    return out


def _train_tiny_soh(device: str, epochs: int, patience: int) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import BaFuseDataset, fit_aging_prior
    from src.losses import SoHPredictionLoss
    from src.models.tiny_soh import TinySOH

    proc = ROOT / "data" / "processed"
    train_df = pd.read_pickle(proc / "train.pkl")
    val_df = pd.read_pickle(proc / "val.pkl")
    test_df = pd.read_pickle(proc / "test.pkl")
    discharge_raw = pd.read_pickle(proc / "discharge_raw.pkl")

    aging = fit_aging_prior(train_df)
    train_ds = BaFuseDataset(train_df, discharge_data_df=discharge_raw, aging_prior_params=aging)
    val_ds = BaFuseDataset(
        val_df,
        discharge_data_df=discharge_raw,
        external_stats=train_ds.stats,
        aging_prior_params=aging,
    )
    test_ds = BaFuseDataset(
        test_df,
        discharge_data_df=discharge_raw,
        external_stats=train_ds.stats,
        aging_prior_params=aging,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device_t = torch.device(device)
    model = TinySOH().to(device_t)
    n_params = model.count_parameters()
    logger.info("TinySOH params: %s", f"{n_params:,}")

    criterion = SoHPredictionLoss(base_loss="mse", lambda_physics=0.1, lambda_smooth=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val = float("inf")
    best_state = None
    wait = 0

    def _run(loader, train: bool):
        model.train(train)
        preds, tgts, bids = [], [], []
        total = 0.0
        n = 0
        for batch in loader:
            discharge = batch["discharge"].to(device_t)
            eis = batch["eis"].to(device_t)
            physics = batch["physics"].to(device_t)
            y = batch["soh_label"].to(device_t)
            cycle = batch["cycle_idx"].to(device_t)
            bids_batch = batch["battery_id"]
            if train:
                opt.zero_grad()
            out = model(discharge, eis, physics)
            pred = out["soh_pred"].view(-1)
            loss = criterion(pred, y, cycle_age=cycle, battery_ids=list(bids_batch))
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += float(loss.item()) * y.size(0)
            n += y.size(0)
            preds.append(pred.detach().cpu().numpy())
            tgts.append(y.detach().cpu().numpy())
            bids.extend(list(bids_batch) if not isinstance(bids_batch, list) else bids_batch)
        p = np.concatenate(preds)
        t = np.concatenate(tgts)
        m = _metrics(t, p)
        m["loss"] = total / max(n, 1)
        return m, p, t, bids

    for epoch in range(1, epochs + 1):
        tr, _, _, _ = _run(train_loader, True)
        va, _, _, _ = _run(val_loader, False)
        logger.info(
            "TinySOH epoch %03d  train MAE%%=%.2f  val MAE%%=%.2f  val R2=%.3f",
            epoch, tr["mae_pct"], va["mae_pct"], va["r2"],
        )
        if va["mae"] + 1e-6 < best_val:
            best_val = va["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("TinySOH early stop at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    ckpt_dir = ROOT / "experiments" / "baseline"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": best_state, "n_params": n_params},
        ckpt_dir / "tiny_soh.pth",
    )

    val_m, _, _, _ = _run(val_loader, False)
    test_m, test_p, test_t, test_bids = _run(test_loader, False)
    per = {}
    tmp = pd.DataFrame({"battery_id": test_bids, "y": test_t, "p": test_p})
    for bid, grp in tmp.groupby("battery_id"):
        per[str(bid)] = _metrics(grp["y"].values, grp["p"].values)
    test_m["per_battery"] = per
    return {
        "n_params": n_params,
        "val": {k: v for k, v in val_m.items() if k != "loss"},
        "test": {k: v for k, v in test_m.items() if k != "loss"},
        "best_val_mae": best_val,
    }


def main(args) -> int:
    from src.data.tabular_features import (
        attach_duration_from_raw,
        build_xy,
        fit_feature_stats,
    )

    proc = ROOT / "data" / "processed"
    train_df = pd.read_pickle(proc / "train.pkl")
    val_df = pd.read_pickle(proc / "val.pkl")
    test_df = pd.read_pickle(proc / "test.pkl")
    discharge_raw = pd.read_pickle(proc / "discharge_raw.pkl")

    train_df = attach_duration_from_raw(train_df, discharge_raw)
    val_df = attach_duration_from_raw(val_df, discharge_raw)
    test_df = attach_duration_from_raw(test_df, discharge_raw)
    dur_med = float(train_df["duration_s"].median())
    for frame in (train_df, val_df, test_df):
        frame["duration_s"] = frame["duration_s"].fillna(dur_med)

    stats = fit_feature_stats(train_df)
    X_tr, y_tr, cols = build_xy(train_df, stats)
    X_va, y_va, _ = build_xy(val_df, stats)
    X_te, y_te, _ = build_xy(test_df, stats)
    logger.info("Tabular features (%d): %s", len(cols), cols)
    logger.info("Shapes train/val/test: %s %s %s", X_tr.shape, X_va.shape, X_te.shape)

    models = {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "hist_gb": HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=0.1,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "tiny_mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=400,
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )),
        ]),
    }

    results = {
        "features": cols,
        "notes": (
            "Leak-free: no capacity in X. duration_s from raw discharge. "
            "Same battery-level split as BaFuse v2 (data/processed/*.pkl). "
            "SOH in [0,1]; mae_pct = mae*100."
        ),
        "models": {},
    }

    mean_soh = float(y_tr.mean())
    results["models"]["mean"] = {
        "val": _metrics(y_va, np.full_like(y_va, mean_soh)),
        "test": _metrics(y_te, np.full_like(y_te, mean_soh)),
        "test_per_battery": _per_battery(test_df, y_te, np.full_like(y_te, mean_soh)),
    }
    logger.info(
        "[mean] val MAE%%=%.2f R2=%.3f | test MAE%%=%.2f R2=%.3f",
        results["models"]["mean"]["val"]["mae_pct"],
        results["models"]["mean"]["val"]["r2"],
        results["models"]["mean"]["test"]["mae_pct"],
        results["models"]["mean"]["test"]["r2"],
    )

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_va)
        test_pred = model.predict(X_te)
        entry = {
            "val": _metrics(y_va, val_pred),
            "test": _metrics(y_te, test_pred),
            "test_per_battery": _per_battery(test_df, y_te, test_pred),
        }
        results["models"][name] = entry
        logger.info(
            "[%s] val MAE%%=%.2f R2=%.3f | test MAE%%=%.2f R2=%.3f",
            name,
            entry["val"]["mae_pct"],
            entry["val"]["r2"],
            entry["test"]["mae_pct"],
            entry["test"]["r2"],
        )

        if name == "hist_gb":
            # impurity-style importances are not available; skip
            pass
        if name == "random_forest":
            imp = dict(zip(cols, model.feature_importances_.tolist()))
            results["rf_feature_importance"] = {
                k: round(v, 4) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])
            }

    if not args.skip_tiny_soh:
        logger.info("Training TinySOH (CNN + MLP)...")
        results["models"]["tiny_soh"] = _train_tiny_soh(
            device=args.device,
            epochs=args.tiny_epochs,
            patience=args.tiny_patience,
        )
        ts = results["models"]["tiny_soh"]
        logger.info(
            "[tiny_soh] val MAE%%=%.2f R2=%.3f | test MAE%%=%.2f R2=%.3f",
            ts["val"]["mae_pct"],
            ts["val"]["r2"],
            ts["test"]["mae_pct"],
            ts["test"]["r2"],
        )

    bafuse_path = ROOT / "results" / "summary_v2.json"
    if bafuse_path.exists():
        with open(bafuse_path, encoding="utf-8") as f:
            bafuse = json.load(f)
        results["bafuse_v2_staged"] = {
            "test": bafuse.get("overall_metrics"),
            "best_val_mae": bafuse.get("best_val_mae"),
            "per_battery": bafuse.get("per_battery"),
        }

    ranking = []
    for name, entry in results["models"].items():
        test = entry["test"]
        ranking.append((name, test["mae"], test["r2"]))
    ranking.sort(key=lambda t: t[1])
    results["test_mae_ranking"] = [
        {"model": n, "mae_pct": round(mae * 100, 3), "r2": round(r2, 4)}
        for n, mae, r2 in ranking
    ]
    results["recommended_baseline"] = ranking[0][0]
    logger.info("Recommended baseline (lowest test MAE): %s", ranking[0][0])

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "baseline_compare.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip_tiny_soh", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tiny_epochs", type=int, default=40)
    p.add_argument("--tiny_patience", type=int, default=8)
    sys.exit(main(p.parse_args()))
