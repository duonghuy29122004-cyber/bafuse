"""
BaFuse Architecture Benchmark v1 — Phase 1.

Compares 9 SOH-only architectures on the existing NASA PCoE split.
- Same 20/7/7 battery split for ALL models.
- Same preprocessing / normalisation statistics.
- MSE-only loss (no monotonicity / physics regulariser).
- Seed 42 for Phase 1.  3-seed runs triggered by --seeds 42 123 2026.

Usage:
    # Quick check (5 epochs):
    python scripts/architecture_benchmark.py --epochs 5 --dry_run

    # Phase 1  (seed=42):
    python scripts/architecture_benchmark.py --epochs 100 --seeds 42

    # 3-seed for top models only:
    python scripts/architecture_benchmark.py --epochs 100 --seeds 42 123 2026
                                             --models C D F G H

    # Phase 2 duration experiment on top-3:
    python scripts/architecture_benchmark.py --epochs 100 --seeds 42 --phase2

    # Phase 3 modality ablation on top-3:
    python scripts/architecture_benchmark.py --epochs 100 --seeds 42 --phase3
"""

import argparse, csv, json, logging, sys, time, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

OUT_DIR = ROOT / "results" / "architecture_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "predictions").mkdir(exist_ok=True)
(OUT_DIR / "plots").mkdir(exist_ok=True)
ERRORS_LOG = OUT_DIR / "errors.log"

# ─── Globals patched by CLI ──────────────────────────────────────────────────
LR         = 1e-3
WD         = 1e-4
EPOCHS     = 100
PATIENCE   = 15
BATCH_SIZE = 32
DEVICE     = "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(with_duration: bool = False):
    """Load cached NASA splits, optionally add duration_s feature."""
    proc = ROOT / "data" / "processed"
    train_df = pd.read_pickle(str(proc / "train.pkl"))
    val_df   = pd.read_pickle(str(proc / "val.pkl"))
    test_df  = pd.read_pickle(str(proc / "test.pkl"))
    disc_df  = pd.read_pickle(str(proc / "discharge_raw.pkl"))

    if with_duration and "duration_s" not in train_df.columns:
        # compute per-cycle duration from discharge_raw
        dur = (disc_df.groupby(["battery_id", "cycle_idx"])["time_s"]
               .agg(lambda t: t.max() - t.min())
               .reset_index()
               .rename(columns={"cycle_idx": "discharge_cycle", "time_s": "duration_s"}))
        train_df = train_df.merge(dur, on=["battery_id","discharge_cycle"], how="left")
        val_df   = val_df.merge(dur,   on=["battery_id","discharge_cycle"], how="left")
        test_df  = test_df.merge(dur,  on=["battery_id","discharge_cycle"], how="left")
        logger.info("Added duration_s to splits")

    return train_df, val_df, test_df, disc_df


def make_loaders(train_df, val_df, test_df, disc_df):
    from src.data.dataset import create_dataloaders
    return create_dataloaders(
        train_df, val_df, test_df,
        discharge_data_df=disc_df,
        batch_size=BATCH_SIZE, num_workers=0,
        save_stats_path=str(ROOT / "data" / "processed" / "nasa_train_stats.json"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    """Return MAE%, RMSE%, R² from [0,1]-scale arrays."""
    p_pct = pred * 100.0; t_pct = tgt * 100.0
    mae   = float(np.mean(np.abs(p_pct - t_pct)))
    rmse  = float(np.sqrt(np.mean((p_pct - t_pct)**2)))
    ss_r  = np.sum((tgt - pred)**2)
    ss_t  = np.sum((tgt - tgt.mean())**2)
    r2    = float(1 - ss_r / (ss_t + 1e-8))
    return dict(mae=mae, rmse=rmse, r2=r2, n=len(pred))


def per_battery_metrics(pred: np.ndarray, tgt: np.ndarray,
                        bids: np.ndarray) -> dict:
    out = {}
    for bid in np.unique(bids):
        m = bids == bid
        sub_pred = pred[m]; sub_tgt = tgt[m]
        ss_t = np.sum((sub_tgt - sub_tgt.mean())**2)
        met  = metrics(sub_pred, sub_tgt)
        met["r2_valid"] = bool(ss_t > 1e-4 and len(sub_tgt) >= 5)
        if not met["r2_valid"]:
            met["r2"] = None
        out[str(bid)] = met
    return out


def soh_bin_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    bins = [(0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]
    out  = {}
    for lo, hi in bins:
        m = (tgt >= lo) & (tgt < hi)
        if m.sum() == 0:
            continue
        label = f"{int(lo*100)}-{int(hi*100)}%"
        out[label] = metrics(pred[m], tgt[m])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader, val_loader,
    epochs: int, patience: int,
    device: torch.device,
) -> Tuple[dict, dict]:
    """
    Train with MSE loss only (no regularisers per benchmark spec).
    Returns (history_dict, best_val_metrics).
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    warmup = min(5, epochs // 5)
    def _lr(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(epochs - warmup, 1)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)

    best_mae = float("inf")
    best_state = None
    p_ctr = 0
    hist = {"train_loss": [], "val_mae": [], "val_r2": []}

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            out  = model(batch["discharge"], batch["eis"], batch["physics"])
            loss = criterion(out["soh_pred"].view(-1), batch["soh_label"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
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

        vp_np = np.array(vp); vt_np = np.array(vt)
        vm    = metrics(vp_np, vt_np)
        hist["train_loss"].append(ep_loss / max(len(train_loader), 1))
        hist["val_mae"].append(vm["mae"])
        hist["val_r2"].append(vm["r2"])

        if vm["mae"] < best_mae - 1e-3:
            best_mae = vm["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            p_ctr = 0; best_ep = ep
        else:
            p_ctr += 1
        if p_ctr >= patience:
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    hist["best_ep"] = best_ep if "best_ep" in dir() else ep
    return hist, {"val_mae": best_mae}


def evaluate_model(
    model: nn.Module, loader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, tgts, bids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch["discharge"], batch["eis"], batch["physics"])
            preds.extend(out["soh_pred"].view(-1).cpu().tolist())
            tgts.extend(batch["soh_label"].view(-1).cpu().tolist())
            b = batch["battery_id"]
            bids.extend(b if isinstance(b, list) else [b]*len(preds))
    return np.array(preds), np.array(tgts), np.array(bids)


# ═══════════════════════════════════════════════════════════════════════════════
# Single experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    model_name: str,
    model: nn.Module,
    train_loader, val_loader, test_loader,
    seed: int,
    tag: str = "",
) -> Optional[dict]:
    """Train and evaluate one model. Returns result dict or None on failure."""
    exp_id = f"{model_name}_{tag}_s{seed}" if tag else f"{model_name}_s{seed}"
    logger.info(f"  [{exp_id}] params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    device = torch.device(DEVICE)
    model.to(device)
    set_seed(seed)

    try:
        t0 = time.time()
        hist, _ = train_model(model, train_loader, val_loader,
                               EPOCHS, PATIENCE, device)
        elapsed = time.time() - t0

        # Validate
        vp, vt, vb = evaluate_model(model, val_loader, device)
        val_m = metrics(vp, vt)

        # Test
        tp, tt, tb = evaluate_model(model, test_loader, device)
        test_m = metrics(tp, tt)

        # Per-battery test
        per_bat = per_battery_metrics(tp, tt, tb)

        # SOH bins test
        soh_bins = soh_bin_metrics(tp, tt)

        # 10 random sample predictions
        rng = np.random.default_rng(0)
        idx = rng.choice(len(tp), min(10, len(tp)), replace=False)

        result = {
            "model":       model_name,
            "tag":         tag,
            "seed":        seed,
            "exp_id":      exp_id,
            "n_params":    sum(p.numel() for p in model.parameters() if p.requires_grad),
            "n_train":     len(vt)*0 or sum(len(next(iter(train_loader))["soh_label"]) for _ in [1]),
            "best_ep":     hist["best_ep"],
            "elapsed_s":   round(elapsed, 1),
            "val_mae":     round(val_m["mae"],  3),
            "val_rmse":    round(val_m["rmse"], 3),
            "val_r2":      round(val_m["r2"],   4),
            "test_mae":    round(test_m["mae"],  3),
            "test_rmse":   round(test_m["rmse"], 3),
            "test_r2":     round(test_m["r2"],   4),
            "per_battery": per_bat,
            "soh_bins":    soh_bins,
            "history":     {k: [round(v,5) for v in vals] for k, vals in hist.items()
                            if isinstance(vals, list)},
            "sample_preds": [
                {"idx": int(i), "pred": round(float(tp[i]),4), "tgt": round(float(tt[i]),4),
                 "err": round(abs(float(tp[i])-float(tt[i]))*100,2), "bid": str(tb[i])}
                for i in idx
            ],
        }

        # Save predictions
        np.save(str(OUT_DIR / "predictions" / f"{exp_id}_pred.npy"), tp)
        np.save(str(OUT_DIR / "predictions" / f"{exp_id}_tgt.npy"),  tt)
        np.save(str(OUT_DIR / "predictions" / f"{exp_id}_bids.npy"), tb)

        logger.info(
            f"  [{exp_id}] DONE  val_MAE={val_m['mae']:.2f}%  "
            f"test_MAE={test_m['mae']:.2f}%  R2={test_m['r2']:.4f}  "
            f"ep={hist['best_ep']}  {elapsed:.0f}s"
        )
        return result

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"  [{exp_id}] FAILED: {e}")
        with open(ERRORS_LOG, "a") as f:
            f.write(f"\n{'='*60}\n{exp_id}\n{tb_str}\n")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def make_plots(summary_df: pd.DataFrame, all_results: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["figure.dpi"] = 130

        df = summary_df.sort_values("test_mae")
        models = df["model"].tolist()
        maes   = df["test_mae"].tolist()
        r2s    = df["test_r2"].tolist()
        params = df["n_params"].tolist()

        # 1. test_mae_vs_model
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["#2196F3" if m == "H" else "#FF9800" for m in models]
        ax.bar(models, maes, color=colors, alpha=0.85)
        ax.set_ylabel("Test MAE (%)"); ax.set_title("Test MAE by Architecture")
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(ax.patches, maes):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(str(OUT_DIR/"plots"/"test_mae_vs_model.png"))
        plt.close()

        # 2. test_r2_vs_model
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(models, r2s, color=colors, alpha=0.85)
        ax.set_ylabel("Test R²"); ax.set_title("Test R² by Architecture")
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(0, color="red", lw=0.8, linestyle="--")
        plt.tight_layout()
        plt.savefig(str(OUT_DIR/"plots"/"test_r2_vs_model.png"))
        plt.close()

        # 3. params_vs_test_mae  (scatter)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(params, maes, s=80, alpha=0.8)
        for m, p, v in zip(models, params, maes):
            ax.annotate(m, (p, v), textcoords="offset points", xytext=(6, 2), fontsize=8)
        ax.set_xlabel("Parameters"); ax.set_ylabel("Test MAE (%)")
        ax.set_title("Parameter efficiency"); ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(str(OUT_DIR/"plots"/"params_vs_test_mae.png"))
        plt.close()

        # 4. per_battery_mae  (best model)
        best = summary_df.loc[summary_df["test_mae"].idxmin(), "model"]
        best_r = next((r for r in all_results if r["model"] == best), None)
        if best_r:
            pb = best_r["per_battery"]
            bids = list(pb.keys()); b_maes = [pb[b]["mae"] for b in bids]
            fig, ax = plt.subplots(figsize=(max(8, len(bids)*0.7), 4))
            ax.bar(bids, b_maes, color="#4CAF50", alpha=0.85)
            ax.set_ylabel("MAE (%)"); ax.set_title(f"Per-battery MAE — {best}")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            plt.savefig(str(OUT_DIR/"plots"/"per_battery_mae.png"))
            plt.close()

        # 5. soh_bin_mae (best model)
        if best_r:
            bins = best_r["soh_bins"]
            labels = list(bins.keys()); bin_maes = [bins[l]["mae"] for l in labels]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, bin_maes, color="#9C27B0", alpha=0.85)
            ax.set_ylabel("MAE (%)"); ax.set_title(f"SOH Bin MAE — {best}")
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            plt.savefig(str(OUT_DIR/"plots"/"soh_bin_mae.png"))
            plt.close()

        # 6. prediction_vs_true (best model)
        pred_p = OUT_DIR/"predictions"/f"{best}_s42_pred.npy"
        tgt_p  = OUT_DIR/"predictions"/f"{best}_s42_tgt.npy"
        if pred_p.exists() and tgt_p.exists():
            preds = np.load(str(pred_p)) * 100
            tgts  = np.load(str(tgt_p))  * 100
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(tgts, preds, s=15, alpha=0.6)
            mn, mx = min(tgts.min(), preds.min()), max(tgts.max(), preds.max())
            ax.plot([mn,mx],[mn,mx],"r--",lw=1)
            ax.set_xlabel("True SOH (%)"); ax.set_ylabel("Pred SOH (%)")
            ax.set_title(f"Prediction vs True — {best}")
            plt.tight_layout()
            plt.savefig(str(OUT_DIR/"plots"/"prediction_vs_true_best.png"))
            plt.close()

        logger.info("Plots saved.")
    except ImportError:
        logger.warning("matplotlib unavailable — plots skipped.")
    except Exception as e:
        logger.warning(f"Plot generation error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark phase helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(all_results: list, prefix: str = ""):
    if not all_results:
        return

    # Flat summary
    flat = []
    for r in all_results:
        if r is None:
            continue
        flat.append({
            "model": r["model"], "tag": r.get("tag",""), "seed": r["seed"],
            "n_params": r["n_params"],
            "val_mae": r["val_mae"], "val_rmse": r["val_rmse"], "val_r2": r["val_r2"],
            "test_mae": r["test_mae"], "test_rmse": r["test_rmse"], "test_r2": r["test_r2"],
            "best_ep": r["best_ep"], "elapsed_s": r["elapsed_s"],
        })

    df = pd.DataFrame(flat)
    csv_p  = OUT_DIR / f"{prefix}benchmark_summary.csv" if prefix else OUT_DIR/"benchmark_summary.csv"
    json_p = csv_p.with_suffix(".json")
    df.to_csv(str(csv_p), index=False)

    with open(str(json_p), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Per-battery CSV
    per_bat_rows = []
    for r in all_results:
        if r is None: continue
        for bid, m in r["per_battery"].items():
            per_bat_rows.append({
                "model": r["model"], "seed": r["seed"], "battery": bid,
                "mae": m["mae"], "rmse": m["rmse"],
                "r2": m["r2"] if m.get("r2_valid") else "N/A",
                "n": m["n"],
            })
    pb_path = OUT_DIR / f"{prefix}per_battery.csv" if prefix else OUT_DIR/"per_battery.csv"
    pd.DataFrame(per_bat_rows).to_csv(str(pb_path), index=False)

    # SOH bins CSV
    bin_rows = []
    for r in all_results:
        if r is None: continue
        for label, m in r["soh_bins"].items():
            bin_rows.append({
                "model": r["model"], "seed": r["seed"], "soh_bin": label,
                "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"], "n": m["n"],
            })
    sb_path = OUT_DIR / f"{prefix}soh_bins.csv" if prefix else OUT_DIR/"soh_bins.csv"
    pd.DataFrame(bin_rows).to_csv(str(sb_path), index=False)

    logger.info(f"Results saved -> {OUT_DIR}")
    return df


def print_table(df: pd.DataFrame, mean_baseline_mae: float):
    SEP = "=" * 90
    print(f"\n{SEP}")
    print("ARCHITECTURE BENCHMARK — FINAL TABLE  (seed=42 or mean over seeds)")
    print(SEP)
    print(f"  {'MODEL':6s}  {'PARAMS':>9s}  {'VAL_MAE':>8s}  {'TEST_MAE':>8s}  {'TEST_RMSE':>9s}  {'TEST_R2':>7s}  {'TIME(s)':>7s}")
    print(f"  {'-'*82}")

    # mean baseline row
    print(f"  {'Mean':6s}  {'0':>9s}  {'  n/a':>8s}  {mean_baseline_mae:>7.2f}%  {'  n/a':>9s}  {'  n/a':>7s}  {'  0':>7s}")

    # sort by test_mae
    gb = df.groupby("model").agg(
        n_params=("n_params","first"),
        val_mae=("val_mae","mean"), val_rmse=("val_rmse","mean"), val_r2=("val_r2","mean"),
        test_mae=("test_mae","mean"), test_rmse=("test_rmse","mean"), test_r2=("test_r2","mean"),
        elapsed_s=("elapsed_s","mean"),
    ).reset_index().sort_values("test_mae")

    for _, row in gb.iterrows():
        delta = mean_baseline_mae - row["test_mae"]
        print(
            f"  {row['model']:6s}  {row['n_params']:>9,}  "
            f"{row['val_mae']:>7.2f}%  {row['test_mae']:>7.2f}%  "
            f"{row['test_rmse']:>8.2f}%  {row['test_r2']:>7.4f}  {row['elapsed_s']:>7.0f}  "
            f"  [vs baseline: {delta:+.2f}pp]"
        )
    print(SEP)

    # Top 3 by MAE
    top3_mae = gb.nsmallest(3, "test_mae")["model"].tolist()
    top3_r2  = gb.nlargest(3, "test_r2")["model"].tolist()
    print(f"\n  Top-3 by Test MAE  : {top3_mae}")
    print(f"  Top-3 by Test R2   : {top3_r2}")

    # Param efficiency
    gb["eff"] = gb["test_mae"] / (gb["n_params"] / 1000 + 1e-3)
    top3_eff = gb.nsmallest(3, "eff")["model"].tolist()
    print(f"  Most param-efficient (MAE/1k params): {top3_eff}")
    print(SEP)
    return top3_mae


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — duration feature experiment
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_duration(top3: list, seeds: list):
    logger.info("\n" + "="*65)
    logger.info("PHASE 2 — Duration feature experiment")
    logger.info("="*65)

    train_df, val_df, test_df, disc_df = load_data(with_duration=True)
    if "duration_s" not in train_df.columns:
        logger.warning("duration_s not computable from cached data — Phase 2 skipped.")
        return []

    # Normalise duration with train stats
    dur_mean = train_df["duration_s"].mean()
    dur_std  = train_df["duration_s"].std() + 1e-8
    for df in [train_df, val_df, test_df]:
        df["duration_norm"] = (df["duration_s"] - dur_mean) / dur_std

    train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, disc_df)

    # We need a custom dataset that includes duration_norm in physics
    # Strategy: append duration_norm to the physics vector (4D → 5D)
    from torch.utils.data import DataLoader

    class DurationDataset:
        """Wraps existing loader batches, appending duration_norm to physics."""
        pass

    # Simpler: monkey-patch by creating augmented dataset in a collate_fn wrapper
    def _add_duration_collate(orig_loader, dur_vals: np.ndarray):
        """Yields batches from orig_loader with duration appended to physics."""
        dur_t = torch.tensor(dur_vals, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
        idx   = 0
        for batch in orig_loader:
            bsize = batch["soh_label"].shape[0]
            dur_b = dur_t[idx:idx+bsize]
            batch = dict(batch)
            batch["physics"] = torch.cat([batch["physics"], dur_b], dim=-1)
            idx += bsize
            yield batch

    results = []
    for model_name in top3:
        for seed in seeds:
            set_seed(seed)
            # Build model with physics_dim=5 (original 4 + duration)
            from src.models.benchmark_models import get_model
            model = get_model(model_name, d_in=3, e_in=3, p_in=5)
            logger.info(f"  Phase2 [{model_name}+dur s{seed}] params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            try:
                r = run_experiment(
                    model_name, model,
                    train_loader, val_loader, test_loader,
                    seed=seed, tag="dur",
                )
                if r:
                    results.append(r)
            except Exception as e:
                logger.error(f"  Phase2 [{model_name}+dur] failed: {e}")
                with open(ERRORS_LOG, "a") as f:
                    f.write(f"\nPhase2 {model_name} dur s{seed}\n{traceback.format_exc()}\n")

    save_results(results, prefix="phase2_")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Modality ablation
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_COMBOS = [
    ("D_only",  True,  False, False),
    ("E_only",  False, True,  False),
    ("P_only",  False, False, True),
    ("D+E",     True,  True,  False),
    ("D+P",     True,  False, True),
    ("E+P",     False, True,  True),
    ("D+E+P",   True,  True,  True),
]


class _MaskedWrapper(nn.Module):
    """Wraps a model and zeros out inactive modalities."""
    def __init__(self, model, use_d, use_e, use_p):
        super().__init__()
        self.m = model
        self.use_d = use_d; self.use_e = use_e; self.use_p = use_p

    def forward(self, discharge, eis, physics, **kw):
        if not self.use_d:
            discharge = torch.zeros_like(discharge)
        if not self.use_e:
            eis = torch.zeros_like(eis)
        if not self.use_p:
            physics = torch.zeros_like(physics)
        return self.m(discharge, eis, physics)

    def parameters(self, recurse=True):
        return self.m.parameters(recurse)

    def state_dict(self, **kw):
        return self.m.state_dict(**kw)

    def load_state_dict(self, d, **kw):
        return self.m.load_state_dict(d, **kw)

    def train(self, mode=True):
        self.m.train(mode); return self

    def eval(self):
        self.m.eval(); return self

    def to(self, device):
        self.m.to(device); return self


def phase3_ablation(top3: list, seeds: list, train_loader, val_loader, test_loader):
    logger.info("\n" + "="*65)
    logger.info("PHASE 3 — Modality ablation on top-3 architectures")
    logger.info("="*65)

    from src.models.benchmark_models import get_model
    results = []

    for model_name in top3:
        for combo_name, use_d, use_e, use_p in ABLATION_COMBOS:
            for seed in seeds:
                base   = get_model(model_name, d_in=3, e_in=3, p_in=4)
                model  = _MaskedWrapper(base, use_d, use_e, use_p)
                tag    = combo_name
                r = run_experiment(model_name, model, train_loader, val_loader,
                                   test_loader, seed=seed, tag=tag)
                if r:
                    r["ablation"] = combo_name
                    results.append(r)

    save_results(results, prefix="phase3_")
    logger.info("Phase 3 complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    global LR, WD, EPOCHS, PATIENCE, BATCH_SIZE, DEVICE
    LR       = args.lr
    WD       = args.wd
    EPOCHS   = args.epochs
    PATIENCE = args.patience
    DEVICE   = args.device

    logger.info("="*65)
    logger.info("BaFuse Architecture Benchmark v1")
    logger.info("="*65)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df, disc_df = load_data()
    train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, disc_df)

    # Confirm split
    logger.info(f"TRAIN batteries : {sorted(train_df.battery_id.unique())}")
    logger.info(f"VAL   batteries : {sorted(val_df.battery_id.unique())}")
    logger.info(f"TEST  batteries : {sorted(test_df.battery_id.unique())}")

    # Sample shapes
    sample = next(iter(train_loader))
    d_in = sample["discharge"].shape[-1]   # 3
    e_in = sample["eis"].shape[-1]          # 3
    p_in = sample["physics"].shape[-1]      # 4

    # Mean baseline
    from src.data.dataset import get_nominal_capacity
    tr_soh = np.clip(train_df["capacity_ahr"].values /
                     train_df["battery_id"].map(get_nominal_capacity).values, 0, 1)
    te_soh = np.clip(test_df["capacity_ahr"].values /
                     test_df["battery_id"].map(get_nominal_capacity).values, 0, 1)
    mean_pred_test_mae = float(np.mean(np.abs(np.full(len(te_soh), tr_soh.mean()) - te_soh))) * 100
    logger.info(f"Mean predictor test MAE: {mean_pred_test_mae:.2f}%")

    # ── Save config ───────────────────────────────────────────────────────────
    cfg = {
        "train_batteries": sorted(train_df.battery_id.unique().tolist()),
        "val_batteries":   sorted(val_df.battery_id.unique().tolist()),
        "test_batteries":  sorted(test_df.battery_id.unique().tolist()),
        "seeds":           args.seeds,
        "lr": LR, "wd": WD, "epochs": EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH_SIZE, "loss": "MSE", "target": "[0,1]",
        "d_shape": list(sample["discharge"].shape[1:]),
        "e_dim": e_in, "p_dim": p_in,
        "mean_baseline_test_mae": round(mean_pred_test_mae, 3),
    }
    with open(OUT_DIR/"config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Model list ────────────────────────────────────────────────────────────
    from src.models.benchmark_models import get_model, all_model_param_counts
    from src.models.bafuse import BaFuse
    from src.models.bafuse_v2 import BaFuseV2

    # Print param table
    counts = all_model_param_counts(d_in, e_in, p_in)
    baf_params = sum(p.numel() for p in BaFuse(d_in, e_in, p_in, 64, 128).parameters())
    baf2_params= sum(p.numel() for p in BaFuseV2(d_in, e_in, p_in, 64, 128).parameters())
    logger.info("Parameter counts:")
    for n, c in sorted(counts.items()):
        logger.info(f"  Model {n}: {c:>10,}")
    logger.info(f"  BaFuse v1 (H): {baf_params:>10,}")
    logger.info(f"  BaFuseV2  (H2):{baf2_params:>10,}")

    # Which models to run
    model_names = args.models if args.models else ["A","B","C","D","E","E2","F","G","H"]

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    all_results = []
    success, failed = [], []

    for seed in args.seeds:
        logger.info(f"\n--- Seed {seed} ---")
        for name in model_names:
            if args.dry_run and name not in ["A", "C", "H"]:
                continue   # dry run: only 3 models

            # Build model
            if name == "H":
                model = BaFuse(d_in, e_in, p_in, 64, 128)
            elif name == "H2":
                model = BaFuseV2(d_in, e_in, p_in, 64, 128)
            else:
                model = get_model(name, d_in, e_in, p_in)

            set_seed(seed)
            r = run_experiment(name, model, train_loader, val_loader,
                               test_loader, seed=seed)
            if r:
                all_results.append(r)
                success.append(f"{name}_s{seed}")
            else:
                failed.append(f"{name}_s{seed}")

    # ── Save Phase 1 ──────────────────────────────────────────────────────────
    summary_df = save_results(all_results)
    if summary_df is not None and len(summary_df):
        top3 = print_table(summary_df, mean_pred_test_mae)
        make_plots(summary_df, all_results)
    else:
        top3 = ["C", "D", "G"]
        logger.warning("No results to summarise — using default top3 for phases 2/3.")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if args.phase2:
        p2 = phase2_duration(top3[:3], args.seeds[:1])
        all_results.extend(p2)

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if args.phase3:
        phase3_ablation(top3[:3], args.seeds[:1], train_loader, val_loader, test_loader)

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("\nBENCHMARK COMPLETE")
    logger.info(f"  SUCCESS ({len(success)}): {success}")
    if failed:
        logger.info(f"  FAILED  ({len(failed)}): {failed}")
        logger.info(f"  Error log: {ERRORS_LOG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BaFuse Architecture Benchmark v1")
    ap.add_argument("--epochs",   type=int,   default=100)
    ap.add_argument("--patience", type=int,   default=15)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--wd",       type=float, default=1e-4)
    ap.add_argument("--seeds",    type=int,   nargs="+", default=[42])
    ap.add_argument("--device",   default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--models",   nargs="+",  default=None,
                    help="Subset of models to run, e.g. --models A C H")
    ap.add_argument("--dry_run",  action="store_true",
                    help="Quick 5-epoch test with 3 models")
    ap.add_argument("--phase2",   action="store_true")
    ap.add_argument("--phase3",   action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        args.epochs = 5
    main(args)
