"""
BaFuse v2 — Visualization utilities.

Generates all plots required for the research paper and dashboard:

    1.  plot_soh_predictions()        — Predicted vs true SOH scatter
    2.  plot_soh_degradation_curve()  — SOH over aging cycles per battery
    3.  plot_training_history()       — Train/val loss curves
    4.  plot_ablation_comparison()    — MAE/RMSE/R² bar chart (A1–A7)
    5.  plot_metric_comparison()      — Grouped metric bars
    6.  plot_degradation_predictions()— LLI/LAM/CL pred vs target
    7.  plot_degradation_trends()     — Degradation mode vs aging cycle

Terminology note:
  LLI/LAM/CL labels are model-derived degradation-mode estimates from
  ECM fitting (Mendeley dataset). Plots are labelled accordingly.

All functions:
  - Accept numpy arrays (no torch dependency).
  - Return matplotlib Figure objects (caller decides whether to save/show).
  - Use Agg backend by default (safe for headless environments).
  - Save to disk when out_path is provided.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Safe import of matplotlib ─────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not installed — visualization disabled.")


def _require_mpl(fn):
    """Decorator: skip and return None if matplotlib unavailable."""
    def wrapper(*args, **kwargs):
        if not _HAS_MPL:
            logger.warning(f"{fn.__name__}: matplotlib not available, skipping.")
            return None
        return fn(*args, **kwargs)
    return wrapper


def _save(fig, out_path: Optional[str], dpi: int = 150):
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Plot saved → {out_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Predicted vs True SOH
# ═══════════════════════════════════════════════════════════════════════════════

@_require_mpl
def plot_soh_predictions(
    predictions:  np.ndarray,
    targets:      np.ndarray,
    battery_ids:  Optional[List[str]] = None,
    title:        str                 = "Predicted vs True SOH",
    out_path:     Optional[str]       = None,
) -> "plt.Figure":
    """
    Scatter plot of predicted vs true SOH.

    Args:
        predictions  : (N,) predicted SOH values.
        targets      : (N,) ground-truth SOH values.
        battery_ids  : Optional list of battery IDs for colour-coding.
        title        : Plot title.
        out_path     : If given, save to this path.
    """
    preds   = np.asarray(predictions).ravel()
    tgts    = np.asarray(targets).ravel()

    mae  = np.mean(np.abs(preds - tgts))
    rmse = np.sqrt(np.mean((preds - tgts) ** 2))
    ss_r = np.sum((tgts - preds) ** 2)
    ss_t = np.sum((tgts - np.mean(tgts)) ** 2)
    r2   = 1.0 - ss_r / (ss_t + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 6))

    if battery_ids is not None:
        import pandas as pd
        unique_bids = sorted(set(battery_ids))
        cmap = plt.cm.get_cmap("tab20", len(unique_bids))
        bid_to_idx = {b: i for i, b in enumerate(unique_bids)}
        colors = [cmap(bid_to_idx[b]) for b in battery_ids]
        sc = ax.scatter(tgts, preds, c=colors, s=18, alpha=0.7, edgecolors="none")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cmap(i), markersize=6, label=b)
            for i, b in enumerate(unique_bids[:12])   # cap legend at 12
        ]
        ax.legend(handles=handles, fontsize=6, loc="upper left",
                  title="Battery", title_fontsize=7)
    else:
        ax.scatter(tgts, preds, s=18, alpha=0.7, color="#2196F3", edgecolors="none")

    # Perfect-prediction line
    mn, mx = min(tgts.min(), preds.min()), max(tgts.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.2, label="Perfect prediction")

    ax.set_xlabel("True SOH (%)")
    ax.set_ylabel("Predicted SOH (%)")
    ax.set_title(f"{title}\nMAE={mae:.3f}%  RMSE={rmse:.3f}%  R²={r2:.4f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SOH Degradation Curve
# ═══════════════════════════════════════════════════════════════════════════════

@_require_mpl
def plot_soh_degradation_curve(
    predictions:  np.ndarray,
    targets:      np.ndarray,
    battery_ids:  List[str],
    cycle_indices: Optional[np.ndarray] = None,
    title:        str                   = "SOH Degradation Curve",
    max_batteries: int                  = 8,
    out_path:     Optional[str]         = None,
) -> "plt.Figure":
    """
    Per-battery SOH curve: predicted (dashed) vs true (solid) over cycles.
    """
    import pandas as pd

    preds  = np.asarray(predictions).ravel()
    tgts   = np.asarray(targets).ravel()
    bids   = np.asarray(battery_ids)
    cycles = np.asarray(cycle_indices) if cycle_indices is not None \
             else np.arange(len(preds))

    df = pd.DataFrame({"pred": preds, "target": tgts,
                       "battery_id": bids, "cycle": cycles})

    unique_bids = sorted(df["battery_id"].unique())[:max_batteries]
    ncols = min(4, len(unique_bids))
    nrows = (len(unique_bids) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    fig.suptitle(title, fontsize=11)

    for i, bid in enumerate(unique_bids):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        sub = df[df["battery_id"] == bid].sort_values("cycle")
        ax.plot(sub["cycle"], sub["target"], "b-",  lw=1.5, label="True SOH")
        ax.plot(sub["cycle"], sub["pred"],   "r--", lw=1.5, label="Predicted")
        ax.set_title(str(bid), fontsize=9)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH (%)")
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(len(unique_bids), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    return _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Training / Validation Loss
# ═══════════════════════════════════════════════════════════════════════════════

@_require_mpl
def plot_training_history(
    history:  Dict[str, List[float]],
    title:    str             = "Training History",
    out_path: Optional[str]   = None,
) -> "plt.Figure":
    """
    Plot training and validation loss + validation MAE/RMSE/R².

    Args:
        history: Dict from trainer.fit() with keys:
                 train_loss / train_soh_loss, val_loss / val_soh_loss,
                 val_mae, val_rmse, val_r2.
    """
    train_key = "train_soh_loss" if "train_soh_loss" in history else "train_loss"
    val_key   = "val_soh_loss"   if "val_soh_loss"   in history else "val_loss"

    train_loss = history.get(train_key, [])
    val_loss   = history.get(val_key,   [])
    val_mae    = history.get("val_mae",  [])
    val_rmse   = history.get("val_rmse", [])
    val_r2     = history.get("val_r2",   [])

    n_plots = 1 + bool(val_mae) + bool(val_r2)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    # ---- Loss subplot ------------------------------------------------
    ax = axes[ax_idx]
    epochs = range(1, len(train_loss) + 1)
    if train_loss:
        ax.plot(epochs, train_loss, label="Train loss")
    if val_loss:
        ax.plot(range(1, len(val_loss) + 1), val_loss,
                label="Val loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax_idx += 1

    # ---- MAE / RMSE subplot -----------------------------------------
    if val_mae and ax_idx < n_plots:
        ax = axes[ax_idx]
        ep = range(1, len(val_mae) + 1)
        ax.plot(ep, val_mae,  label="Val MAE")
        if val_rmse:
            ax.plot(range(1, len(val_rmse) + 1), val_rmse,
                    label="Val RMSE", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error (%)")
        ax.set_title("MAE / RMSE")
        ax.legend()
        ax_idx += 1

    # ---- R² subplot --------------------------------------------------
    if val_r2 and ax_idx < n_plots:
        ax = axes[ax_idx]
        ax.plot(range(1, len(val_r2) + 1), val_r2, color="green", label="Val R²")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R²")
        ax.set_title("R²")
        ax.legend()

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Ablation Comparison
# ═══════════════════════════════════════════════════════════════════════════════

@_require_mpl
def plot_ablation_comparison(
    ablation_df,                             # pd.DataFrame from AblationAnalyzer
    metric:   str           = "rmse",
    title:    str           = "Ablation Study",
    out_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Bar chart comparing one metric across all ablation experiments.

    Args:
        ablation_df : DataFrame from AblationAnalyzer.build_summary_dataframe().
        metric      : "mae", "rmse", or "r2".
        title       : Plot title.
    """
    if ablation_df is None or ablation_df.empty or metric not in ablation_df.columns:
        logger.warning(f"plot_ablation_comparison: no data for metric '{metric}'")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [
        "#2196F3" if row["experiment"] == "A1_full" else "#FF9800"
        for _, row in ablation_df.iterrows()
    ]
    bars = ax.bar(ablation_df["experiment"], ablation_df[metric],
                  color=colors, alpha=0.85)

    for bar, val in zip(bars, ablation_df[metric]):
        if not np.isnan(float(val)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(bar.get_height()) * 1.01,
                f"{float(val):.4f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Ablation Experiment")
    ylabel = {"mae": "MAE (%)", "rmse": "RMSE (%)", "r2": "R²"}.get(metric, metric)
    ax.set_ylabel(ylabel)
    note = "lower is better" if metric != "r2" else "higher is better"
    ax.set_title(f"{title} — {ylabel} ({note})")
    ax.tick_params(axis="x", rotation=30)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2196F3", label="Full model (A1)"),
        Patch(color="#FF9800", label="Ablation variant"),
    ])
    plt.tight_layout()
    return _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAE / RMSE / R² Grouped Comparison
# ═══════════════════════════════════════════════════════════════════════════════

@_require_mpl
def plot_metric_comparison(
    ablation_df,
    out_path: Optional[str] = None,
) -> "plt.Figure":
    """Three-panel grouped bar chart for MAE, RMSE, R²."""
    if ablation_df is None or ablation_df.empty:
        return None

    metrics = [("mae", "MAE (%)", False), ("rmse", "RMSE (%)", False), ("r2", "R²", True)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("BaFuse v2 Ablation — Metric Comparison", fontsize=11)

    for ax, (col, label, higher_better) in zip(axes, metrics):
        if col not in ablation_df.columns:
            ax.set_visible(False)
            continue
        colors = ["#2196F3" if e == "A1_full" else "#FF9800"
                  for e in ablation_df["experiment"]]
        ax.bar(ablation_df["experiment"], ablation_df[col], color=colors, alpha=0.85)
        ax.set_ylabel(label)
        ax.set_title(f"{label} ({'↑' if higher_better else '↓'} better)")
        ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    return _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LLI / LAM / CL Prediction vs Target
# ═══════════════════════════════════════════════════════════════════════════════

@_require_mpl
def plot_degradation_predictions(
    lli_pred:  np.ndarray,
    lli_target: np.ndarray,
    lam_pred:  np.ndarray,
    lam_target: np.ndarray,
    cl_pred:   np.ndarray,
    cl_target:  np.ndarray,
    title:     str           = "Model-Derived Degradation Mode Estimates",
    out_path:  Optional[str] = None,
) -> "plt.Figure":
    """
    Scatter plots: predicted vs target for LLI, LAM, CL.

    NOTE: Targets are model-derived from ECM fitting — not absolute ground truth.
    This is stated in the subtitle.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"{title}\n"
        "(Targets are model-derived degradation-mode estimates from ECM fitting, "
        "NOT physical ground truth)",
        fontsize=9,
    )

    pairs = [
        (axes[0], lli_pred, lli_target, "LLI (%)", "#E91E63"),
        (axes[1], lam_pred, lam_target, "LAM (%)", "#9C27B0"),
        (axes[2], cl_pred,  cl_target,  "CL (%)",  "#009688"),
    ]

    for ax, pred, target, label, color in pairs:
        p = np.asarray(pred).ravel()
        t = np.asarray(target).ravel()
        mae = np.mean(np.abs(p - t))
        ax.scatter(t, p, s=20, alpha=0.7, color=color, edgecolors="none")
        mn, mx = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.set_xlabel(f"Target {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}  MAE={mae:.3f}")

    plt.tight_layout()
    return _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Degradation Trends over Aging Cycle
# ═══════════════════════════════════════════════════════════════════════════════

@_require_mpl
def plot_degradation_trends(
    aging_cycles: np.ndarray,
    lli_values:   np.ndarray,
    lam_values:   np.ndarray,
    cl_values:    np.ndarray,
    cell_ids:     Optional[np.ndarray] = None,
    title:        str                  = "Model-Derived Degradation Mode Trends",
    out_path:     Optional[str]        = None,
) -> "plt.Figure":
    """
    Line plot of LLI/LAM/CL model-derived estimates over aging cycles.

    If cell_ids provided, plots one line per cell.
    """
    import pandas as pd

    cycles = np.asarray(aging_cycles).ravel()
    lli    = np.asarray(lli_values).ravel()
    lam    = np.asarray(lam_values).ravel()
    cl     = np.asarray(cl_values).ravel()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"{title}\n(Values are model-derived estimates from ECM fitting)",
        fontsize=9,
    )

    for ax, vals, label, color in [
        (axes[0], lli, "LLI (%)", "#E91E63"),
        (axes[1], lam, "LAM (%)", "#9C27B0"),
        (axes[2], cl,  "CL (%)",  "#009688"),
    ]:
        if cell_ids is not None:
            df = pd.DataFrame({"cycle": cycles, "val": vals, "cell_id": cell_ids})
            for cid, grp in df.groupby("cell_id"):
                grp = grp.sort_values("cycle")
                ax.plot(grp["cycle"], grp["val"], marker="o", ms=3,
                        label=f"Cell {cid}", alpha=0.8)
            ax.legend(fontsize=6)
        else:
            ax.plot(cycles, vals, "o-", color=color, ms=3, alpha=0.7)
        ax.set_xlabel("Aging Cycle")
        ax.set_ylabel(label)
        ax.set_title(label)

    plt.tight_layout()
    return _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: generate all plots from a result dict
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_plots(
    soh_preds:    np.ndarray,
    soh_targets:  np.ndarray,
    battery_ids:  List[str],
    history:      Dict,
    ablation_df,
    out_dir:      str,
    deg_results:  Optional[Dict] = None,
):
    """
    Generate all standard BaFuse v2 plots and save to out_dir.

    Args:
        soh_preds/targets : SOH predictions & targets.
        battery_ids       : Per-sample battery IDs.
        history           : Training history dict.
        ablation_df       : AblationAnalyzer.build_summary_dataframe().
        out_dir           : Directory for saved plots.
        deg_results       : Optional dict from MultiTaskTrainer.predict_deg().
    """
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)

    plot_soh_predictions(
        soh_preds, soh_targets, battery_ids,
        out_path=str(od / "soh_predictions.png"),
    )
    plot_soh_degradation_curve(
        soh_preds, soh_targets, battery_ids,
        out_path=str(od / "soh_degradation_curve.png"),
    )
    plot_training_history(
        history,
        out_path=str(od / "training_history.png"),
    )
    if ablation_df is not None and not ablation_df.empty:
        plot_ablation_comparison(
            ablation_df, metric="rmse",
            out_path=str(od / "ablation_rmse.png"),
        )
        plot_metric_comparison(
            ablation_df,
            out_path=str(od / "ablation_metric_comparison.png"),
        )
    if deg_results is not None:
        plot_degradation_predictions(
            deg_results["lli_pred"],  deg_results["lli_target"],
            deg_results["lam_pred"],  deg_results["lam_target"],
            deg_results["cl_pred"],   deg_results["cl_target"],
            out_path=str(od / "degradation_predictions.png"),
        )
        plot_degradation_trends(
            deg_results["aging_cycles"],
            deg_results["lli_pred"],
            deg_results["lam_pred"],
            deg_results["cl_pred"],
            cell_ids=np.array(deg_results.get("cell_ids", [])) or None,
            out_path=str(od / "degradation_trends.png"),
        )
    logger.info(f"All plots saved → {od}")
