"""
Ablation Results Evaluator — loads saved experiment results and generates:

    1. Summary CSV  (ablation_results.csv)
    2. Bar charts   — MAE / RMSE / R² comparison
    3. Delta table  — performance change vs full model
    4. Contribution scores (ablation-based, ΔRMSE-normalised)

IMPORTANT:
  Contribution scores are defined as:
      contribution_pct(m) = 100 * ΔRMSE(m) / Σ ΔRMSE(all modality-removal exps)
  where ΔRMSE(m) = RMSE_without_m − RMSE_full.

  This is an *ablation-based* contribution measure, NOT a causal importance
  statement. See evaluation/ablation_metrics.py for full definition.

Usage:
    python experiments/evaluate_ablation.py \\
        --results_dir experiments/ablation \\
        --out_dir     experiments/ablation/summary
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.ablation_metrics import AblationAnalyzer, ABLATION_EXPERIMENTS
from evaluation.metrics import compute_metrics


def load_experiment_results(results_dir: Path) -> AblationAnalyzer:
    """
    Scan results_dir for per-experiment subdirectories and load their results.
    """
    analyzer = AblationAnalyzer()

    for exp in ABLATION_EXPERIMENTS:
        exp_id  = exp["id"]
        exp_dir = results_dir / exp_id

        pred_path = exp_dir / "predictions.npy"
        tgt_path  = exp_dir / "targets.npy"
        met_path  = exp_dir / "metrics.json"

        if not pred_path.exists() or not tgt_path.exists():
            print(f"  [SKIP] {exp_id}: no predictions found")
            continue

        preds   = np.load(str(pred_path))
        targets = np.load(str(tgt_path))
        extra   = {}
        if met_path.exists():
            with open(met_path) as f:
                saved = json.load(f)
            extra["elapsed_sec"] = saved.get("elapsed_sec", 0)
            extra["seed"]        = saved.get("seed", 42)

        analyzer.add_result(
            experiment_id=exp_id,
            predictions=preds,
            targets=targets,
            discharge=exp["discharge"],
            eis=exp["eis"],
            physics=exp["physics"],
            extra_info=extra,
        )
        print(f"  [OK]   {exp_id}: {len(preds)} samples")

    return analyzer


def generate_plots(df: pd.DataFrame, out_dir: Path):
    """Generate comparison bar charts for MAE, RMSE, R²."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("mae",  "MAE (%)",  "lower is better"),
        ("rmse", "RMSE (%)", "lower is better"),
        ("r2",   "R²",       "higher is better"),
    ]

    for metric_col, ylabel, note in metrics:
        if metric_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [
            "#2196F3" if row["experiment"] == "A1_full" else "#FF9800"
            for _, row in df.iterrows()
        ]
        bars = ax.bar(df["experiment"], df[metric_col], color=colors, alpha=0.85)
        ax.set_xlabel("Ablation Experiment")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Ablation Study — {ylabel} ({note})")
        ax.tick_params(axis="x", rotation=30)

        for bar, val in zip(bars, df[metric_col]):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

        from matplotlib.patches import Patch
        legend = [
            Patch(color="#2196F3", label="Full model (A1)"),
            Patch(color="#FF9800", label="Ablation variant"),
        ]
        ax.legend(handles=legend)
        plt.tight_layout()
        plt.savefig(str(out_dir / f"ablation_{metric_col}.png"), dpi=150)
        plt.close()
        print(f"  Plot saved: ablation_{metric_col}.png")


def generate_delta_table(df: pd.DataFrame, out_dir: Path):
    """Print and save a delta-metrics table."""
    cols = ["experiment", "discharge", "eis", "physics",
            "mae", "rmse", "r2", "delta_mae", "delta_rmse", "delta_r2"]
    avail = [c for c in cols if c in df.columns]
    print("\n── Delta Metrics vs Full Model ──────────────────────────────────")
    print(df[avail].to_string(index=False, float_format="{:.4f}".format))

    delta_df = df[avail].copy()
    delta_df.to_csv(str(out_dir / "delta_table.csv"), index=False)
    print(f"\n  Saved → {out_dir / 'delta_table.csv'}")


def main(args):
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ablation results from: {results_dir}")
    analyzer = load_experiment_results(results_dir)

    if not analyzer._results:
        print("No experiment results found. Run run_ablation.py first.")
        return 1

    df = analyzer.build_summary_dataframe()

    # Save summary CSV + JSON
    analyzer.save_csv(str(out_dir / "ablation_results.csv"))
    analyzer.save_json(str(out_dir / "ablation_results.json"))

    # Print table
    analyzer.print_table()

    # Generate plots
    generate_plots(df, out_dir / "plots")

    # Generate delta table
    generate_delta_table(df, out_dir)

    # Print contribution note
    print("\n── Ablation-based Contribution Definition ───────────────────────")
    print("  contribution_pct(m) = 100 * ΔRMSE(m) / Σ ΔRMSE(all)")
    print("  ΔRMSE(m) = RMSE_without_m − RMSE_full")
    print("  NOTE: This is NOT a causal importance score.")
    print("  A modality removal may trigger compensation by other modalities.")
    print()

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate BaFuse v2 ablation results")
    p.add_argument("--results_dir", default="experiments/ablation")
    p.add_argument("--out_dir",     default="experiments/ablation/summary")
    sys.exit(main(p.parse_args()))
