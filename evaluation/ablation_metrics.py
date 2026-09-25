"""
Ablation-based contribution analysis for BaFuse v2.

Terminology note:
  "ablation-based contribution" ≠ "causal importance".
  The contribution scores reported here reflect performance *degradation*
  when a modality is removed from a trained model — they do NOT imply that
  the modality is definitively responsible for that share of the prediction.

  If a percentage is reported it is defined as:

      contribution_pct(m) = 100 * ΔRMSE(m) / Σ ΔRMSE(all)

  where ΔRMSE(m) = RMSE_without_m − RMSE_full  (higher = more contribution).

  This definition is documented here and must be cited whenever the percentage
  is used in any report.

Classes:
    AblationAnalyzer — loads a result CSV and computes delta metrics + scores.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import compute_metrics

# ── Ablation experiment IDs ────────────────────────────────────────────────────
ABLATION_EXPERIMENTS = [
    {"id": "A1_full",          "discharge": True,  "eis": True,  "physics": True},
    {"id": "A2_no_discharge",  "discharge": False, "eis": True,  "physics": True},
    {"id": "A3_no_eis",        "discharge": True,  "eis": False, "physics": True},
    {"id": "A4_no_physics",    "discharge": True,  "eis": True,  "physics": False},
    {"id": "A5_discharge_only","discharge": True,  "eis": False, "physics": False},
    {"id": "A6_eis_only",      "discharge": False, "eis": True,  "physics": False},
    {"id": "A7_physics_only",  "discharge": False, "eis": False, "physics": True},
]

ABLATION_CSV_COLUMNS = [
    "experiment", "discharge", "eis", "physics",
    "mae", "rmse", "r2",
    "delta_mae", "delta_rmse", "delta_r2",
    "contribution_pct_mae", "contribution_pct_rmse",
]


class AblationAnalyzer:
    """
    Compute ablation-based contribution analysis from per-experiment results.

    Usage:
        analyzer = AblationAnalyzer()
        analyzer.add_result("A1_full", preds_full, targets, discharge=True, eis=True, physics=True)
        analyzer.add_result("A2_no_discharge", preds_no_d, targets, ...)
        ...
        df = analyzer.build_summary_dataframe()
        analyzer.save_csv("experiments/ablation_results.csv")
        analyzer.save_json("experiments/ablation_results.json")
    """

    def __init__(self):
        self._results: List[Dict] = []

    # -----------------------------------------------------------------------

    def add_result(
        self,
        experiment_id: str,
        predictions:   np.ndarray,
        targets:       np.ndarray,
        discharge:     bool,
        eis:           bool,
        physics:       bool,
        extra_info:    Optional[Dict] = None,
    ):
        """
        Register predictions for one ablation configuration.

        Args:
            experiment_id : e.g. "A1_full", "A3_no_eis".
            predictions   : (N,) predicted SOH values.
            targets       : (N,) ground-truth SOH values.
            discharge/eis/physics : which modalities were active.
            extra_info    : optional dict (seed, epochs, runtime, etc.).
        """
        m = compute_metrics(predictions, targets)
        row = {
            "experiment": experiment_id,
            "discharge":  discharge,
            "eis":        eis,
            "physics":    physics,
            "mae":        m["mae"],
            "rmse":       m["rmse"],
            "r2":         m["r2"],
            "n_samples":  len(np.asarray(predictions).ravel()),
            **(extra_info or {}),
        }
        self._results.append(row)

    # -----------------------------------------------------------------------

    def build_summary_dataframe(self) -> pd.DataFrame:
        """
        Build the summary DataFrame with delta metrics and contribution scores.

        Delta metrics are relative to the FULL model (A1_full).
        Contribution % is defined as documented in this module's docstring.

        Returns:
            pd.DataFrame with columns matching ABLATION_CSV_COLUMNS.
        """
        if not self._results:
            return pd.DataFrame(columns=ABLATION_CSV_COLUMNS)

        df = pd.DataFrame(self._results)

        # Baseline (full model)
        full_rows = df[df["experiment"] == "A1_full"]
        if full_rows.empty:
            # Fallback: use row with best MAE as reference
            full_rows = df.loc[[df["mae"].idxmin()]]

        baseline_mae  = float(full_rows["mae"].iloc[0])
        baseline_rmse = float(full_rows["rmse"].iloc[0])
        baseline_r2   = float(full_rows["r2"].iloc[0])

        df["delta_mae"]  = df["mae"]  - baseline_mae
        df["delta_rmse"] = df["rmse"] - baseline_rmse
        df["delta_r2"]   = df["r2"]   - baseline_r2    # negative = worse R²

        # ── Contribution % (only for ablation experiments, not full model) ──
        # ΔRMSE for modality-removal ablations: A2, A3, A4
        removal_exp = {
            "discharge": "A2_no_discharge",
            "eis":       "A3_no_eis",
            "physics":   "A4_no_physics",
        }
        delta_per_modality: Dict[str, float] = {}
        for mod, exp_id in removal_exp.items():
            rows = df[df["experiment"] == exp_id]
            if not rows.empty:
                delta_per_modality[mod] = max(
                    float(rows["rmse"].iloc[0]) - baseline_rmse, 0.0
                )
            else:
                delta_per_modality[mod] = 0.0

        total_delta = sum(delta_per_modality.values()) + 1e-10

        def _contrib_pct_rmse(row) -> float:
            """Contribution % for this experiment's removal experiment."""
            for mod, exp_id in removal_exp.items():
                if row["experiment"] == exp_id:
                    return 100.0 * delta_per_modality[mod] / total_delta
            return np.nan

        df["contribution_pct_rmse"] = df.apply(_contrib_pct_rmse, axis=1)

        # Same for MAE
        delta_mae_per_mod: Dict[str, float] = {}
        for mod, exp_id in removal_exp.items():
            rows = df[df["experiment"] == exp_id]
            if not rows.empty:
                delta_mae_per_mod[mod] = max(
                    float(rows["mae"].iloc[0]) - baseline_mae, 0.0
                )
            else:
                delta_mae_per_mod[mod] = 0.0

        total_delta_mae = sum(delta_mae_per_mod.values()) + 1e-10

        def _contrib_pct_mae(row) -> float:
            for mod, exp_id in removal_exp.items():
                if row["experiment"] == exp_id:
                    return 100.0 * delta_mae_per_mod[mod] / total_delta_mae
            return np.nan

        df["contribution_pct_mae"] = df.apply(_contrib_pct_mae, axis=1)

        # Reorder columns
        ordered_cols = ABLATION_CSV_COLUMNS + [
            c for c in df.columns if c not in ABLATION_CSV_COLUMNS
        ]
        df = df[[c for c in ordered_cols if c in df.columns]]

        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------

    def save_csv(self, path: str):
        """Save summary DataFrame to CSV."""
        df = self.build_summary_dataframe()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Ablation results saved → {path}")

    def save_json(self, path: str):
        """Save summary as JSON (for downstream use)."""
        df = self.build_summary_dataframe()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        records = df.to_dict(orient="records")
        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"Ablation JSON saved → {path}")

    # -----------------------------------------------------------------------

    def print_table(self):
        """Print a formatted summary table to stdout."""
        df = self.build_summary_dataframe()
        if df.empty:
            print("No ablation results to display.")
            return

        cols = ["experiment", "discharge", "eis", "physics",
                "mae", "rmse", "r2", "delta_mae", "delta_rmse"]
        available = [c for c in cols if c in df.columns]
        print("\n── Ablation Results ─────────────────────────────────────────────")
        print(df[available].to_string(index=False, float_format="{:.4f}".format))
        print()

        # Contribution note
        print("── Ablation-based Contribution Scores (ΔRMSE-normalised) ────────")
        print("NOTE: These reflect performance change when a modality is removed.")
        print("They are NOT causal importance scores.")
        print()
        contrib_df = df[df["contribution_pct_rmse"].notna()][
            ["experiment", "contribution_pct_rmse", "contribution_pct_mae"]
        ]
        if not contrib_df.empty:
            print(contrib_df.to_string(index=False, float_format="{:.1f}".format))
        print()

    # -----------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str) -> "AblationAnalyzer":
        """Load a previously saved ablation CSV (for re-analysis)."""
        analyzer = cls()
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            analyzer._results.append(row.to_dict())
        return analyzer
