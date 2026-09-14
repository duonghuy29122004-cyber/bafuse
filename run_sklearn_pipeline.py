"""
BaFuse sklearn pipeline — no PyTorch required.

Sử dụng Random Forest và MLP (scikit-learn) để dự đoán SoH.
Bao gồm: parse → pair → split → feature engineering → train → evaluate → ablation.

Usage:
    python run_sklearn_pipeline.py
    python run_sklearn_pipeline.py --data_dir "5. BatteryDataSet"
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

def build_features(paired_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and label vector y from paired DataFrame.

    Features (per cycle):
      Discharge  : voltage_mean, voltage_min, voltage_max, voltage_std,
                   current_mean, current_std, temp_mean, temp_std
      EIS        : impedance_ohm
      Physics    : cycle_age_norm, capacity_fade, voltage_droop, impedance_rise
    Label: SoH = (capacity_ahr / 2.0) * 100
    """
    df = paired_df.copy()

    # ---- SoH label ----
    nominal = 2.0  # Ahr
    df["soh"] = (df["capacity_ahr"] / nominal * 100.0).clip(0, 100)

    # ---- Physics features ----
    imp_mean = df["impedance_ohm"].mean()
    imp_std  = df["impedance_ohm"].std() + 1e-8
    cap_mean = df["capacity_ahr"].mean()
    cap_std  = df["capacity_ahr"].std() + 1e-8

    df["cycle_age_norm"]  = df["discharge_cycle"] / 200.0
    df["capacity_fade"]   = (2.0 - df["capacity_ahr"] - cap_mean) / cap_std
    df["voltage_droop"]   = (df["voltage_min"] - 2.7) / 1.5
    df["impedance_rise"]  = (df["impedance_ohm"] - imp_mean) / imp_std

    feature_cols = [
        # Discharge
        "voltage_mean", "voltage_min", "voltage_max", "voltage_std",
        "current_mean", "current_std",
        "temp_mean", "temp_std",
        # EIS
        "impedance_ohm",
        # Physics
        "cycle_age_norm", "capacity_fade", "voltage_droop", "impedance_rise",
    ]
    # Keep only existing columns (handle missing gracefully)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y = df["soh"].values.astype(np.float32)
    return X, y, feature_cols


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "mape": float(mape)}


# ─────────────────────────────────────────────
# Ablation helpers
# ─────────────────────────────────────────────

MODALITY_COLS = {
    "discharge": ["voltage_mean", "voltage_min", "voltage_max", "voltage_std",
                  "current_mean", "current_std", "temp_mean", "temp_std"],
    "eis":       ["impedance_ohm"],
    "physics":   ["cycle_age_norm", "capacity_fade", "voltage_droop", "impedance_rise"],
}


def ablation_study(train_df, test_df, all_feature_cols):
    """Train RF with one modality zeroed-out at a time."""
    results = {}
    for ablate_mod, zero_cols in MODALITY_COLS.items():
        X_tr, y_tr, cols = build_features(train_df)
        X_te, y_te, _    = build_features(test_df)

        # Zero out columns belonging to the ablated modality
        for col in zero_cols:
            if col in cols:
                idx = cols.index(col)
                X_tr[:, idx] = 0.0
                X_te[:, idx] = 0.0

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ])
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        results[f"no_{ablate_mod}"] = compute_metrics(y_te, preds)
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    from src.data.parse_mat import parse_all_mat_files
    from src.data.pairing import pair_discharge_eis, validate_pairs
    from src.data.split import split_by_battery

    # ── 1. Parse ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Parse .mat files")
    logger.info("=" * 60)
    discharge_df, eis_df = parse_all_mat_files(args.data_dir)
    logger.info(f"  discharge records : {len(discharge_df):,}")
    logger.info(f"  EIS records       : {len(eis_df):,}")

    if discharge_df.empty:
        logger.error("No discharge data. Check --data_dir.")
        return 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    discharge_df.to_pickle(out / "discharge_raw.pkl")
    eis_df.to_pickle(out / "eis_raw.pkl")

    # ── 2. Pair ───────────────────────────────────────────────
    logger.info("\nSTEP 2 — Pair discharge ↔ EIS")
    paired_df = pair_discharge_eis(discharge_df, eis_df, max_cycle_gap=2)
    logger.info(f"  pairs created     : {len(paired_df):,}")

    if paired_df.empty:
        logger.error("Pairing produced no results.")
        return 1

    stats = validate_pairs(paired_df)
    logger.info(f"  batteries         : {stats.get('num_batteries', '?')}")
    logger.info(f"  capacity range    : {stats.get('capacity_range', '?')}")
    paired_df.to_pickle(out / "paired.pkl")

    # ── 3. Split ──────────────────────────────────────────────
    logger.info("\nSTEP 3 — Battery-level split (60/20/20)")
    train_df, val_df, test_df = split_by_battery(
        paired_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        random_state=42, stratify_by_soh=True,
    )
    logger.info(f"  train / val / test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    train_df.to_pickle(out / "train.pkl")
    val_df.to_pickle(out  / "val.pkl")
    test_df.to_pickle(out / "test.pkl")

    # ── 4. Features ───────────────────────────────────────────
    logger.info("\nSTEP 4 — Feature engineering")
    X_train, y_train, feat_cols = build_features(train_df)
    X_val,   y_val,   _         = build_features(val_df)
    X_test,  y_test,  _         = build_features(test_df)
    logger.info(f"  feature columns   : {feat_cols}")
    logger.info(f"  train shape       : {X_train.shape}")

    # ── 5. Train three models ─────────────────────────────────
    logger.info("\nSTEP 5 — Train models")

    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, max_depth=None,
                                            random_state=42, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                learning_rate=0.05, random_state=42)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(256, 128, 64),
                                   activation="relu", max_iter=500,
                                   learning_rate_init=0.001, random_state=42,
                                   early_stopping=True, validation_fraction=0.1)),
        ]),
    }

    # Combine train+val for final training
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    val_results = {}
    test_results = {}

    for name, pipe in models.items():
        # Fit on train only first → validate
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_m = compute_metrics(y_val, val_pred)
        val_results[name] = val_m
        logger.info(f"  [{name}] Val  — MAE={val_m['mae']:.3f}  RMSE={val_m['rmse']:.3f}  R²={val_m['r2']:.4f}")

        # Refit on train+val → test
        pipe.fit(X_trainval, y_trainval)
        test_pred = pipe.predict(X_test)
        test_m = compute_metrics(y_test, test_pred)
        test_results[name] = test_m
        logger.info(f"  [{name}] Test — MAE={test_m['mae']:.3f}  RMSE={test_m['rmse']:.3f}  R²={test_m['r2']:.4f}")

    # ── 6. Best model deep-dive ───────────────────────────────
    best_name = min(test_results, key=lambda k: test_results[k]["mae"])
    best_pipe  = models[best_name]
    best_pipe.fit(X_trainval, y_trainval)
    test_pred  = best_pipe.predict(X_test)
    logger.info(f"\n  Best model: {best_name}")

    # Per-battery breakdown
    logger.info("\n  Per-battery test MAE:")
    test_df2 = test_df.copy()
    test_df2["pred"] = test_pred
    test_df2["soh"]  = (test_df2["capacity_ahr"] / 2.0 * 100).clip(0, 100)
    per_bat = {}
    for bid, grp in test_df2.groupby("battery_id"):
        m = compute_metrics(grp["soh"].values, grp["pred"].values)
        per_bat[str(bid)] = m
        logger.info(f"    {bid}: MAE={m['mae']:.3f}%  R²={m['r2']:.4f}")

    # ── 7. Feature importance ─────────────────────────────────
    logger.info("\nSTEP 6 — Feature importance (Random Forest)")
    rf_pipe = models["RandomForest"]
    rf_pipe.fit(X_trainval, y_trainval)
    importances = rf_pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["importance"] * 40)
        logger.info(f"  {row['feature']:25s} {bar} {row['importance']:.4f}")

    # Modality-level importance
    mod_imp = {}
    for mod, cols in MODALITY_COLS.items():
        mod_imp[mod] = float(imp_df[imp_df["feature"].isin(cols)]["importance"].sum())
    total = sum(mod_imp.values()) + 1e-8
    logger.info("\n  Modality contributions (by feature importance):")
    for mod, imp in mod_imp.items():
        logger.info(f"    {mod:12s}: {imp/total*100:.1f}%")

    # ── 8. Ablation study ─────────────────────────────────────
    logger.info("\nSTEP 7 — Ablation study")
    ablation = ablation_study(pd.concat([train_df, val_df]), test_df, feat_cols)
    base_mae = test_results["RandomForest"]["mae"]
    for key, m in ablation.items():
        delta = m["mae"] - base_mae
        logger.info(f"  {key:20s}: MAE={m['mae']:.3f}%  (+{delta:+.3f} vs full)")

    # ── 9. Save results ───────────────────────────────────────
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    summary = {
        "best_model": best_name,
        "test_metrics": test_results,
        "val_metrics": val_results,
        "per_battery_test": per_bat,
        "feature_importance": imp_df.set_index("feature")["importance"].to_dict(),
        "modality_contributions_pct": {k: round(v/total*100, 2) for k, v in mod_imp.items()},
        "ablation": ablation,
    }
    out_json = results_dir / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n  Results saved → {out_json}")
    logger.info("\n" + "=" * 60)
    logger.info("✓  Pipeline complete!")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BaFuse sklearn pipeline (no PyTorch)")
    p.add_argument("--data_dir",   default="5. BatteryDataSet")
    p.add_argument("--output_dir", default="data/processed")
    args = p.parse_args()
    sys.exit(main(args))
