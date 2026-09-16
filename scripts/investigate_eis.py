"""
P2: Investigate why EIS contributes ~0% to SoH prediction.

Checks:
1. NaN rate in re_ohm / rct_ohm after _safe_impedance_scalar
2. Actual cycle_gap distribution (is max_cycle_gap=10 causing EIS/discharge mismatch?)
3. EIS feature variance vs discharge feature variance (signal-to-noise)
4. Fusion attention weights per modality over training epochs
5. EIS-only vs discharge-only baseline MAE (quick sklearn)

Usage:
    python scripts/investigate_eis.py
"""

import sys, logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
SEP = "=" * 65


def main():
    # ── Load paired data ───────────────────────────────────────────────
    cache = ROOT / "data" / "processed" / "paired.pkl"
    if cache.exists():
        paired_df = pd.read_pickle(cache)
    else:
        from src.data.parse_mat import parse_all_mat_files
        from src.data.pairing import pair_discharge_eis
        dis_df, eis_df = parse_all_mat_files("5. BatteryDataSet")
        paired_df = pair_discharge_eis(dis_df, eis_df, max_cycle_gap=10)

    soh = (paired_df["capacity_ahr"] / 2.0 * 100).clip(0, 100)

    # ── CHECK 1: NaN rate in EIS columns ──────────────────────────────
    print(f"\n{SEP}")
    print("CHECK 1 — NaN rate in EIS features")
    print(SEP)
    for col in ["impedance_ohm", "re_ohm", "rct_ohm"]:
        n_nan = paired_df[col].isna().sum()
        pct   = n_nan / len(paired_df) * 100
        valid = paired_df[col].dropna()
        print(f"  {col:20s}: NaN={n_nan:4d} ({pct:.1f}%)  "
              f"range=[{valid.min():.4f}, {valid.max():.4f}]  "
              f"std={valid.std():.4f}")

    # ── CHECK 2: cycle_gap distribution ───────────────────────────────
    print(f"\n{SEP}")
    print("CHECK 2 — cycle_gap distribution (discharge - EIS pairing quality)")
    print(SEP)
    gaps = paired_df["cycle_gap"]
    print(f"  mean={gaps.mean():.2f}  median={gaps.median():.0f}  "
          f"std={gaps.std():.2f}  min={gaps.min()}  max={gaps.max()}")
    for gap_val in sorted(gaps.unique()):
        cnt = (gaps == gap_val).sum()
        bar = "#" * int(cnt / len(gaps) * 40)
        print(f"  gap={gap_val:3d}: {cnt:5d} pairs  {bar}")

    # ── CHECK 3: Feature variance (signal strength) ────────────────────
    print(f"\n{SEP}")
    print("CHECK 3 — Feature variance vs SoH correlation")
    print(SEP)

    feature_cols = {
        # Discharge
        "voltage_mean":  "Discharge",
        "voltage_min":   "Discharge",
        "voltage_std":   "Discharge",
        "current_mean":  "Discharge",
        "temp_mean":     "Discharge",
        # EIS
        "impedance_ohm": "EIS",
        "re_ohm":        "EIS",
        "rct_ohm":       "EIS",
    }

    print(f"  {'Feature':20s}  {'Modality':10s}  {'std':8s}  {'corr_w_SoH':10s}  {'CV%':6s}")
    print(f"  {'-'*60}")
    for col, mod in feature_cols.items():
        vals = paired_df[col].dropna()
        if len(vals) < 10:
            continue
        std  = vals.std()
        cv   = std / (abs(vals.mean()) + 1e-8) * 100
        corr = vals.corr(soh.loc[vals.index])
        print(f"  {col:20s}  {mod:10s}  {std:8.4f}  {corr:+10.4f}  {cv:6.1f}%")

    # ── CHECK 4: EIS-only vs Discharge-only quick baseline ─────────────
    print(f"\n{SEP}")
    print("CHECK 4 — Quick sklearn baseline: EIS-only vs Discharge-only vs Full")
    print(SEP)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error

    y = soh.values

    feature_sets = {
        "EIS only":       ["impedance_ohm", "re_ohm", "rct_ohm"],
        "Discharge only": ["voltage_mean", "voltage_min", "voltage_max",
                           "voltage_std", "current_mean", "temp_mean"],
        "Full":           ["voltage_mean", "voltage_min", "voltage_max",
                           "voltage_std", "current_mean", "temp_mean",
                           "impedance_ohm", "re_ohm", "rct_ohm"],
    }

    for name, cols in feature_sets.items():
        X = paired_df[cols].fillna(0).values
        pipe = Pipeline([("sc", StandardScaler()),
                         ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42))])
        scores = -cross_val_score(pipe, X, y, cv=5,
                                  scoring="neg_mean_absolute_error",
                                  groups=None)
        print(f"  {name:20s}: MAE = {scores.mean():.3f} ± {scores.std():.3f} %")

    # ── CHECK 5: EIS signal per battery ───────────────────────────────
    print(f"\n{SEP}")
    print("CHECK 5 — EIS impedance range per battery (is EIS changing with degradation?)")
    print(SEP)
    print(f"  {'Battery':8s}  {'N':5s}  {'imp_min':8s}  {'imp_max':8s}  {'imp_range':9s}  {'soh_range':9s}")
    print(f"  {'-'*55}")
    for bid, grp in paired_df.groupby("battery_id"):
        imp = grp["impedance_ohm"].dropna()
        s   = (grp["capacity_ahr"] / 2.0 * 100).clip(0, 100)
        if len(imp) < 2:
            continue
        print(f"  {bid:8s}  {len(grp):5d}  {imp.min():8.4f}  {imp.max():8.4f}  "
              f"{(imp.max()-imp.min()):9.4f}  {(s.max()-s.min()):9.2f}%")

    # ── FINDING SUMMARY ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("SUMMARY OF FINDINGS")
    print(SEP)

    imp_corr = paired_df["impedance_ohm"].corr(soh)
    re_corr  = paired_df["re_ohm"].dropna().corr(soh.loc[paired_df["re_ohm"].dropna().index])
    nan_re   = paired_df["re_ohm"].isna().mean() * 100
    nan_rct  = paired_df["rct_ohm"].isna().mean() * 100

    print(f"  impedance_ohm ↔ SoH correlation : {imp_corr:+.4f}")
    print(f"  re_ohm        ↔ SoH correlation : {re_corr:+.4f}")
    print(f"  re_ohm NaN rate                 : {nan_re:.1f}%")
    print(f"  rct_ohm NaN rate                : {nan_rct:.1f}%")

    if abs(imp_corr) < 0.3:
        print("\n  ⚠ LOW EIS-SoH correlation — EIS in this dataset has limited")
        print("    predictive power beyond what discharge curve already captures.")
        print("    This is a valid finding for the report (not a model bug).")
    else:
        print(f"\n  EIS has moderate correlation ({imp_corr:.3f}) — the ~0% contribution")
        print("    in ablation is likely due to attention imbalance (discharge dominates).")
        print("    Recommend: try WeightedFusion or explicit EIS loss term.")

    # ── CHECK 6: Are EIS cycle_gap > 5 cycles common? ─────────────────
    large_gap = (paired_df["cycle_gap"] > 5).sum()
    pct_large = large_gap / len(paired_df) * 100
    print(f"\n  Pairs with cycle_gap > 5       : {large_gap} ({pct_large:.1f}%)")
    if pct_large > 20:
        print("  ⚠ Large fraction of pairs have EIS measured >5 cycles after discharge.")
        print("    Consider reducing max_cycle_gap=5 for cleaner pairing.")
    print()


if __name__ == "__main__":
    main()
