"""Inspect Mendeley data: columns, cell distribution, label ranges, EIS features."""
import sys, pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

proc = ROOT / "data" / "mendeley_processed"
print("=== FILES IN mendeley_processed ===")
for f in sorted(proc.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

print()
print("=== LOADING mendeley_combined.pkl ===")
df = pd.read_pickle(str(proc / "mendeley_combined.pkl"))
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print()
print("=== CELL DISTRIBUTION ===")
for cid, grp in df.groupby("cell_id"):
    print(
        f"  Cell {cid}: {len(grp)} samples, "
        f"aging_cycle [{grp.aging_cycle.min()}-{grp.aging_cycle.max()}]"
    )

print()
print("=== LABEL DISTRIBUTIONS (model-derived from ECM fitting) ===")
for col in ["lli_pct", "lam_pct", "cl_pct"]:
    if col in df.columns:
        v = df[col].dropna()
        print(
            f"{col}: n={len(v)} min={v.min():.2f} max={v.max():.2f} "
            f"mean={v.mean():.2f} std={v.std():.2f} NaN={df[col].isna().sum()}"
        )
        pct = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
        print(f"  percentiles [p1,p5,p25,p50,p75,p95,p99]: {np.round(pct, 2)}")
    else:
        print(f"{col}: MISSING")

print()
print("=== EIS FEATURE STATS ===")
eis_cols = ["zmod_ohm", "zphz_deg", "zreal_ohm", "zimg_ohm",
            "R_electrolyte", "R_ct1", "R_ct2", "Zw", "ocv_v"]
for col in eis_cols:
    if col in df.columns:
        v = df[col].dropna()
        print(
            f"{col:15s}: min={v.min():.5f}  max={v.max():.5f}  "
            f"mean={v.mean():.5f}  NaN={df[col].isna().sum()}"
        )
    else:
        print(f"  {col}: MISSING")

print()
print("=== SOH ===")
for col in ["soh_norm", "soh_pct"]:
    if col in df.columns:
        v = df[col].dropna()
        print(f"{col}: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")
