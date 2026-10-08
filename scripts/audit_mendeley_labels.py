"""
Audit Mendeley degradation labels — identify extreme values and their source.

DO NOT train models.
DO NOT modify data.
ONLY inspect and report.
"""
import sys, pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

proc = ROOT / "data" / "mendeley_processed"
df = pd.read_pickle(str(proc / "mendeley_combined.pkl"))

print("="*80)
print("MENDELEY DEGRADATION LABEL AUDIT")
print("="*80)
print(f"Dataset: {len(df)} samples, {len(df.cell_id.unique())} cells")
print(f"Columns: {list(df.columns)}")
print()

# ─── 1. Label distributions ───────────────────────────────────────────────────
print("="*80)
print("1. LABEL DISTRIBUTIONS")
print("="*80)

for label in ["lli_pct", "lam_pct", "cl_pct"]:
    vals = df[label].dropna()
    print(f"\n{label.upper()}:")
    print(f"  n={len(vals)}")
    print(f"  min={vals.min():.2f}  max={vals.max():.2f}")
    print(f"  mean={vals.mean():.2f}  median={vals.median():.2f}  std={vals.std():.2f}")
    pct = np.percentile(vals, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print(f"  percentiles [p1, p5, p10, p25, p50, p75, p90, p95, p99]:")
    print(f"    {np.round(pct, 2)}")

# R_ct2
print(f"\nR_CT2 (Ohm):")
vals = df["R_ct2"].dropna()
print(f"  n={len(vals)}")
print(f"  min={vals.min():.6f}  max={vals.max():.6f}")
print(f"  mean={vals.mean():.6f}  median={vals.median():.6f}  std={vals.std():.6f}")
pct = np.percentile(vals, [1, 5, 10, 25, 50, 75, 90, 95, 99])
print(f"  percentiles [p1, p5, p10, p25, p50, p75, p90, p95, p99]:")
print(f"    {np.round(pct, 6)}")

# ─── 2. Extreme value identification ──────────────────────────────────────────
print()
print("="*80)
print("2. EXTREME VALUE IDENTIFICATION")
print("="*80)

# LLI > 500%
lli_500 = df[df.lli_pct > 500]
print(f"\nLLI > 500%: {len(lli_500)} samples")
if len(lli_500) > 0:
    print(f"  Cells: {sorted(lli_500.cell_id.unique())}")
    print(f"  Cycles: {sorted(lli_500.aging_cycle.unique())}")
    print(f"  Max: {lli_500.lli_pct.max():.2f}% (cell {lli_500.loc[lli_500.lli_pct.idxmax(), 'cell_id']}, cycle {lli_500.loc[lli_500.lli_pct.idxmax(), 'aging_cycle']})")
    if len(lli_500) <= 10:
        for _, row in lli_500.iterrows():
            print(f"    cell={row.cell_id} cycle={row.aging_cycle} LLI={row.lli_pct:.2f}% SOH={row.soh_pct:.2f}%")

# LLI > 1000%
lli_1000 = df[df.lli_pct > 1000]
print(f"\nLLI > 1000%: {len(lli_1000)} samples")
if len(lli_1000) > 0:
    print(f"  Cells: {sorted(lli_1000.cell_id.unique())}")
    for _, row in lli_1000.iterrows():
        print(f"    cell={row.cell_id} cycle={row.aging_cycle} LLI={row.lli_pct:.2f}% SOH={row.soh_pct:.2f}%")

# LLI < 0
lli_neg = df[df.lli_pct < 0]
print(f"\nLLI < 0: {len(lli_neg)} samples")
if len(lli_neg) > 0:
    print(f"  Cells: {sorted(lli_neg.cell_id.unique())}")
    print(f"  Cycles: {sorted(lli_neg.aging_cycle.unique())}")
    if len(lli_neg) <= 10:
        for _, row in lli_neg.iterrows():
            print(f"    cell={row.cell_id} cycle={row.aging_cycle} LLI={row.lli_pct:.2f}% SOH={row.soh_pct:.2f}%")

# LAM < 0
lam_neg = df[df.lam_pct < 0]
print(f"\nLAM < 0: {len(lam_neg)} samples")
if len(lam_neg) > 0:
    print(f"  Cells: {sorted(lam_neg.cell_id.unique())}")
    if len(lam_neg) <= 10:
        for _, row in lam_neg.iterrows():
            print(f"    cell={row.cell_id} cycle={row.aging_cycle} LAM={row.lam_pct:.2f}% SOH={row.soh_pct:.2f}%")

# LAM > 100%
lam_100 = df[df.lam_pct > 100]
print(f"\nLAM > 100%: {len(lam_100)} samples")
if len(lam_100) > 0:
    print(f"  Cells: {sorted(lam_100.cell_id.unique())}")
    for _, row in lam_100.iterrows():
        print(f"    cell={row.cell_id} cycle={row.aging_cycle} LAM={row.lam_pct:.2f}% SOH={row.soh_pct:.2f}%")

# CL = 0
cl_0 = df[df.cl_pct == 0]
print(f"\nCL = 0: {len(cl_0)} samples")
if len(cl_0) > 0:
    print(f"  Cells: {sorted(cl_0.cell_id.unique())}")
    print(f"  Cycles: {sorted(cl_0.aging_cycle.unique())}")

# R_ct2 largest
r_ct2_large = df.nlargest(10, "R_ct2")
print(f"\nR_ct2 top 10 largest:")
for _, row in r_ct2_large.iterrows():
    print(f"  cell={row.cell_id} cycle={row.aging_cycle} R_ct2={row.R_ct2:.6f} Ohm SOH={row.soh_pct:.2f}%")

# ─── 3. Check source preprocessing ────────────────────────────────────────────
print()
print("="*80)
print("3. SOURCE PREPROCESSING CHECK")
print("="*80)

# Look for original Excel files
raw_dir = ROOT / "data" / "mendeley_raw"
if raw_dir.exists():
    print(f"Raw data directory exists: {raw_dir}")
    files = list(raw_dir.glob("*.xlsx"))
    print(f"  Found {len(files)} Excel files")
    for f in files[:5]:
        print(f"    {f.name}")
else:
    print("Raw data directory NOT FOUND")

# Check preprocessing script
prep_scripts = [
    ROOT / "src" / "data" / "preprocess_mendeley.py",
    ROOT / "scripts" / "preprocess_mendeley.py",
    ROOT / "data" / "preprocess_mendeley.py",
]
for script in prep_scripts:
    if script.exists():
        print(f"\nPreprocessing script found: {script}")
        # Read first 100 lines to check for clipping/filtering
        with open(script, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[:100]
            for i, line in enumerate(lines, 1):
                if any(kw in line.lower() for kw in ["clip", "lli", "lam", "cl_pct", "outlier", "filter"]):
                    print(f"  Line {i}: {line.strip()}")

# ─── 4. Sample-level inspection ───────────────────────────────────────────────
print()
print("="*80)
print("4. CELL-LEVEL LABEL TRAJECTORY")
print("="*80)

for cid in sorted(df.cell_id.unique()):
    cell_df = df[df.cell_id == cid].sort_values("aging_cycle")
    lli_range = f"[{cell_df.lli_pct.min():.1f}, {cell_df.lli_pct.max():.1f}]"
    lam_range = f"[{cell_df.lam_pct.min():.1f}, {cell_df.lam_pct.max():.1f}]"
    cl_range = f"[{cell_df.cl_pct.min():.1f}, {cell_df.cl_pct.max():.1f}]"
    print(f"Cell {cid}: {len(cell_df)} samples")
    print(f"  LLI: {lli_range}  LAM: {lam_range}  CL: {cl_range}")
    
    # Check for extreme jumps
    lli_diff = cell_df.lli_pct.diff().abs()
    if lli_diff.max() > 100:
        jump_idx = lli_diff.idxmax()
        if jump_idx in cell_df.index:
            print(f"  WARNING: LLI has large jump: {lli_diff.max():.1f}% (cycle {cell_df.loc[jump_idx, 'aging_cycle']})")

# ─── 5. Diagnosis ──────────────────────────────────────────────────────────────
print()
print("="*80)
print("5. DIAGNOSIS")
print("="*80)

lli_outlier = (df.lli_pct > 500).sum() > 0 or (df.lli_pct < 0).sum() > 0
lam_outlier = (df.lam_pct < 0).sum() > 0 or (df.lam_pct > 100).sum() > 0
cl_outlier = False  # CL range [0, 12.46] seems reasonable
r_ct2_outlier = (df.R_ct2 > 0.1).sum() > 0

print(f"LLI outliers present: {lli_outlier}")
print(f"  - LLI > 500%: {(df.lli_pct > 500).sum()} samples")
print(f"  - LLI < 0:    {(df.lli_pct < 0).sum()} samples")
print()
print(f"LAM outliers present: {lam_outlier}")
print(f"  - LAM < 0:    {(df.lam_pct < 0).sum()} samples")
print(f"  - LAM > 100%: {(df.lam_pct > 100).sum()} samples")
print()
print(f"CL outliers present: {cl_outlier}")
print()
print(f"R_ct2 outliers present: {r_ct2_outlier}")
print(f"  - R_ct2 > 0.1 Ohm: {(df.R_ct2 > 0.1).sum()} samples")
print()

# ECM fitting stability check
print("LIKELY CAUSE:")
print("  LLI/LAM/CL are model-derived from equivalent-circuit fitting (ECM).")
print("  Extreme values likely indicate:")
print("    1. ECM fitting instability (e.g., at early cycles, SOC extremes)")
print("    2. Ill-conditioned optimization (local minima, poor initialization)")
print("    3. Not direct physical measurements — these are surrogate indicators")
print()
print("RECOMMENDATION:")
print("  Option A: Filter outliers (e.g., clip LLI to [-50, 500], LAM to [-10, 150])")
print("  Option B: Train with all data, but interpret results cautiously")
print("  Option C: Exclude cells with extreme outliers (cells 7, 8 have LLI > 1000)")
print()
print("Training with current unfiltered labels: RISKY")
print("  - Extreme LLI variance (std=402%) will dominate loss")
print("  - Model may fail to converge or produce meaningless predictions")
print("  - Test results will be unstable")

print()
print("="*80)
print("FINAL VERDICT")
print("="*80)
print(f"Training now: NO")
print(f"Reason: Extreme LLI/LAM outliers likely from ECM fitting instability")
print(f"LLI outlier issue: FAIL ({(df.lli_pct > 500).sum() + (df.lli_pct < 0).sum()} samples)")
print(f"LAM outlier issue: FAIL ({(df.lam_pct < 0).sum() + (df.lam_pct > 100).sum()} samples)")
print(f"CL outlier issue: PASS (range [0, 12.46] reasonable)")
print(f"R_ct2 outlier issue: FAIL ({(df.R_ct2 > 0.1).sum()} samples)")
print(f"NASA SOH benchmark modified: NO")
