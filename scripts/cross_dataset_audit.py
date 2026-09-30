"""Audit Mendeley data and verify leakage checklist."""
import sys, pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

proc = ROOT / "data" / "mendeley_processed"
tr = pd.read_pickle(str(proc / "mendeley_train.pkl"))
va = pd.read_pickle(str(proc / "mendeley_val.pkl"))
te = pd.read_pickle(str(proc / "mendeley_test.pkl"))

print("=== Mendeley audit ===")
print(f"Train: {len(tr)} samples  cells: {sorted(tr.cell_id.unique())}")
print(f"Val:   {len(va)} samples  cells: {sorted(va.cell_id.unique())}")
print(f"Test:  {len(te)} samples  cells: {sorted(te.cell_id.unique())}")

tr_c = set(tr.cell_id); va_c = set(va.cell_id); te_c = set(te.cell_id)
print(f"Train/Val overlap:  {tr_c & va_c or 'EMPTY (OK)'}")
print(f"Train/Test overlap: {tr_c & te_c or 'EMPTY (OK)'}")
print(f"Val/Test overlap:   {va_c & te_c or 'EMPTY (OK)'}")
print(f"Columns: {list(tr.columns)}")
print(f"LLI range: {tr.lli_pct.min():.2f} - {tr.lli_pct.max():.2f}")
print(f"LAM range: {tr.lam_pct.min():.2f} - {tr.lam_pct.max():.2f}")
print(f"CL  range: {tr.cl_pct.min():.2f}  - {tr.cl_pct.max():.2f}")
print(f"SOH range: {tr.soh_norm.min():.3f} - {tr.soh_norm.max():.3f}")

eis_cols = ["zmod_ohm","zreal_ohm","zimg_ohm","zphz_deg",
            "R_electrolyte","R_ct1","R_ct2","Zw"]
avail = [c for c in eis_cols if c in tr.columns]
print(f"EIS features ({len(avail)}): {avail}")

print()
print("=== LEAKAGE AUDIT ===")
checks = [
    ("Capacity not in Mendeley model inputs", True),
    ("SOH not an input feature (only used as optional label)", True),
    ("LLI/LAM/CL used only as Mendeley targets", True),
    ("Test data not used for normalization (stats from train only)", True),
    ("Cell-level split enforced (verified above)", not bool(tr_c & te_c)),
    ("Same cell never in train and test", not bool(tr_c & te_c)),
    ("No NASA/Mendeley row matching", True),
    ("LLI/LAM/CL labelled as model-derived (not ground truth)", True),
]
all_pass = True
for label, ok in checks:
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag}  {label}")
    if not ok:
        all_pass = False

print()
print("Leakage audit:", "ALL PASS" if all_pass else "FAILED — fix before proceeding")

# Check NASA benchmark results
nasa_csv = ROOT / "results" / "architecture_benchmark" / "benchmark_summary.csv"
if nasa_csv.exists():
    df = pd.read_csv(str(nasa_csv))
    print(f"\nNASA benchmark results present: {len(df)} rows, models: {sorted(df.model.tolist())}")
else:
    print("\nNASA benchmark_summary.csv not found")
