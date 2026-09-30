"""Inspect Samsung processed data and NASA normalization stats."""
import sys, pandas as pd, json, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src"))

# === Samsung data ===
df = pd.read_pickle(str(ROOT/"data"/"mendeley_processed"/"mendeley_test.pkl"))
print("=== Samsung test.pkl columns ===")
print(list(df.columns))
print(f"Shape: {df.shape}")
print()

print("=== What Samsung DOES have ===")
for col in df.columns:
    v = df[col]
    try:
        print(f"  {col:25s}  dtype={v.dtype}  min={v.min():.4f}  max={v.max():.4f}")
    except:
        print(f"  {col:25s}  dtype={v.dtype}  (non-numeric)")

print()
print("=== What Samsung does NOT have ===")
required_discharge = ['voltage_v','current_a','temperature_c','time_s']
required_physics   = ['discharge_cycle','voltage_min','voltage_mean']
for col in required_discharge + required_physics:
    has = col in df.columns
    print(f"  {col:30s}  {'PRESENT' if has else 'MISSING'}")

print()
print("=== OCV available? (could approximate voltage) ===")
print(f"  ocv_v present: {'ocv_v' in df.columns}")
if 'ocv_v' in df.columns:
    print(f"  ocv_v range: {df['ocv_v'].min():.3f} - {df['ocv_v'].max():.3f}")

# === NASA normalization stats ===
print()
print("=== NASA dataset normalization stats (from BaFuseDataset) ===")
nasa_train = pd.read_pickle(str(ROOT/"data"/"processed"/"train.pkl"))
from src.data.dataset import BaFuseDataset, fit_aging_prior
ds = BaFuseDataset(nasa_train, normalize=True)
print("NASA stats keys:", list(ds.stats.keys()))
for k, s in ds.stats.items():
    print(f"  {k:15s}  mean={s['mean']:.4f}  std={s['std']:.4f}")

# === Samsung mendeley_stats.json ===
print()
print("=== mendeley_stats.json (Samsung-derived, do NOT use for NASA eval) ===")
mstats_path = ROOT/"data"/"mendeley_processed"/"mendeley_stats.json"
if mstats_path.exists():
    with open(mstats_path) as f:
        mstats = json.load(f)
    for k, s in mstats.items():
        print(f"  {k:20s}  mean={s['mean']:.4f}  std={s['std']:.4f}")
else:
    print("  mendeley_stats.json NOT FOUND")

# === Key compatibility table ===
print()
print("=== COMPATIBILITY TABLE ===")
print(f"  {'Modality':20s}  {'NASA':8s}  {'Samsung':8s}  {'Compatible?':15s}  Notes")
print(f"  {'-'*75}")

rows = [
    ("Discharge V/I/T/t", True,  False, "NO",  "Samsung has no discharge time-series"),
    ("EIS zmod_ohm",      True,  'zmod_ohm' in df.columns, None, ""),
    ("EIS zreal_ohm",     True,  'zreal_ohm' in df.columns, None, ""),
    ("EIS zimg_ohm",      True,  'zimg_ohm' in df.columns, None, ""),
    ("EIS zphz_deg",      True,  'zphz_deg' in df.columns, None, ""),
    ("EIS R_electrolyte", True,  'R_electrolyte' in df.columns, None, ""),
    ("EIS R_ct1",         True,  'R_ct1' in df.columns, None, ""),
    ("EIS R_ct2",         True,  'R_ct2' in df.columns, None, ""),
    ("EIS Zw",            True,  'Zw' in df.columns, None, ""),
    ("Physics cycle_age", True,  'aging_cycle' in df.columns, None, "aging_cycle = cycle index"),
    ("Physics v_droop",   True,  False, "NO",  "Needs voltage_min — no discharge data"),
    ("Physics imp_rise",  True,  'zmod_ohm' in df.columns, "PARTIAL", "Can compute from EIS if we define BOL"),
    ("Physics fade_prior",True,  'aging_cycle' in df.columns, "PARTIAL", "Can use cycle age only (population prior)"),
    ("SOH label",         True,  'soh_norm' in df.columns, "YES (eval only)", ""),
]
for mod, nasa, sam, compat, note in rows:
    if compat is None:
        compat = "YES" if sam else "NO"
    sam_str = "YES" if sam else "NO"
    print(f"  {mod:20s}  {'YES':8s}  {sam_str:8s}  {compat:15s}  {note}")

print()
print("CONCLUSION:")
print("  Samsung CANNOT support full multimodal evaluation.")
print("  Discharge time-series: NOT available (no V/I/T/t).")
print("  Voltage droop: NOT computable (requires voltage_min from discharge).")
print("  EIS modality: AVAILABLE (8 features).")
print("  Physics (partial): cycle_age and impedance_rise can be approximated.")
print("  Physics (missing): voltage_droop cannot be computed.")
print("  SOH label: AVAILABLE for evaluation only.")
