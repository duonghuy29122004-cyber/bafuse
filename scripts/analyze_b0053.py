"""
Analyze B0053 vs other test batteries -- capacity trajectory + SoH distribution.
"""
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load paired data
paired_path = ROOT / "data" / "processed" / "paired.pkl"
if not paired_path.exists():
    # Re-run quick parse/pair
    sys.path.insert(0, str(ROOT / "src"))
    from src.data.parse_mat import parse_all_mat_files
    from src.data.pairing import pair_discharge_eis
    discharge_df, eis_df = parse_all_mat_files("5. BatteryDataSet")
    paired_df = pair_discharge_eis(discharge_df, eis_df, max_cycle_gap=10)
    paired_df.to_pickle(paired_path)
else:
    paired_df = pd.read_pickle(paired_path)

TEST_BATTERIES = ["B0005", "B0026", "B0039", "B0045", "B0048", "B0051", "B0053"]

print("=" * 65)
print("Capacity trajectory -- test batteries")
print("=" * 65)
print(f"{'Battery':8}  {'N_cyc':6}  {'cap_min':8}  {'cap_max':8}  {'cap_mean':8}  {'soh_min':7}  {'soh_max':7}  {'range':7}")
print("-" * 65)

for bid in TEST_BATTERIES:
    grp = paired_df[paired_df["battery_id"] == bid]
    if grp.empty:
        print(f"{bid:8}  NO DATA")
        continue
    cap   = grp["capacity_ahr"]
    soh   = (cap / 2.0 * 100).clip(0, 100)
    print(
        f"{bid:8}  {len(grp):6}  {cap.min():8.4f}  {cap.max():8.4f}  "
        f"{cap.mean():8.4f}  {soh.min():7.1f}  {soh.max():7.1f}  "
        f"{(soh.max()-soh.min()):7.1f}"
    )

# B0053 detailed
print()
print("=" * 65)
print("B0053 -- detailed capacity per cycle")
print("=" * 65)
b53 = paired_df[paired_df["battery_id"] == "B0053"].sort_values("discharge_cycle")
print(b53[["discharge_cycle", "capacity_ahr", "impedance_ohm", "voltage_min"]].to_string(index=False))

# Compare with a "normal" battery from train set
print()
print("=" * 65)
print("B0005 (train) -- capacity per cycle (first 20)")
print("=" * 65)
b5 = paired_df[paired_df["battery_id"] == "B0005"].sort_values("discharge_cycle").head(20)
print(b5[["discharge_cycle", "capacity_ahr", "impedance_ohm", "voltage_min"]].to_string(index=False))

