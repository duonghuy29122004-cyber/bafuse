"""
Validate data pipeline after bug fixes.
Prints stats for: impedance, capacity, cycle_gap, sequence shape.
"""
import sys, logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

from src.data.parse_mat import parse_all_mat_files
from src.data.pairing import pair_discharge_eis, validate_pairs
from src.data.dataset import create_dataloaders, TARGET_SEQ_LEN
from src.data.split import split_by_battery

SEP = "=" * 60

# ── STEP 1: Parse ─────────────────────────────────────────────
print(SEP)
print("STEP 1 -- Parse .mat files (deduplicated)")
print(SEP)
discharge_df, eis_df = parse_all_mat_files("5. BatteryDataSet")
print(f"  discharge records : {len(discharge_df):,}")
print(f"  EIS records       : {len(eis_df):,}")
print(f"  unique batteries  : {discharge_df['battery_id'].nunique()}")

# quick EIS sanity check before pairing
if not eis_df.empty:
    print(f"  raw imp range     : {eis_df['impedance_ohm'].min():.4f} – {eis_df['impedance_ohm'].max():.4f} Ohm")
    print(f"  raw re  range     : {eis_df['re_ohm'].min():.4f} – {eis_df['re_ohm'].max():.4f} Ohm")
    print(f"  raw rct range     : {eis_df['rct_ohm'].min():.4f} – {eis_df['rct_ohm'].max():.4f} Ohm")

# ── STEP 2: Pair ──────────────────────────────────────────────
print()
print(SEP)
print("STEP 2 -- Pair discharge <-> EIS  (max_cycle_gap=10)")
print(SEP)
paired_df = pair_discharge_eis(discharge_df, eis_df, max_cycle_gap=10)
stats = validate_pairs(paired_df)

print()
print("  === VALIDATION STATS (post-fix) ===")
print(f"  pairs             : {stats['num_pairs']}")
print(f"  batteries         : {stats['num_batteries']}")
print(f"  capacity_range    : {stats['capacity_range'][0]:.4f} – {stats['capacity_range'][1]:.4f} Ahr")
print(f"  impedance_range   : {stats['impedance_range'][0]:.4f} – {stats['impedance_range'][1]:.4f} Ohm")
print(f"  re_range          : {stats['re_range'][0]:.4f} – {stats['re_range'][1]:.4f} Ohm")
print(f"  rct_range         : {stats['rct_range'][0]:.4f} – {stats['rct_range'][1]:.4f} Ohm")
g = stats["cycle_gap_stats"]
print(f"  cycle_gap         : mean={g['mean']:.2f}  median={g['median']}  "
      f"std={g['std']:.2f}  min={g['min']}  max={g['max']}")
print(f"  missing_impedance : {stats['missing_impedance']}")
print(f"  missing_capacity  : {stats['missing_capacity']}")

# Physical plausibility checks
assert stats["capacity_range"][0] >= 0.5,        f"FAIL: min capacity {stats['capacity_range'][0]:.4f} < 0.5 Ahr"
assert stats["impedance_range"][1] <= 1000.0,     f"FAIL: max impedance {stats['impedance_range'][1]:.2f} > 1000 Ohm"
assert stats["impedance_range"][0] >= 0.0,        f"FAIL: min impedance {stats['impedance_range'][0]:.4f} < 0"
assert stats["cycle_gap_stats"]["max"] <= 10,     f"FAIL: max cycle_gap {stats['cycle_gap_stats']['max']} > 10"
assert stats["cycle_gap_stats"]["min"] >= 0,      f"FAIL: negative cycle_gap {stats['cycle_gap_stats']['min']}"
print()
print("  [CHECK] capacity   >= 0.5 Ahr          PASS")
print("  [CHECK] impedance  in [0, 1000] Ohm    PASS")
print("  [CHECK] cycle_gap  in [0, 10]          PASS")

# ── STEP 3: Split + DataLoader ────────────────────────────────
print()
print(SEP)
print("STEP 3 -- Split + DataLoader")
print(SEP)
train_df, val_df, test_df = split_by_battery(paired_df, stratify_by_soh=True)
print(f"  train/val/test    : {len(train_df)} / {len(val_df)} / {len(test_df)}")

train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    discharge_data_df=discharge_df,
    batch_size=16, num_workers=0,
)
batch = next(iter(train_loader))
d_shape = tuple(batch["discharge"].shape)
print(f"  discharge shape   : {d_shape}  (target_seq_len={TARGET_SEQ_LEN})")
print(f"  eis shape         : {tuple(batch['eis'].shape)}")
print(f"  physics shape     : {tuple(batch['physics'].shape)}")
soh = batch["soh_label"]
print(f"  soh range (batch) : {soh.min():.1f} – {soh.max():.1f} %")

assert d_shape[1] == TARGET_SEQ_LEN, f"FAIL: seq_len={d_shape[1]} != TARGET={TARGET_SEQ_LEN}"
print(f"  [CHECK] seq_len == {TARGET_SEQ_LEN}                 PASS")
assert soh.min() >= 0 and soh.max() <= 100, "FAIL: SoH out of [0,100]"
print("  [CHECK] SoH in [0, 100]                PASS")

print()
print(SEP)
print("ALL CHECKS PASSED -- pipeline is clean")
print(SEP)
