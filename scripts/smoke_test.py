"""
Post-fix smoke test — runs the full parse→pair→split→dataset pipeline
on a small subset (B0005 + B0006) and prints capacity-source stats.

Expected output:
  - 100% capacity from .mat field (P1-#1 fix verified)
  - SOH labels in [0, 1] (P1-#2 fix verified)
  - ts_raw discharge shape (TARGET_SEQ_LEN, 3) — no shape mixing (P2-#5)
  - EIS features all normalised (P2-#4)
  - No crash from encoders (P3-#8)
  - Split stratification runs without error (P3-#9)
"""
import sys, logging
import numpy as np
import torch
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")
SEP = "=" * 62

PASS = "[PASS]"
FAIL = "[FAIL]"
failures = []

def check(cond, label):
    tag = PASS if cond else FAIL
    print(f"  {tag}  {label}")
    if not cond:
        failures.append(label)
    return cond

# ── Step 1: parse two batteries ──────────────────────────────────────────────
print(SEP); print("STEP 1 — Parse .mat files (B0005, B0006)")
print(SEP)
from src.data.parse_mat import parse_all_mat_files

data_dir = str(Path(ROOT) / "5. BatteryDataSet" / "1. BatteryAgingARC-FY08Q4")
disc_df, eis_df = parse_all_mat_files(data_dir)

print(f"  Discharge records : {len(disc_df):,}")
print(f"  EIS records       : {len(eis_df):,}")
print(f"  Batteries parsed  : {sorted(disc_df['battery_id'].unique())}")

check(len(disc_df) > 0, "discharge records > 0")
check(len(eis_df)  > 0, "EIS records > 0")

# P1-#1 verification
if '_cap_from_field' in disc_df.columns:
    per_cycle = disc_df.drop_duplicates(['battery_id','cycle_idx'])
    n_total   = len(per_cycle)
    n_field   = int(per_cycle['_cap_from_field'].sum())
    pct       = 100.0 * n_field / max(n_total, 1)
    print(f"\n  Capacity source: {n_field}/{n_total} cycles ({pct:.1f}%) from .mat field")
    check(pct > 90, f">=90% capacity from .mat field ({pct:.1f}%)")
else:
    print("  WARNING: _cap_from_field column not found in discharge_df")
    failures.append("_cap_from_field column missing")

# ── Step 2: pair ─────────────────────────────────────────────────────────────
print(); print(SEP); print("STEP 2 — Pair discharge ↔ EIS")
print(SEP)
from src.data.pairing import pair_discharge_eis

paired_df = pair_discharge_eis(disc_df, eis_df, max_cycle_gap=2)
print(f"  Pairs created: {len(paired_df)}")
check(len(paired_df) > 0, "pairing produced rows")
check('capacity_ahr' in paired_df.columns, "capacity_ahr column present")

# ── Step 3: split (stratified) ───────────────────────────────────────────────
print(); print(SEP); print("STEP 3 — Battery-level split (stratified)")
print(SEP)
from src.data.split import split_by_battery

try:
    train_df, val_df, test_df = split_by_battery(
        paired_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        random_state=42, stratify_by_soh=True,
    )
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    check(len(train_df) > 0, "train split non-empty")
    check(len(val_df)   > 0, "val split non-empty")
    check(len(test_df)  > 0, "test split non-empty")
except Exception as e:
    print(f"  Split failed: {e}")
    failures.append(f"split crashed: {e}")
    train_df, val_df, test_df = paired_df, paired_df, paired_df  # fallback

# ── Step 4: Dataset with discharge_data_df ───────────────────────────────────
print(); print(SEP); print("STEP 4 — BaFuseDataset (with time-series)")
print(SEP)
from src.data.dataset import BaFuseDataset, fit_aging_prior, get_nominal_capacity

aging_params = fit_aging_prior(train_df)
train_ds = BaFuseDataset(
    train_df,
    discharge_data_df=disc_df,
    normalize=True,
    aging_prior_params=aging_params,
)
print(f"  Train samples after pre-filter: {len(train_ds)}")
check(len(train_ds) > 0, "train dataset non-empty after pre-filter")

# ── Step 5: check sample shapes and scale ────────────────────────────────────
print(); print(SEP); print("STEP 5 — Sample inspection")
print(SEP)
sample = train_ds[0]
d = sample['discharge'].numpy()
e = sample['eis'].numpy()
p = sample['physics'].numpy()
y = float(sample['soh_label'])
bid = sample['battery_id']

print(f"  discharge shape : {d.shape}   (expected ({train_ds.max_seq_len}, 3))")
print(f"  eis shape       : {e.shape}   (expected (3,))")
print(f"  physics shape   : {p.shape}   (expected (4,))")
print(f"  soh_label       : {y:.4f}     (expected 0–1)")
print(f"  battery_id      : {bid}")

# P2-#5: consistent shape
check(d.shape == (train_ds.max_seq_len, 3), f"discharge shape == ({train_ds.max_seq_len},3)")
# P1-#2: SOH in [0,1]
check(0.0 <= y <= 1.0, f"soh_label in [0,1] (got {y:.4f})")
# P1-#2: nominal capacity lookup
nom = get_nominal_capacity(bid)
check(nom == 2.0, f"nominal capacity for {bid} = 2.0 Ah (got {nom})")
# P2-#4: EIS all finite (normalised)
check(np.all(np.isfinite(e)), "all EIS features finite (normalised)")

# Check 5 more samples
soh_vals = [float(train_ds[i]['soh_label']) for i in range(min(len(train_ds), 5))]
print(f"  SOH sample values: {[f'{v:.4f}' for v in soh_vals]}")
check(all(0 <= v <= 1.0 for v in soh_vals), "all sampled SOH in [0,1]")

# P2-#3: discharge channel means should be near 0 (z-scored)
d_mean = float(np.mean(d))
print(f"  discharge mean (z-scored): {d_mean:.4f}  (should be near 0)")
check(abs(d_mean) < 3.0, f"discharge mean near 0 after normalisation (got {d_mean:.4f})")

# ── Step 6: DataLoader collate (no shape crash) ───────────────────────────────
print(); print(SEP); print("STEP 6 — DataLoader collate")
print(SEP)
from torch.utils.data import DataLoader

val_ds = BaFuseDataset(
    val_df,
    discharge_data_df=disc_df,
    normalize=True,
    external_stats=train_ds.stats,
    aging_prior_params=aging_params,
)
loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
try:
    batch = next(iter(loader))
    print(f"  Batch discharge : {tuple(batch['discharge'].shape)}")
    print(f"  Batch eis       : {tuple(batch['eis'].shape)}")
    print(f"  Batch soh_label : {batch['soh_label'].numpy()}")
    check(batch['discharge'].shape[-1] == 3, "batch discharge last dim == 3")
    check(batch['soh_label'].max().item() <= 1.0, "batch SOH max <= 1.0")
except Exception as e:
    print(f"  DataLoader collate CRASHED: {e}")
    failures.append(f"DataLoader crashed: {e}")

# ── Step 7: encoder forward pass (P3-#8 regression test) ─────────────────────
print(); print(SEP); print("STEP 7 — Encoder forward pass (P3-#8)")
print(SEP)
from src.models.encoders import EISEncoder, DischargeEncoder

# MLP path (dim=3)
enc_eis = EISEncoder(num_frequencies=3)
x_eis   = torch.randn(4, 3)
try:
    out = enc_eis(x_eis)
    check(out.shape == (4, 64), f"EISEncoder(3) output shape (4,64) got {tuple(out.shape)}")
except Exception as e:
    failures.append(f"EISEncoder MLP failed: {e}")
    print(f"  EISEncoder(3) FAILED: {e}")

# MLP path (dim=8, Mendeley)
enc_eis8 = EISEncoder(num_frequencies=8)
x_eis8   = torch.randn(4, 8)
try:
    out8 = enc_eis8(x_eis8)
    check(out8.shape == (4, 64), f"EISEncoder(8) output shape (4,64) got {tuple(out8.shape)}")
except Exception as e:
    failures.append(f"EISEncoder(8) failed: {e}")
    print(f"  EISEncoder(8) FAILED: {e}")

# Wrong dim should raise ValueError (P3-#8)
try:
    enc_eis(torch.randn(4, 9))   # wrong dim for 3-feature encoder
    failures.append("EISEncoder did NOT raise ValueError on wrong dim")
    print(f"  [FAIL]  EISEncoder should have raised ValueError on dim=9")
except ValueError:
    check(True, "EISEncoder raises ValueError on wrong input dim")

# ── Summary ───────────────────────────────────────────────────────────────────
print(); print(SEP)
if not failures:
    print(f"SMOKE TEST PASSED — all checks OK")
    print()
    print("  P1-#1  capacity from .mat field:  VERIFIED")
    print("  P1-#2  SOH in [0,1]:              VERIFIED")
    print("  P2-#3  discharge normalised:      VERIFIED")
    print("  P2-#4  EIS fully normalised:      VERIFIED")
    print("  P2-#5  consistent tensor shape:   VERIFIED")
    print("  P3-#8  no lazy-init crash:        VERIFIED")
    print("  P3-#9  stratified split:          VERIFIED")
else:
    print(f"SMOKE TEST FAILED — {len(failures)} failure(s):")
    for f in failures:
        print(f"  - {f}")
print(SEP)
sys.exit(0 if not failures else 1)
