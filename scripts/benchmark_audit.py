"""Dry-run audit for Architecture Benchmark v1."""
import sys, numpy as np, pandas as pd, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

proc = ROOT / "data" / "processed"
train_df = pd.read_pickle(str(proc / "train.pkl"))
val_df   = pd.read_pickle(str(proc / "val.pkl"))
test_df  = pd.read_pickle(str(proc / "test.pkl"))
disc_df  = pd.read_pickle(str(proc / "discharge_raw.pkl"))

from src.data.dataset import create_dataloaders, get_nominal_capacity
tl, vl, tel = create_dataloaders(
    train_df, val_df, test_df,
    discharge_data_df=disc_df, batch_size=32
)

SEP = "=" * 65
print(SEP)
print("DATASET AUDIT — Architecture Benchmark v1")
print(SEP)

tr_b = sorted(train_df["battery_id"].unique())
va_b = sorted(val_df["battery_id"].unique())
te_b = sorted(test_df["battery_id"].unique())

print(f"Train  samples={len(train_df):4d}  batteries({len(tr_b)}): {tr_b}")
print(f"Val    samples={len(val_df):4d}  batteries({len(va_b)}): {va_b}")
print(f"Test   samples={len(test_df):4d}  batteries({len(te_b)}): {te_b}")
print()

tr_s = set(tr_b); va_s = set(va_b); te_s = set(te_b)
print(f"Train/Val  overlap : {tr_s & va_s or 'EMPTY (OK)'}")
print(f"Train/Test overlap : {tr_s & te_s or 'EMPTY (OK)'}")
print(f"Val/Test   overlap : {va_s & te_s or 'EMPTY (OK)'}")
print()

all_df = pd.concat([train_df, val_df, test_df])
dup = all_df.duplicated(subset=["battery_id", "discharge_cycle"]).sum()
print(f"Duplicate (battery_id, cycle) across splits: {dup}  {'OK' if dup==0 else 'WARNING'}")
print()

batch = next(iter(tl))
d_shape = tuple(batch["discharge"].shape)
e_shape = tuple(batch["eis"].shape)
p_shape = tuple(batch["physics"].shape)
soh_arr = batch["soh_label"].numpy()
print(f"discharge shape : {d_shape}   seq_len={d_shape[1]}  channels={d_shape[2]}")
print(f"eis shape       : {e_shape}   features={e_shape[1]}")
print(f"physics shape   : {p_shape}   features={p_shape[1]}")
print(f"SOH label range : min={soh_arr.min():.4f}  max={soh_arr.max():.4f}  mean={soh_arr.mean():.4f}")

# full SOH range across all splits
def _soh(df):
    return np.clip(
        df["capacity_ahr"].values /
        df["battery_id"].map(get_nominal_capacity).values,
        0, 1)

tr_soh = _soh(train_df); va_soh = _soh(val_df); te_soh = _soh(test_df)
print()
print(f"Train SOH : min={tr_soh.min():.4f}  max={tr_soh.max():.4f}  mean={tr_soh.mean():.4f}  std={tr_soh.std():.4f}")
print(f"Val   SOH : min={va_soh.min():.4f}  max={va_soh.max():.4f}  mean={va_soh.mean():.4f}  std={va_soh.std():.4f}")
print(f"Test  SOH : min={te_soh.min():.4f}  max={te_soh.max():.4f}  mean={te_soh.mean():.4f}  std={te_soh.std():.4f}")
print()

# paired_df column list
print(f"paired_df columns: {list(train_df.columns)}")
print()

# ── Stat features available for flat-input models (Model A/B) ──────────────
# These come from paired_df directly (no time-series needed)
stat_cols = [
    "voltage_mean", "voltage_min", "voltage_max", "voltage_std",
    "current_mean", "current_std",
    "temp_mean", "temp_std",
    "impedance_ohm", "re_ohm", "rct_ohm",
    "discharge_cycle",   # proxy for cycle_age
]
avail = [c for c in stat_cols if c in train_df.columns]
missing = [c for c in stat_cols if c not in train_df.columns]
print(f"Stat features available ({len(avail)}): {avail}")
if missing:
    print(f"  MISSING from paired_df: {missing}")

# duration_s column
has_dur = "duration_s" in train_df.columns
print(f"duration_s available: {has_dur}")
print()

# ── Parameter counts for all benchmark models ─────────────────────────────
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

from src.models.tiny_soh import TinySOH
from src.models.bafuse import BaFuse
from src.models.bafuse_v2 import BaFuseV2

d_in  = d_shape[-1]   # 3
e_in  = e_shape[-1]   # 3
p_in  = p_shape[-1]   # 4

# TinySOH (existing)
tiny = TinySOH(discharge_channels=d_in, eis_dim=e_in, physics_dim=p_in)

# BaFuse v1
baf1 = BaFuse(discharge_input_size=d_in, eis_num_frequencies=e_in,
              physics_num_features=p_in, latent_dim=64, fusion_dim=128)

# BaFuseV2
baf2 = BaFuseV2(discharge_input_size=d_in, eis_num_frequencies=e_in,
                physics_num_features=p_in, latent_dim=64, fusion_dim=128)

print("Parameter counts:")
print(f"  TinySOH    : {count_params(tiny):>10,}")
print(f"  BaFuse v1  : {count_params(baf1):>10,}")
print(f"  BaFuseV2   : {count_params(baf2):>10,}")
print()

# stat_in = number of flat input features for Model A/B
n_stat = len(avail) + (1 if has_dur else 0)
print(f"Flat-input feature count for Model A/B: {n_stat}")
print(SEP)
print("Audit complete — ready to implement benchmark models.")
