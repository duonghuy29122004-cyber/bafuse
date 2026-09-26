"""
SOH Scale Diagnostic Script.

Traces the full pipeline:
  raw capacity -> SOH calculation -> dataset target -> model output -> metrics

Prints min/max/mean/std at every step.
"""
import sys, logging
import numpy as np
import torch

ROOT = __file__[:__file__.rfind("scripts")]
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "src")

logging.basicConfig(level=logging.WARNING)

# ── 1. Load processed NASA data ──────────────────────────────────────────────
import pandas as pd
from pathlib import Path

proc = Path(ROOT) / "data" / "processed"
train_df = pd.read_pickle(str(proc / "train.pkl"))
val_df   = pd.read_pickle(str(proc / "val.pkl"))
test_df  = pd.read_pickle(str(proc / "test.pkl"))

SEP = "=" * 62

def stats(arr, label):
    a = np.asarray(arr, dtype=float).ravel()
    print(f"  {label:<30s}  min={a.min():.4f}  max={a.max():.4f}"
          f"  mean={a.mean():.4f}  std={a.std():.4f}")

print(SEP)
print("STEP 1 — Raw capacity_ahr in NASA paired DataFrame")
print(SEP)
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    stats(df["capacity_ahr"].values, f"{name} capacity_ahr (Ah)")

# ── 2. SOH as computed in BaFuseDataset ──────────────────────────────────────
print()
print(SEP)
print("STEP 2 — SOH label = (capacity_ahr / 2.0) * 100   [0-100 scale]")
print(SEP)
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    soh = np.clip((df["capacity_ahr"].values / 2.0) * 100.0, 0, 100)
    stats(soh, f"{name} soh_label [0-100]")

# ── 3. Sample 10 rows to manually verify ─────────────────────────────────────
print()
print(SEP)
print("STEP 3 — Manual spot-check of 10 train samples")
print(SEP)
sample = train_df.sample(10, random_state=42)[["battery_id","discharge_cycle","capacity_ahr"]].copy()
sample["soh_0_100"] = np.clip((sample["capacity_ahr"] / 2.0) * 100.0, 0, 100)
sample["soh_0_1"]   = sample["soh_0_100"] / 100.0
print(sample.to_string(index=False))

# ── 4. DataLoader targets ────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 4 — Actual dataset __getitem__ soh_label (from DataLoader)")
print(SEP)
from src.data.dataset import create_dataloaders, BaFuseDataset

disc_pkl = proc / "discharge_raw.pkl"
disc_df  = pd.read_pickle(str(disc_pkl)) if disc_pkl.exists() else None

train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    discharge_data_df=disc_df,
    batch_size=256, num_workers=0,
)

all_targets = []
for batch in train_loader:
    all_targets.append(batch["soh_label"].numpy())
all_targets = np.concatenate(all_targets)
stats(all_targets, "train soh_label from DataLoader")
print(f"  -> Scale is [0-1]? {all_targets.max() <= 1.05}")
print(f"  -> Scale is [0-100]? {all_targets.max() > 5.0}")

# ── 5. Untrained model output ─────────────────────────────────────────────────
print()
print(SEP)
print("STEP 5 — Untrained model output (raw, no activation)")
print(SEP)
from src.models.bafuse_v2 import BaFuseV2

sample_batch = next(iter(train_loader))
d_shape = tuple(sample_batch["discharge"].shape)
e_shape = tuple(sample_batch["eis"].shape)
p_shape = tuple(sample_batch["physics"].shape)
print(f"  Batch shapes: discharge={d_shape}  eis={e_shape}  physics={p_shape}")

model = BaFuseV2(
    discharge_input_size=d_shape[-1],
    eis_num_frequencies=e_shape[-1],
    physics_num_features=p_shape[-1],
    latent_dim=64, fusion_dim=128,
)
model.eval()
with torch.no_grad():
    out = model(sample_batch["discharge"], sample_batch["eis"], sample_batch["physics"])
raw_preds = out["soh_pred"].view(-1).numpy()
targets   = sample_batch["soh_label"].view(-1).numpy()
stats(raw_preds, "untrained model soh_pred (raw)")
stats(targets,   "batch soh_label (target)")

# ── 6. Loss magnitude check ───────────────────────────────────────────────────
print()
print(SEP)
print("STEP 6 — Loss magnitude with correct vs wrong scale")
print(SEP)
from src.losses import SoHPredictionLoss

criterion = SoHPredictionLoss(base_loss="mse", lambda_physics=0.0, lambda_smooth=0.0)
preds_t = out["soh_pred"].view(-1)
tgts_t  = sample_batch["soh_label"].view(-1)

loss_actual = criterion(preds_t, tgts_t).item()
print(f"  MSE loss (targets as-is):          {loss_actual:.2f}")

# What would it look like if target was accidentally 0-1 but model outputs ~0?
fake_target_01 = tgts_t / 100.0  # pretend target is 0-1
loss_01 = criterion(preds_t, fake_target_01).item()
print(f"  MSE loss (targets forced to 0-1):  {loss_01:.6f}")

# Scale of targets
print(f"  Target scale:  min={tgts_t.min():.3f}  max={tgts_t.max():.3f}")
print(f"  Pred scale:    min={preds_t.min():.3f}  max={preds_t.max():.3f}")
print()

# ── 7. Diagnosis summary ──────────────────────────────────────────────────────
print(SEP)
print("DIAGNOSIS SUMMARY")
print(SEP)
target_scale_100 = all_targets.max() > 5.0
target_scale_01  = all_targets.max() <= 1.05
print(f"  Dataset soh_label scale:  {'[0-100]' if target_scale_100 else '[0-1]'}")
print(f"  Untrained model output range: [{raw_preds.min():.3f}, {raw_preds.max():.3f}]")
if target_scale_100 and raw_preds.max() < 5.0:
    print()
    print("  *** ROOT CAUSE IDENTIFIED ***")
    print("  Target is [0-100] but untrained model outputs near-zero (raw MLP).")
    print("  MSE loss = ~mean(target^2) ~ (mean_soh)^2 ~ 80^2 = 6400.")
    print("  This matches observed loss ~5941.")
    print()
    print("  FIX OPTIONS:")
    print("  A) Normalize target to [0-1]:  soh = capacity/2.0  (remove *100)")
    print("     Pro: model regresses on [0,1], loss ~0.01 range")
    print("     Con: must multiply predictions by 100 for % reporting")
    print()
    print("  B) Keep [0-100], add sigmoid*100 or initialize final bias=80")
    print("     Pro: no scale change needed")
    print("     Con: harder to initialize well")
    print()
    print("  RECOMMENDED: Option A — normalize to [0-1] throughout.")
    print("  Report MAE_pct = MAE*100, RMSE_pct = RMSE*100.")
elif target_scale_01:
    print("  Target already [0-1]. Check model output range.")
    if raw_preds.max() < 0.1:
        print("  Model output also near 0 — initialization issue or missing activation.")
    else:
        print("  Scales look compatible.")
else:
    print("  Could not determine scale definitively — check manually.")
print(SEP)
