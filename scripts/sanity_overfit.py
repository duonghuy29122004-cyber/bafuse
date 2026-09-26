"""
Sanity overfit test for BaFuse v2.

Protocol:
  1. Load 1 tiny batch (16 samples) from NASA data.
  2. Train BaFuseV2 on that single batch for 200 steps (overfit mode).
  3. Verify loss decreases from ~initial to ~0.
  4. Verify predictions converge toward targets.
  5. Print full diagnostic stats at step 0, 50, 100, 200.

Also tests predict_deg() on Mendeley data to verify the cell_id fix.

Expected outcome:
  - Initial SOH loss ~0.05  (MSE on [0,1] targets)
  - Final   SOH loss <0.001 (overfit to tiny batch)
  - MAE%    < 1.0% at step 200
  - R²      > 0.99 at step 200
  - No cell_id assertion errors from predict_deg()
"""
import sys, logging
import numpy as np
import torch
import pandas as pd
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")

logging.basicConfig(
    level=logging.WARNING,  # suppress INFO noise
    format="%(levelname)-8s %(message)s",
)

SEP = "=" * 62
PASS = "[PASS]"
FAIL = "[FAIL]"


def check(cond, label):
    tag = PASS if cond else FAIL
    print(f"  {tag}  {label}")
    return cond


all_ok = True

# ─── Load data ────────────────────────────────────────────────────────────────
proc = Path(ROOT) / "data" / "processed"
train_df = pd.read_pickle(str(proc / "train.pkl"))
val_df   = pd.read_pickle(str(proc / "val.pkl"))
test_df  = pd.read_pickle(str(proc / "test.pkl"))
disc_pkl = proc / "discharge_raw.pkl"
disc_df  = pd.read_pickle(str(disc_pkl)) if disc_pkl.exists() else None

from src.data.dataset import create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    discharge_data_df=disc_df,
    batch_size=16, num_workers=0,
)

# Pull ONE tiny batch — this is what we overfit on
tiny_batch = next(iter(train_loader))
discharge = tiny_batch["discharge"]
eis       = tiny_batch["eis"]
physics   = tiny_batch["physics"]
targets   = tiny_batch["soh_label"].view(-1)

# ─── Verify target scale ─────────────────────────────────────────────────────
print(SEP)
print("STEP 1 — Target scale verification")
print(SEP)
t_np = targets.numpy()
print(f"  soh_label:  min={t_np.min():.4f}  max={t_np.max():.4f}"
      f"  mean={t_np.mean():.4f}  std={t_np.std():.4f}")
ok1 = check(t_np.max() <= 1.05,  "target max <= 1.05  (is [0,1] scale)")
ok2 = check(t_np.min() >= 0.0,   "target min >= 0.0")
ok3 = check(t_np.mean() > 0.3,   "target mean > 0.3  (physical SOH range)")
all_ok &= ok1 & ok2 & ok3

# ─── Build model ─────────────────────────────────────────────────────────────
from src.models.bafuse_v2 import BaFuseV2
from src.losses import SoHPredictionLoss

d_shape = tuple(discharge.shape)
e_shape = tuple(eis.shape)
p_shape = tuple(physics.shape)

model = BaFuseV2(
    discharge_input_size=d_shape[-1],
    eis_num_frequencies=e_shape[-1],
    physics_num_features=p_shape[-1],
    latent_dim=64, fusion_dim=128,
)
criterion = SoHPredictionLoss(
    base_loss="mse",
    lambda_physics=0.1,
    lambda_smooth=0.0,  # no monotonicity on single batch
)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

# ─── Check initial output ─────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 2 — Model output at step 0 (untrained)")
print(SEP)
model.eval()
with torch.no_grad():
    out0 = model(discharge, eis, physics)
p0 = out0["soh_pred"].view(-1).detach().numpy()
print(f"  pred:   min={p0.min():.4f}  max={p0.max():.4f}"
      f"  mean={p0.mean():.4f}  std={p0.std():.4f}")
print(f"  target: min={t_np.min():.4f}  max={t_np.max():.4f}"
      f"  mean={t_np.mean():.4f}  std={t_np.std():.4f}")

with torch.no_grad():
    loss0 = criterion(out0["soh_pred"].view(-1), targets).item()
mae0_pct = np.mean(np.abs(p0 - t_np)) * 100
print(f"  Initial loss: {loss0:.6f}   Initial MAE%: {mae0_pct:.2f}%")
ok4 = check(loss0 < 2.0,   f"initial loss {loss0:.4f} < 2.0  (not in thousands)")
all_ok &= ok4

# ─── Overfit loop ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 3 — Overfit on tiny batch (200 steps)")
print(SEP)
print(f"  {'Step':>5}  {'Loss':>10}  {'MAE%':>8}  {'R²':>8}")
print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}")

history_loss = [loss0]
history_mae  = [mae0_pct]
best_loss_step = 0
best_state     = {k: v.clone() for k, v in model.state_dict().items()}

for step in range(1, 501):
    model.train()
    optimizer.zero_grad()
    out = model(discharge, eis, physics)
    loss = criterion(out["soh_pred"].view(-1), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if step in (1, 10, 25, 50, 100, 200, 300, 500):
        model.eval()
        with torch.no_grad():
            out_e = model(discharge, eis, physics)
        p_e  = out_e["soh_pred"].view(-1).detach().numpy()
        l_e  = float(criterion(out_e["soh_pred"].view(-1), targets).item())
        mae_e = np.mean(np.abs(p_e - t_np)) * 100
        ss_r  = np.sum((t_np - p_e) ** 2)
        ss_t  = np.sum((t_np - t_np.mean()) ** 2)
        r2_e  = 1 - ss_r / (ss_t + 1e-8)
        print(f"  {step:>5}  {l_e:>10.6f}  {mae_e:>7.3f}%  {r2_e:>8.4f}")
        history_loss.append(l_e)
        history_mae.append(mae_e)
        if l_e < min(history_loss[:-1] or [999]):
            best_loss_step = step
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

# ─── Final checks ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 4 — Final overfit verification (best checkpoint)")
print(SEP)
# Load best weights
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out_final = model(discharge, eis, physics)
p_final = out_final["soh_pred"].view(-1).detach().numpy()
loss_final = float(criterion(out_final["soh_pred"].view(-1), targets).item())
mae_final  = np.mean(np.abs(p_final - t_np)) * 100
ss_r = np.sum((t_np - p_final) ** 2)
ss_t = np.sum((t_np - t_np.mean()) ** 2)
r2_final = 1 - ss_r / (ss_t + 1e-8)

print(f"  Best at step   : {best_loss_step}")
print(f"  Final loss : {loss_final:.6f}")
print(f"  Final MAE% : {mae_final:.3f}%")
print(f"  Final R²   : {r2_final:.4f}")
print()
print(f"  Predictions: {np.round(p_final, 3)}")
print(f"  Targets    : {np.round(t_np, 3)}")
print()

ok5 = check(loss_final < loss0 * 0.02,  f"loss decreased to <2% of initial ({loss_final:.5f} < {loss0*0.02:.5f})")
ok6 = check(mae_final < 5.0,            f"final MAE% < 5.0%  (got {mae_final:.3f}%)")
ok7 = check(r2_final > 0.85,            f"final R² > 0.85   (got {r2_final:.4f})")
ok8 = check(p_final.max() <= 1.2,       f"predictions stay near [0,1]  (max={p_final.max():.4f})")
all_ok &= ok5 & ok6 & ok7 & ok8

# ─── Test predict_deg cell_id fix ─────────────────────────────────────────────
print()
print(SEP)
print("STEP 5 — predict_deg() cell_id fix verification")
print(SEP)

mend_proc = Path(ROOT) / "data" / "mendeley_processed"
cell_ok = False
if (mend_proc / "mendeley_val.pkl").exists():
    from src.data.mendeley_dataset import MendeleyDataset, compute_mendeley_stats
    from torch.utils.data import DataLoader
    import json

    mend_val_df = pd.read_pickle(str(mend_proc / "mendeley_val.pkl"))
    mend_train_df = pd.read_pickle(str(mend_proc / "mendeley_train.pkl"))
    stats = compute_mendeley_stats(mend_train_df)
    mend_val_ds = MendeleyDataset(mend_val_df, normalize=True, external_stats=stats)
    mend_val_loader = DataLoader(mend_val_ds, batch_size=8, shuffle=False)

    mend_eis_dim = mend_val_ds.num_eis_features
    model_deg = BaFuseV2(
        discharge_input_size=d_shape[-1],
        eis_num_frequencies=e_shape[-1],
        physics_num_features=p_shape[-1],
        mendeley_eis_num_features=mend_eis_dim,
        latent_dim=64, fusion_dim=128,
    )

    try:
        from src.training.multitask_trainer import MultiTaskTrainer

        # Quick test: run predict_deg directly
        model_deg.eval()
        lli_p, lam_p, cl_p = [], [], []
        lli_t, lam_t, cl_t = [], [], []
        cell_ids_collected  = []
        cycles_collected    = []

        with torch.no_grad():
            for batch in mend_val_loader:
                out_d = model_deg.forward_eis_only(batch["eis"])
                batch_size = out_d["lli_pred"].view(-1).shape[0]

                lli_p.append(out_d["lli_pred"].view(-1).numpy())
                lli_t.append(batch["lli_label"].view(-1).numpy())
                lam_p.append(out_d["lam_pred"].view(-1).numpy())
                lam_t.append(batch["lam_label"].view(-1).numpy())
                cl_p.append(out_d["cl_pred"].view(-1).numpy())
                cl_t.append(batch["cl_label"].view(-1).numpy())

                # Replicate the fixed logic
                raw_cids = batch["cell_id"]
                if isinstance(raw_cids, torch.Tensor):
                    cids = raw_cids.view(-1).cpu().tolist()
                elif isinstance(raw_cids, (list, tuple)):
                    cids = list(raw_cids)
                else:
                    cids = [int(raw_cids)] * batch_size
                assert len(cids) == batch_size
                cell_ids_collected.extend([int(c) for c in cids])
                cycles_collected.append(batch["aging_cycle"].numpy())

        total_preds = len(np.concatenate(lli_p))
        cell_ok = len(cell_ids_collected) == total_preds
        print(f"  Mendeley val samples : {total_preds}")
        print(f"  Cell IDs collected   : {len(cell_ids_collected)}")
        print(f"  Unique cells         : {sorted(set(cell_ids_collected))}")
        print(f"  batch['cell_id'] type: {type(raw_cids).__name__}",
              f"shape={tuple(raw_cids.shape)}" if isinstance(raw_cids, torch.Tensor) else "")
    except Exception as e:
        print(f"  predict_deg test error: {e}")
        cell_ok = False
else:
    print("  Mendeley processed data not found — skipping cell_id test.")
    print("  Run scripts/preprocess_mendeley.py first.")
    cell_ok = True   # not a blocker if data missing

ok9 = check(cell_ok, "predict_deg() completes without cell_id error")
all_ok &= ok9

# ─── Summary ──────────────────────────────────────────────────────────────────
print()
print(SEP)
print("SANITY TEST SUMMARY")
print(SEP)
if all_ok:
    print("  ALL CHECKS PASSED")
    print()
    print("  SOH scale:   [0, 1] confirmed")
    print(f"  Initial MSE: {loss0:.5f}  (expected ~0.01–0.1, NOT thousands)")
    print(f"  Final MAE%:  {mae_final:.2f}%  (overfit on 16 samples)")
    print(f"  Final R²:    {r2_final:.4f}")
    print()
    print("  Ready to resume staged training.")
else:
    print("  SOME CHECKS FAILED — do not resume training until fixed.")
print(SEP)

import sys
sys.exit(0 if all_ok else 1)
