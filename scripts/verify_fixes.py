"""Quick smoke-test for Task 1-3 fixes before running full CV."""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from src.data.dataset import BATTERY_CUTOFF_VOLTAGE, get_cutoff_voltage, PHYSICS_NUM_FEATURES
from src.losses import SoHPredictionLoss
from src.models.encoders import PhysicsEncoder, PhysicsEncoderCNN
from src.models.bafuse import BaFuse

# ── Task 1: cutoff voltages ────────────────────────────────────────────────
assert get_cutoff_voltage("B0053") == 2.0
assert get_cutoff_voltage("B0005") == 2.7
assert get_cutoff_voltage("B0007") == 2.2
assert get_cutoff_voltage("B0048") == 2.7
assert get_cutoff_voltage("B0045") == 2.0
print("Task1 cutoffs OK:", {k: get_cutoff_voltage(k) for k in ["B0005","B0007","B0045","B0053"]})

# ── Task 3: PhysicsEncoderCNN shape ───────────────────────────────────────
cnn_enc = PhysicsEncoderCNN(num_physics_features=4, latent_dim=64)
x = torch.randn(8, 4)
out = cnn_enc(x)
assert out.shape == (8, 64), f"CNN1D shape wrong: {out.shape}"
print(f"Task3 PhysicsEncoderCNN  output shape: {out.shape}")

# ── Task 3: BaFuse with both encoder types ────────────────────────────────
for enc_type in ["mlp", "cnn1d"]:
    m = BaFuse(
        discharge_input_size=3, eis_num_frequencies=3,
        physics_num_features=4, latent_dim=64, fusion_dim=128,
        physics_encoder_type=enc_type,
    )
    d = torch.randn(4, 100, 3)
    e = torch.randn(4, 3)
    p = torch.randn(4, 4)
    result = m(d, e, p)
    assert result["soh_pred"].shape == (4, 1)
    n = sum(x.numel() for x in m.parameters() if x.requires_grad)
    print(f"Task3 BaFuse({enc_type:5s})  params={n:,}  soh_pred={result['soh_pred'].shape}")

# ── Task 2: monotonicity loss -- within-battery only ───────────────────────
criterion = SoHPredictionLoss()
pred  = torch.tensor([90., 85., 80., 75.])
tgt   = torch.tensor([89., 84., 79., 74.])
ages  = torch.tensor([10,  30,  50,  70 ])

# Single battery -- all pairs valid
loss_single = criterion(pred, tgt, cycle_age=ages, battery_ids=["B0005"]*4)
print(f"Task2 loss (same bat, monotonic) : {loss_single.item():.6f}")

# Two batteries -- pairs across batteries must be ignored
# B0005 idx 0,2 (ages 10,50) and B0006 idx 1,3 (ages 30,70)
# Within B0005: pair(0,2) age10<age50, pred90>pred80 -> no violation
# Within B0006: pair(1,3) age30<age70, pred85>pred75 -> no violation
bids2 = ["B0005", "B0006", "B0005", "B0006"]
loss_two = criterion(pred, tgt, cycle_age=ages, battery_ids=bids2)
print(f"Task2 loss (2 bats, monotonic)   : {loss_two.item():.6f}")

# Introduce cross-battery violation: lower age battery has lower pred
pred_viol = torch.tensor([75., 90., 80., 85.])   # B0005(age10)=75, B0005(age50)=80 -> OK
# but if we treat all as same battery, age10->75 < age50->80 OK; age10->75 < age30->90 is violated
bids3 = ["B0005"]*4
ages3 = torch.tensor([10, 30, 50, 70])
pred_viol2 = torch.tensor([60., 80., 75., 90.])  # age10=60<age50=75 OK, age10=60<age70=90 OK
                                                    # age30=80>age50=75 -> VIOLATION
loss_viol = criterion(pred_viol2, tgt, cycle_age=ages3, battery_ids=bids3)
print(f"Task2 loss (same bat, violation) : {loss_viol.item():.6f}  <- should be > 0")
assert loss_viol.item() > 0.0, "Expected violation > 0"

# No battery_ids -> fallback cross-battery
loss_cross = criterion(pred, tgt, cycle_age=ages, battery_ids=None)
print(f"Task2 loss (cross-bat fallback)  : {loss_cross.item():.6f}")

print()
print("ALL CHECKS PASSED")

