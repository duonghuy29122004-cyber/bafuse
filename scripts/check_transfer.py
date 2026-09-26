"""
Inspect which parameter was skipped during Mendeley->NASA BaFuseV2 transfer
and verify dimensional compatibility.
"""
import sys, logging
import torch
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")
logging.basicConfig(level=logging.WARNING)

from src.models.bafuse_v2 import BaFuseV2

stage1_ckpt = Path(ROOT) / "experiments" / "degradation" / "best_degradation.pth"
if not stage1_ckpt.exists():
    print("Stage-1 checkpoint not found — run train_degradation.py first.")
    sys.exit(0)

ckpt = torch.load(str(stage1_ckpt), map_location="cpu", weights_only=False)
v1_state = ckpt.get("model_state_dict", ckpt)

# Build v2 model matching NASA dims
nasa_eis_dim  = 3
mend_eis_dim  = 8

model_v2 = BaFuseV2(
    discharge_input_size=3,
    eis_num_frequencies=nasa_eis_dim,
    physics_num_features=4,
    latent_dim=64, fusion_dim=128,
    mendeley_eis_num_features=mend_eis_dim,
)
v2_state = model_v2.state_dict()

SEP = "=" * 62
print(SEP)
print("Transfer analysis: Stage-1 checkpoint -> BaFuseV2")
print(SEP)
print(f"  Stage-1 checkpoint keys: {len(v1_state)}")
print(f"  BaFuseV2 state keys:     {len(v2_state)}")
print()

transferred, skipped_mismatch, skipped_missing = [], [], []

for key, val in v1_state.items():
    if key in v2_state:
        if v2_state[key].shape == val.shape:
            transferred.append((key, tuple(val.shape)))
        else:
            skipped_mismatch.append((key, tuple(val.shape), tuple(v2_state[key].shape)))
    else:
        skipped_missing.append((key, tuple(val.shape)))

# Keys in v2 not in v1 (newly added)
new_keys = [k for k in v2_state if k not in v1_state]

print(f"  Transferred:       {len(transferred)}")
print(f"  Skipped (shape):   {len(skipped_mismatch)}")
print(f"  Skipped (missing): {len(skipped_missing)}")
print(f"  New (v2 only):     {len(new_keys)}")

if skipped_mismatch:
    print()
    print("-- Shape-mismatched parameters (SKIPPED) --")
    for name, src_shape, tgt_shape in skipped_mismatch:
        print(f"  {name}")
        print(f"    source shape: {src_shape}  (Stage-1)")
        print(f"    target shape: {tgt_shape}  (BaFuseV2)")
        # Determine which encoder these belong to
        if "eis_encoder" in name:
            print(f"    -> EIS encoder weight. Stage-1 trained on Mendeley (8D input).")
            print(f"       BaFuseV2 NASA eis_encoder uses 3D input. NOT compatible.")
            print(f"       Correctly skipped — NASA eis_encoder stays randomly initialised.")

if skipped_missing:
    print()
    print("-- Parameters in Stage-1 not in BaFuseV2 (MISSING) --")
    for name, shape in skipped_missing:
        print(f"  {name}  shape={shape}")

print()
print("-- New BaFuseV2 parameters (not in Stage-1) --")
for k in new_keys[:20]:
    print(f"  {k}  shape={tuple(v2_state[k].shape)}")
if len(new_keys) > 20:
    print(f"  ... and {len(new_keys)-20} more")

print()
print(SEP)
print("TRANSFER VERDICT")
print(SEP)

# Key insight: mendeley_eis_encoder in v2 has input_size=8 (matches Stage-1 eis_encoder)
# Find if Stage-1 eis_encoder weights can be loaded into mendeley_eis_encoder
print("  Stage-1 trained:  mendeley_eis_encoder (8D EIS input)")
print("  BaFuseV2 has:     eis_encoder (3D NASA EIS)")
print("                    mendeley_eis_encoder (8D Mendeley EIS)")
print()

# Check mendeley_eis_encoder compatibility
mend_keys_in_v1 = {k: v for k, v in v1_state.items() if "eis_encoder" in k}
mend_keys_in_v2 = {k: v for k, v in v2_state.items() if "mendeley_eis_encoder" in k}

# Map Stage-1 eis_encoder -> v2 mendeley_eis_encoder
print("  Checking eis_encoder -> mendeley_eis_encoder mapping:")
compatible = 0
incompatible = 0
for k, v in mend_keys_in_v1.items():
    mapped_k = k.replace("eis_encoder", "mendeley_eis_encoder")
    if mapped_k in v2_state:
        if v2_state[mapped_k].shape == v.shape:
            compatible += 1
            print(f"    [OK] {k} ({tuple(v.shape)}) -> {mapped_k}")
        else:
            incompatible += 1
            print(f"    [!!] {k} {tuple(v.shape)} -> {mapped_k} {tuple(v2_state[mapped_k].shape)}")
    else:
        print(f"    [?]  {mapped_k} not found in v2")

print()
if incompatible == 0 and compatible > 0:
    print(f"  -> Stage-1 eis_encoder weights CAN be loaded into mendeley_eis_encoder ({compatible} layers).")
    print("  -> The 3D NASA eis_encoder stays randomly initialised (correct).")
    print("  -> Transfer is dimensionally valid WITH the separate encoder design.")
else:
    print(f"  -> {incompatible} incompatible layers found.")
print(SEP)
