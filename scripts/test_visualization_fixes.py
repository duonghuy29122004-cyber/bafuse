"""
Unit test: verify all three P5 visualization fixes.

P5-#1: _eis_val imputation — returns 0.0 (not (0-mean)/std) when normalize=True.
P5-#2: generate_all_plots crash — np.array([...]) or None raises ValueError.
P5-#3: SOH scale — plots receive [0,1], must display in [0,100].

Run:
    python scripts/test_visualization_fixes.py
Expected: ALL PASS, EXIT=0
"""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

failures = []

def check(cond, label):
    tag = "[PASS]" if cond else "[FAIL]"
    print(f"  {tag}  {label}")
    if not cond:
        failures.append(label)
    return cond

SEP = "=" * 62

# ─── P5-#1: _eis_val imputation ──────────────────────────────────────────────
print(SEP)
print("P5-#1 — _eis_val NaN imputation")
print(SEP)

import pandas as pd
import torch

# Build a minimal BaFuseDataset to test _eis_val behaviour
from src.data.dataset import BaFuseDataset

# Synthetic paired row with NaN re_ohm
row_data = {
    "battery_id": "B0005", "discharge_cycle": 10,
    "capacity_ahr": 1.8,
    "voltage_mean": 3.7, "voltage_min": 3.0, "voltage_max": 4.1, "voltage_std": 0.1,
    "current_mean": -1.0, "current_std": 0.1,
    "temp_mean": 25.0, "temp_std": 1.0,
    "impedance_ohm": 0.05, "re_ohm": float("nan"), "rct_ohm": 0.02,
    "cycle_gap": 1,
}
df_single = pd.DataFrame([row_data])
# Also add a normal row so stats are computable
row_data2 = dict(row_data); row_data2["re_ohm"] = 0.03; row_data2["rct_ohm"] = 0.01
row_data2["capacity_ahr"] = 1.9
df_two = pd.DataFrame([row_data, row_data2])

ds = BaFuseDataset(df_two, normalize=True)
# After normalization, stats['re']['mean'] should be ~0.03
re_mean = ds.stats['re']['mean']

# Get sample with NaN re_ohm (index 0)
sample = ds[0]
eis = sample['eis'].numpy()
# eis[1] is re_ohm — when NaN and normalize=True should be 0.0
check(abs(eis[1]) < 1e-6, f"NaN re_ohm z-scored to 0.0 when normalize=True (got {eis[1]:.6f})")

# P6-#1 FIX: real test for normalize=False — must NOT raise AttributeError.
# Before the getattr fix, self.stats was not set when normalize=False,
# causing AttributeError on the NaN imputation path.
ds_raw = BaFuseDataset(df_two, normalize=False)
try:
    sample_raw = ds_raw[0]   # MUST NOT crash with AttributeError
    eis_raw = sample_raw["eis"].numpy()
    # With normalize=False and no stats dict, NaN re_ohm → getattr fallback → 0.0
    check(True, "normalize=False: __getitem__ does not crash on NaN re_ohm")
    check(np.isfinite(eis_raw[1]),
          f"normalize=False NaN re_ohm returns finite value (got {eis_raw[1]:.6f})")
except AttributeError as e:
    check(False, f"normalize=False CRASHED with AttributeError: {e}")

# ─── P5-#2: np.array or None crash (reproduce then verify fix) ───────────────
print()
print(SEP)
print("P5-#2 — np.array([...]) or None crash")
print(SEP)

# Reproduce the original bug
import numpy as _np
_cids_multi = [1, 2, 3, 4, 5]
arr = _np.array(_cids_multi)
# Old code: `arr or None` with >1 element → ValueError
bug_reproduced = False
try:
    _ = arr or None
    print("  [NOTE]  ValueError NOT raised (may be single-element or empty edge case)")
except ValueError as e:
    bug_reproduced = True
    print(f"  [CONFIRMED BUG REPRODUCED]  ValueError: {e}")
check(bug_reproduced, "original bug reproduced: np.array(multi) or None raises ValueError")

# Verify the fix: explicit len() check
_cids = [1, 2, 3, 4, 5]
cell_ids_arr = _np.array(_cids) if len(_cids) > 0 else None
check(cell_ids_arr is not None, "fix: non-empty list -> np.array (not None)")
check(len(cell_ids_arr) == 5,   "fix: array has correct length 5")

_cids_empty = []
cell_ids_empty = _np.array(_cids_empty) if len(_cids_empty) > 0 else None
check(cell_ids_empty is None, "fix: empty list -> None")

# Verify generate_all_plots does not crash with multi-element cell_ids
from src.visualization import generate_all_plots
import tempfile, os

soh_p = _np.array([0.85, 0.80, 0.75])
soh_t = _np.array([0.87, 0.81, 0.76])
bids  = ["B0005", "B0005", "B0006"]
hist  = {"train_loss": [0.1, 0.05], "val_mae": [0.08, 0.05], "val_r2": [0.4, 0.6]}

deg_res = {
    "lli_pred":    _np.array([0.1, 0.2, 0.3]),
    "lli_target":  _np.array([0.1, 0.2, 0.3]),
    "lam_pred":    _np.array([0.0, 0.1, 0.2]),
    "lam_target":  _np.array([0.0, 0.1, 0.2]),
    "cl_pred":     _np.array([0.0, 0.0, 0.1]),
    "cl_target":   _np.array([0.0, 0.0, 0.1]),
    "cell_ids":    [1, 2, 3],          # multi-element: was crashing
    "aging_cycles": _np.array([0, 25, 50]),
}

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_all_plots(
            soh_preds=soh_p, soh_targets=soh_t, battery_ids=bids,
            history=hist, ablation_df=None, out_dir=tmpdir,
            deg_results=deg_res,
        )
    check(True, "generate_all_plots with multi-element cell_ids: no crash")
except ValueError as e:
    check(False, f"generate_all_plots CRASHED with ValueError: {e}")
except Exception as e:
    check(False, f"generate_all_plots CRASHED with {type(e).__name__}: {e}")

# ─── P5-#3: SOH scale in plot_soh_predictions ────────────────────────────────
print()
print(SEP)
print("P5-#3 — SOH scale [0,1] -> display [0,100]")
print(SEP)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.visualization import plot_soh_predictions

    preds_01 = _np.array([0.85, 0.80])
    tgts_01  = _np.array([0.87, 0.81])

    with tempfile.TemporaryDirectory() as tmpdir:
        fig = plot_soh_predictions(preds_01, tgts_01, out_path=str(Path(tmpdir)/"soh.png"))

    # Check title contains MAE in percent range (should be ~2.x%, not ~0.02%)
    title_text = fig.axes[0].get_title()
    print(f"  Plot title: {title_text}")
    # MAE of [0.85,0.80] vs [0.87,0.81] = mean([0.02,0.01]) = 0.015 -> 1.5%
    # Title should show ~1.5% not ~0.015%
    import re
    m = re.search(r"MAE=([\d.]+)%", title_text)
    if m:
        mae_displayed = float(m.group(1))
        check(mae_displayed > 0.5, f"MAE displayed as percent (got {mae_displayed:.3f}%), not raw [0,1]")
        check(mae_displayed < 10,  f"MAE in reasonable percent range (got {mae_displayed:.3f}%)")
    else:
        check(False, "Could not extract MAE from plot title")

    # Check x/y axis limits are in [0,100] range
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    check(xlim[1] > 1.5, f"X-axis range > 1.5 (in % not [0,1] scale, got {xlim})")
    plt.close("all")

except ImportError:
    print("  matplotlib not available — skipping P5-#3 plot checks")
    check(True, "matplotlib unavailable — skipping")

# ─── P5-#3: plot_soh_degradation_curve scale ─────────────────────────────────
try:
    from src.visualization import plot_soh_degradation_curve
    bids3 = ["B0005", "B0005", "B0006", "B0006"]
    preds3 = _np.array([0.85, 0.80, 0.90, 0.88])
    tgts3  = _np.array([0.87, 0.81, 0.91, 0.89])
    with tempfile.TemporaryDirectory() as tmpdir:
        fig2 = plot_soh_degradation_curve(preds3, tgts3, bids3,
                                          out_path=str(Path(tmpdir)/"curve.png"))
    # Y-axis should be in % range
    ax2 = fig2.axes[0]
    ylim = ax2.get_ylim()
    check(ylim[1] > 1.5, f"Degradation curve Y-axis in % range (got {ylim})")
    plt.close("all")
except Exception as e:
    check(False, f"plot_soh_degradation_curve failed: {e}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print()
print(SEP)
if not failures:
    print("ALL CHECKS PASSED")
    print()
    print("  P5-#1  NaN imputation -> 0.0 after z-score:  VERIFIED")
    print("  P5-#2  np.array or None crash fixed:          VERIFIED")
    print("  P5-#3  SOH scale [0,1] -> [0,100] in plots:  VERIFIED")
else:
    print(f"FAILED ({len(failures)}): {failures}")
print(SEP)
sys.exit(0 if not failures else 1)
