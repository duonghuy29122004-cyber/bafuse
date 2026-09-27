"""
Audit data splits: overlap, SOH distribution, mean-predictor baseline.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

proc = ROOT / "data" / "processed"
train = pd.read_pickle(str(proc / "train.pkl"))
val   = pd.read_pickle(str(proc / "val.pkl"))
test  = pd.read_pickle(str(proc / "test.pkl"))

SEP  = "=" * 62
PASS = "[PASS]"
FAIL = "[FAIL]"

# ─────────────────────────────────────────────────────────────────────────────
# Q1: Battery overlap
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("Q1 -- Battery overlap between splits")
print(SEP)

tr_b = set(train["battery_id"].unique())
va_b = set(val["battery_id"].unique())
te_b = set(test["battery_id"].unique())

tr_va = tr_b & va_b
tr_te = tr_b & te_b
va_te = va_b & te_b

print(f"  Train batteries  ({len(tr_b):2d}): {sorted(tr_b)}")
print(f"  Val   batteries  ({len(va_b):2d}): {sorted(va_b)}")
print(f"  Test  batteries  ({len(te_b):2d}): {sorted(te_b)}")
print()

def overlap_line(name, s):
    if s:
        print(f"  {FAIL}  {name}: {sorted(s)}")
    else:
        print(f"  {PASS}  {name}: EMPTY -- no overlap")

overlap_line("Train ∩ Val ", tr_va)
overlap_line("Train ∩ Test", tr_te)
overlap_line("Val   ∩ Test", va_te)
no_overlap = not tr_va and not tr_te and not va_te
print()
print(f"  --> Battery-level data leakage: {'NONE' if no_overlap else 'DETECTED'}")

# ─────────────────────────────────────────────────────────────────────────────
# Q2: SOH distribution comparison
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("Q2 -- SOH distribution (SOH = capacity_ahr / nominal_capacity per battery)")
print(SEP)

from src.data.dataset import get_nominal_capacity

def soh_series(df):
    return df.apply(
        lambda r: r["capacity_ahr"] / get_nominal_capacity(r["battery_id"]), axis=1
    )

tr_soh = soh_series(train)
va_soh = soh_series(val)
te_soh = soh_series(test)

def print_dist(label, soh, n_bins=5):
    s = soh.values
    print(f"\n  {label} (n={len(s)})")
    print(f"    mean   = {s.mean():.4f}  ({s.mean()*100:.2f}%)")
    print(f"    std    = {s.std():.4f}  ({s.std()*100:.2f}%)")
    print(f"    min    = {s.min():.4f}  ({s.min()*100:.2f}%)")
    print(f"    median = {np.median(s):.4f}  ({np.median(s)*100:.2f}%)")
    print(f"    max    = {s.max():.4f}  ({s.max()*100:.2f}%)")
    # histogram in text
    counts, edges = np.histogram(s, bins=5, range=(0, 1))
    print("    histogram (5 bins, SOH 0-100%):")
    for i, cnt in enumerate(counts):
        lo, hi = edges[i]*100, edges[i+1]*100
        bar = "#" * int(cnt / max(counts) * 20 + 0.5)
        print(f"      [{lo:5.1f}%-{hi:5.1f}%]  {bar:<20s}  {cnt}")

print_dist("Train", tr_soh)
print_dist("Val  ", va_soh)
print_dist("Test ", te_soh)

# KS test for distributional similarity
from scipy import stats as scipy_stats
ks_tr_te, p_tr_te = scipy_stats.ks_2samp(tr_soh.values, te_soh.values)
ks_tr_va, p_tr_va = scipy_stats.ks_2samp(tr_soh.values, va_soh.values)

print()
print("  KS test (p > 0.05 => distributions not significantly different):")
similar_te = "SIMILAR (p>0.05)" if p_tr_te > 0.05 else f"DIFFERENT (p={p_tr_te:.4f})"
similar_va = "SIMILAR (p>0.05)" if p_tr_va > 0.05 else f"DIFFERENT (p={p_tr_va:.4f})"
print(f"  Train vs Test : KS={ks_tr_te:.4f}  p={p_tr_te:.4f}  --> {similar_te}")
print(f"  Train vs Val  : KS={ks_tr_va:.4f}  p={p_tr_va:.4f}  --> {similar_va}")

# Mean shift (absolute)
print()
shift_te = abs(float(tr_soh.mean()) - float(te_soh.mean()))
shift_va = abs(float(tr_soh.mean()) - float(va_soh.mean()))
print(f"  Mean SOH shift  train->test : {shift_te:.4f}  ({shift_te*100:.2f} pp)")
print(f"  Mean SOH shift  train->val  : {shift_va:.4f}  ({shift_va*100:.2f} pp)")
if shift_te > 0.10:
    print(f"  WARNING: test mean differs from train by >{10} pp -- model may generalise poorly")

# ─────────────────────────────────────────────────────────────────────────────
# Q3: Mean predictor baseline
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("Q3 -- Baseline: constant mean predictor")
print(SEP)

train_mean = float(tr_soh.mean())
print(f"  Predictor = train mean SOH = {train_mean:.4f}  ({train_mean*100:.2f}%)")
print()
print("  This is the simplest non-trivial baseline.")
print("  Any model must beat it on ALL three metrics (MAE, RMSE, R^2 > 0).")

def baseline_metrics(label, soh):
    tgt = soh.values.astype(float)
    pred_const = np.full(len(tgt), train_mean)

    mae  = float(np.mean(np.abs(pred_const - tgt)))
    rmse = float(np.sqrt(np.mean((pred_const - tgt)**2)))
    ss_r = np.sum((tgt - pred_const)**2)
    ss_t = np.sum((tgt - tgt.mean())**2)
    r2   = float(1.0 - ss_r / (ss_t + 1e-10))
    # R² when predicting own mean (self-reference)
    ss_r2 = np.sum((tgt - tgt.mean())**2)
    r2_own = float(1.0 - ss_r2 / (ss_t + 1e-10))

    print(f"\n  {label}:")
    print(f"    predict train_mean ({train_mean*100:.2f}%) for all samples:")
    print(f"      MAE   = {mae:.4f}  ({mae*100:.2f}%)")
    print(f"      RMSE  = {rmse:.4f}  ({rmse*100:.2f}%)")
    print(f"      R^2   = {r2:.4f}  (negative = worse than predicting test/val mean)")
    print(f"    predict own split mean ({tgt.mean()*100:.2f}%):")
    print(f"      MAE   = {float(np.mean(np.abs(tgt.mean()-tgt))):.4f}  ({float(np.mean(np.abs(tgt.mean()-tgt)))*100:.2f}%)")
    print(f"      R^2   = 0.0000  (by definition)")
    return mae, rmse, r2

b_val  = baseline_metrics("Val ", va_soh)
b_test = baseline_metrics("Test", te_soh)

print()
print(SEP)
print("Summary")
print(SEP)
print(f"  No battery overlap:    {'YES' if no_overlap else 'NO'}")
print(f"  Train/Test KS p-val:   {p_tr_te:.4f}  ({similar_te})")
print(f"  Train/Test mean shift: {shift_te*100:.2f} percentage points")
print()
print("  Baselines to beat (predict train mean on test set):")
print(f"    MAE  < {b_test[0]*100:.2f}%")
print(f"    RMSE < {b_test[1]*100:.2f}%")
print(f"    R^2  > {b_test[2]:.4f}")
print()
print("  Current best val MAE from training: ~4.9%")
print("  (read from epoch 9 checkpoint -- compare to baseline above)")
print(SEP)
