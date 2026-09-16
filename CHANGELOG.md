# BaFuse — Changelog & Results Summary

## Overview

Multimodal battery SoH estimation using NASA PCoE dataset.  
Fuses: Discharge curve (LSTM) + EIS (MLP) + Physics-informed features → CrossAttention → SoH prediction.

---

## Phase 1 — Initial Implementation (skeleton → runnable)

**Files created/implemented:**

| File | What was done |
|---|---|
| `src/models/encoders.py` | LSTM (Discharge), MLP (EIS, Physics) |
| `src/models/fusion.py` | CrossAttentionFusion, WeightedFusion, ConcatFusion |
| `src/models/bafuse.py` | Unified model, all 3 fusion methods |
| `src/losses.py` | SoHPredictionLoss: MSE + physics bound penalty + monotonicity |
| `src/train.py` | Full training loop, early stopping, cosine LR |
| `src/evaluate.py` | Test metrics, modality ablation, MC Dropout uncertainty |
| `src/data/__init__.py` | Lazy-import torch to avoid DLL load on import |
| `run_pipeline.py` | End-to-end PyTorch pipeline |
| `run_sklearn_pipeline.py` | sklearn baseline (RF, GBM, MLP) |

**Blocker resolved:** Windows Smart App Control blocked torch DLLs.  
Fix: disabled Smart App Control in Windows Security → torch 2.14.0+cpu works.

---

## Phase 2 — Data Pipeline Bug Fixes (5 bugs)

### BUG 1 — Impedance computed incorrectly (`parse_mat.py`)
- **Problem:** `np.nanmedian()` on complex array silently discards imaginary part (ComplexWarning). No outlier filter → impedance range was `-1161 to 1.47e14 Ω` (physically impossible).
- **Fix:** New `_safe_impedance_scalar()`: compute `|Z|` magnitude first, hard filter `0–1000 Ω`, percentile clip `1st–99th`, split `re_ohm` / `rct_ohm` from real/imag parts. Re values clipped to ≥ 0.

### BUG 2 — Low-capacity faulty cycles not filtered (`pairing.py`)
- **Problem:** `capacity_ahr` as low as 0.0001 Ahr → SoH label ~0% from device faults, not real degradation.
- **Fix:** Filter cycles with `capacity_ahr < 0.5 Ahr` before pairing. **185 faulty cycles removed across 13 batteries.**

### BUG 3 — Discharge–EIS pairing too loose (`pairing.py`)
- **Problem:** `cycle_gap` up to 39 (EIS measured 39 cycles after discharge) → label mismatch.
- **Fix:** Hard enforce `max_cycle_gap=10`. No unlimited fallback — unpaired cycles are dropped. **72 cycles dropped.**

### BUG 4 — Duplicate battery parsing (`parse_mat.py`)
- **Problem:** B0025–B0028 exist in 2 subdirectories → parsed twice → log reported 835,422 records (inflated).
- **Fix:** `seen_ids` set in `parse_all_mat_files()` — skip if battery already parsed.

### BUG 5 — Discharge sequence too long → slow training (`dataset.py`)
- **Problem:** Raw sequence length up to 500 steps → LSTM takes ~4 min/epoch on CPU.
- **Fix:** Resample every discharge curve to 100 points via `np.interp`. **~6× speedup.**

**Stats before vs after:**

| Metric | Before | After |
|---|---|---|
| Discharge records | 835,422 (with dupes) | 770,070 (clean) |
| `impedance_range` | -1161 – 1.47e14 Ω | **0.15 – 648 Ω** |
| `re_ohm` range | had negatives | **0.0 – 85.9 Ω** |
| `capacity_range` min | 0.0001 Ahr | **0.52 Ahr** |
| `cycle_gap` max | 39 | **10** |
| discharge `seq_len` | 500 | **100** |

---

## Phase 3 — Critical Data Leakage & Normalization Fixes

### BUG 6 — Data leakage: physics feature = linear function of label (`dataset.py`)
- **Problem:** `capacity_fade = 2.0 - capacity_ahr` was in physics features. Since `soh_label = (capacity_ahr / 2.0) * 100`, these two are **perfectly linearly correlated (r ≈ -1.0)**. Model only needed to learn a scalar multiply — no real learning from discharge/EIS needed.
- **Evidence:** Ablation showed physics=79.1% contribution, EIS=0.7% — impossible if fusion was real.
- **Fix:** Removed `capacity_fade`. Replaced with **empirical aging prior** fitted from train population: power-law `fade = A × (age/N_ref)^b` using log-log regression on train data. Uses only `cycle_age` as input, NOT per-sample `capacity_ahr` → no leakage.

### BUG 7 — Normalization skew: val/test use their own stats (`dataset.py`)
- **Problem:** Each `BaFuseDataset` computed mean/std from its own `data_df`. Same physical value normalized differently in train vs val vs test → distribution shift at inference.
- **Fix:** `BaFuseDataset` accepts `external_stats` param. `create_dataloaders()` computes stats once from train set, passes to val/test via `external_stats=train_dataset.stats`.

**Physics features after fix:**

| Feature | Source | Leaks label? |
|---|---|---|
| `cycle_age_norm` | `discharge_cycle / N_ref` | No |
| `empirical_fade_prior` | Power-law fit on train population | No |
| `voltage_droop` | `(voltage_min - 2.7) / 1.5` | No |
| `impedance_rise` | `(impedance_ohm - train_mean) / train_std` | No |

---

## Phase 4 — Model & Training Improvements

### DischargeEncoder capacity increase (`encoders.py`)
- LSTM: `hidden_size` 128 → **256**, `num_layers` 2 → **3**
- Discharge now carries ~70% of signal → needs more capacity

### LR schedule fix (`train.py`)
- **Problem:** Cosine LR started high → loss spikes at epoch 1-3 (MAE=841 → 500 → 146).
- **Fix:** Linear warmup (10 epochs, LR ramps 0 → target) then cosine decay via `LambdaLR`. LR logged every epoch.

### Early stopping criterion (`train.py`)
- Changed from tracking `val_loss` to tracking `val_MAE` — less noisy on small val set (103 samples / 4 batches).

### Config (`configs/config.yaml`)
- `learning_rate`: 0.001 → **0.0005**
- `warmup_epochs`: 5 → **10**
- Removed Vietnamese comments (caused cp1252 UnicodeDecodeError on Windows)

---

## Phase 5 — Evaluation Infrastructure

### Battery-group k-fold CV (`scripts/cross_validate.py`)
- Stratified 5-fold split by mean battery SoH
- Reports MAE ± std across folds
- Identifies short-trajectory batteries (`< 20 cycles`)

### EIS Investigation (`scripts/investigate_eis.py`)
EIS contribution findings:

| Check | Result |
|---|---|
| NaN in re_ohm/rct_ohm | **0%** — all EIS fully parsed |
| cycle_gap distribution | 98% gap=1, max=2 — **very clean pairing** |
| impedance ↔ SoH correlation | **-0.37** (moderate) |
| EIS-only sklearn MAE | 9.56% |
| Discharge-only sklearn MAE | 7.21% |

**Conclusion:** EIS has real signal (r=-0.37) but discharge curve is intrinsically stronger for this dataset. Impedance variance per battery is small (~0.03 Ω) compared to SoH range → attention mechanism assigns most weight to discharge. This is a **valid research finding**, not a model bug.

### Data validation script (`scripts/validate_pipeline.py`)
- Runs all 5 physical plausibility assertions after each pipeline run.

---

## Final Model Results

### Single-split test set (60/20/20 battery-level)

| Metric | Buggy baseline | After leakage fix | **Final (all fixes)** |
|---|---|---|---|
| Test MAE | ~9.0% | 11.44% | **6.97%** |
| Test RMSE | ~10.5% | 13.22% | **10.82%** |
| Test R² | ~0.87 (inflated) | 0.297 | **0.529** |
| Test MAPE | — | 18.3% | **13.68%** |

> MAE improved from 11.44% → 6.97% after fixing LR warmup and increasing discharge encoder capacity (not by re-adding leakage).

### Modality Contributions (ablation — zero-out test)

| Modality | Buggy | Fixed v1 | **Final** |
|---|---|---|---|
| Discharge | ~20% | ~100% | **70.4%** |
| Physics | **79.1% ← leakage** | ~0% | **27.1%** |
| EIS | 0.7% | ~0% | **2.5%** |

### Per-battery test results

| Battery | N cycles (test) | MAE | R² | Notes |
|---|---|---|---|---|
| B0005 | 142 | **3.45%** | **0.811** | Best — long trajectory |
| B0048 | 20 | 5.44% | 0.268 | OK |
| B0053 | 13 | 6.95% | -21.1 | Outlier: SoH range only 4.5%, voltage_min ~1.97V vs typical ~2.6V |
| B0026 | 9 | 11.95% | -2.96 | Too few cycles |
| B0039 | 13 | 16.28% | 0.016 | Short trajectory |
| B0045 | 20 | 19.71% | -10.1 | Low SoH range in test fold |
| B0051 | 6 | 25.22% | -0.34 | Only 6 cycles — insufficient |

---

## Cleanup

**Removed (debug/redundant files):**
- `debug_mat.py`, `debug_mat2.py`, `debug_mat3.py`
- `debug_parse.py`, `debug_parse2.py`, `debug_parse3.py`
- `check_cycle_type.py`, `test_imported.py`
- `process_data.py` (superseded by `run_pipeline.py`)
- `DATA_GUIDE.md` (content merged into README)

**Added:**
- `scripts/inspect_data.py` — unified .mat inspection (replaces all debug files)
- `scripts/test_pipeline.py` — integration test
- `scripts/validate_pipeline.py` — data quality assertions
- `scripts/cross_validate.py` — k-fold CV
- `scripts/investigate_eis.py` — EIS signal analysis
- `scripts/analyze_b0053.py` — outlier analysis

---

## How to Run

```powershell
# Full pipeline (parse → train → evaluate)
.\bafuse_env\Scripts\python.exe run_pipeline.py --epochs 100 --patience 20 --batch_size 32

# k-fold cross-validation (recommended for reporting)
.\bafuse_env\Scripts\python.exe scripts\cross_validate.py --folds 5 --epochs 30

# EIS investigation
.\bafuse_env\Scripts\python.exe scripts\investigate_eis.py

# Data quality check
.\bafuse_env\Scripts\python.exe scripts\validate_pipeline.py

# Streamlit dashboard
.\bafuse_env\Scripts\python.exe -m streamlit run app\dashboard.py
```
