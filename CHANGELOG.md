# BaFuse — Changelog & Results Summary

## Overview

Multimodal battery SoH estimation using NASA PCoE dataset.

---

## v2.1 — Bug-fix sprint (14 fixes, Priority 1–4)

**Smoke-test result:** 17/17 checks PASSED after all fixes.
**Key verified facts:** 100 % capacity from .mat field; SOH ∈ [0, 1];
discharge z-scored (mean ≈ −0.07); discharge shape (100, 3) always consistent;
EIS fully normalised; EISEncoder raises `ValueError` on wrong dim.

---

### Priority 1 — Ground-truth label bugs

#### P1-#1 · `src/data/parse_mat.py` · `parse_discharge_curve()`

**Bug:** Condition `if len(capacity_ts) >= min_len` was almost always `False`
because the NASA `.mat` Capacity field is a scalar/short array, not a
time-series. Result: every cycle silently fell through to coulomb-counting
(current integration), biasing the SOH label versus the official NASA value.

**Fix:**
- Changed condition to `if len(capacity_ts) > 0` — uses `max(abs(capacity_ts))`
  directly when any capacity value is present.
- Added `_cap_from_field` boolean column per row to track source.
- Added per-battery and dataset-wide capacity-source summary log line:
  `"636/636 cycles (100.0%) used .mat Capacity field"`.
- Coulomb-counting fallback now emits a `WARNING` per cycle so regressions
  are immediately visible.

**Verified:** 100 % of all parsed discharge cycles now use the official
`.mat` Capacity field (0 coulomb-counting fallbacks) on the full dataset.

---

#### P1-#2 · `src/data/dataset.py` + `src/data/pairing.py`

**Bug A (dataset.py):** `SOH = capacity_ahr / 2.0` hardcoded nominal
capacity of 2.0 Ah for all batteries. If any campaign used a different
cell, all SOH labels for that campaign would be wrong. Also, `fit_aging_prior`
used `capacity_ahr / 2.0 * 100` (double-scale error already fixed earlier
separately for the [0–100] bug).

**Fix A:**
- Added `BATTERY_NOMINAL_CAPACITY: Dict[str, float]` covering all 22 NASA
  PCoE campaign cells (B0005–B0056). All confirmed 2.0 Ah from README files.
- Added `get_nominal_capacity(battery_id)` helper with 2.0 Ah fallback.
- Replaced every `/2.0` hardcode in `__getitem__` and `fit_aging_prior`
  with `get_nominal_capacity(battery_id)`.

**Bug B (pairing.py):** `MIN_CAPACITY_AHR = 0.5` was an absolute threshold
(0.5 Ah) shared across all campaigns. With different nominal capacities this
would either pass faulty cycles or reject healthy ones.

**Fix B:**
- Added `_MIN_CAPACITY_FRACTION = 0.25` (25 % of nominal capacity per battery).
- Filter now computes `min_cap = 0.25 × get_nominal_capacity(battery_id)` per row.
- Replaced slow `apply(lambda)` row-wise loop with vectorised
  `pd.MultiIndex.isin()` (also fixes P4-#14 simultaneously).

---

### Priority 2 — Normalisation / scale bugs

#### P2-#3 · `src/data/dataset.py` · `BaFuseDataset.__getitem__()`

**Bug:** When `discharge_data_df` was provided, `ts_raw` (raw V/I/T
time-series) was returned **unnormalised** directly from `.mat` values,
while the aggregate fallback (5-feature vector) was z-scored via
`self._normalize()`. Two code paths returned tensors on different scales.

**Fix:** Added per-channel z-score of `ts_raw` using `self.stats` keys
`'voltage'`, `'current'`, `'temperature'` before resampling. Normalisation
happens on the raw array; resampling follows. Both paths now on the same
scale. Documented in docstring with explicit "normalise before resample" note.

---

#### P2-#4 · `src/data/dataset.py` · `_compute_normalization_stats()` + `__getitem__()`

**Bug:** `re_ohm` and `rct_ohm` were returned as raw ohm values (up to tens
of ohms) while `impedance_ohm` was z-scored. Three EIS features on three
different scales fed into the same EIS encoder.

**Fix:**
- Added `'re'` and `'rct'` keys to `_compute_normalization_stats()` computed
  from `re_ohm` and `rct_ohm` in the training set.
- Added `_eis_val(col, stat_key)` helper in `__getitem__` that handles `NaN`
  → 0.0 and applies `self._normalize()`. All three EIS features now z-scored.

---

#### P2-#5 · `src/data/dataset.py` · `BaFuseDataset.__init__()`

**Bug:** If `discharge_data_df` was provided but a sample's
`(battery_id, discharge_cycle)` was absent, `__getitem__` silently fell
back to the 5-feature aggregate vector `(5,)` while other samples returned
`(TARGET_SEQ_LEN, 3)`. Mixed shapes in the same `Dataset` crash
`DataLoader.collate_fn` at a random later batch.

**Fix:** Added pre-filter in `__init__`: when `discharge_data_df` is given,
`data_df` rows without a matching time-series are dropped at construction
time with a logged `WARNING` showing the count. All remaining samples are
guaranteed to produce `(TARGET_SEQ_LEN, 3)` tensors.

---

### Priority 3 — Training logic bugs

#### P3-#6 · `src/training/multitask_trainer.py` · `_train_epoch()`

**Bug:** Docstring described "L_total = λ_soh·L_soh + λ_deg·L_deg combined
loss" but code ran two **sequential** loops — full NASA loader then full
Mendeley loader — each with its own `optimizer.step()`. The shared EIS
encoder received competing gradients from two separate tasks without any
balancing, leading to oscillating validation metrics.

**Fix:** Rewrote `_train_epoch` for `"staged"` / `"joint"` modes:
- Uses `itertools.cycle` to interleave NASA and Mendeley batches step-by-step
  (shorter loader repeats).
- Each step computes both losses, sums `λ_soh·L_soh + λ_deg·L_deg`, calls
  `loss.backward()` and `optimizer.step()` **once**.
- `"nasa_only"` mode is unchanged.
- Docstring updated to accurately describe each mode's behaviour.

---

#### P3-#7 · `src/training/degradation_trainer.py` · `_freeze_non_deg_params()`

**Bug:** `"eis_encoder" in name` (substring match) accidentally matched both
`mendeley_eis_encoder.*` (intended) **and** `eis_encoder.*` (the NASA encoder,
unintended). With the BaFuseV2 architecture that has two separate encoders,
this kept the NASA EIS encoder trainable during Stage-1 Mendeley pre-training.

**Fix:** Replaced with exact `startswith()` prefix match:
```python
name.startswith("mendeley_eis_encoder.") or
name.startswith("degradation_head_eis_only.")
```
Only the Mendeley encoder and its head are now trainable in Stage-1.

---

#### P3-#8 · `src/models/encoders.py` · `EISEncoder`

**Bug:** `_mlp_fallback()` created an `nn.Linear` inside `forward()` using
lazy init (`if not hasattr(self, '_fallback_proj')`). Any layer created after
`optimizer` is built is invisible to the optimizer — it is never trained
(random weights forever). This was dead code with the current config but a
silent trap for future changes.

**Fix:** Removed `_mlp_fallback()` entirely. Both code paths now fail loudly:
- MLP path: raises `ValueError` if `x.shape[-1] != num_frequencies`.
- CNN path: raises `ValueError` if input is not 3-D.
All needed `nn.Linear` layers are declared in `__init__` and are therefore
always visible to the optimizer.

---

#### P3-#9 · `src/data/split.py` · `split_by_battery()`

**Bug:** The first `train_test_split` (train vs temp) stratified by
`capacity_bin`, but the second split (val vs test from `temp_batteries`) did
**not** stratify. With only a few batteries per split the val and test sets
could end up with very different SOH distributions depending on random seed.

**Fix:** Added stratification to the second split using bins computed
only on `temp_batteries`. Both splits now fail gracefully with a `WARNING`
and fall back to unstratified splitting when there are too few batteries for
the minimum stratum size (< 2 members per bin) — verified with the 4-battery
smoke-test subset.

---

### Priority 4 — Data quality / documentation

#### P4-#10 · `src/data/parse_mat.py` · `_safe_impedance_scalar()`

Added docstring note classifying `Re_median` and `Rct_proxy` as
**engineered proxy features**, not true impedance spectroscopy quantities
(true Re/Rct require high-frequency Nyquist intercept / semicircle fitting).
Consistent with the "model-derived" convention used for Mendeley LLI/LAM/CL.

---

#### P4-#11 · `src/data/mendeley_dataset.py` · `build_mendeley_dataframe()`

Added per-cell **match-rate log** after the EIS–Circuit_parameter inner join:
```
Cell 1: inner join matched 12/12 cycles (100%)
```
Emits `WARNING` when match rate < 90 % to detect silent partial-join data
loss from `aging_cycle` numbering mismatches between the two Excel files.

---

#### P4-#12 · `src/data/mendeley_dataset.py`

Added `leave_one_cell_out_splits(df)` function implementing **8-fold
Leave-One-Cell-Out cross-validation** for the Mendeley dataset.
With only 8 cells a single fixed split is too thin for reliable evaluation.
Returns a list of `(train_df, test_df)` tuples, one per fold.

---

#### P4-#13 · `src/data/mendeley_dataset.py` · `COLUMN_MAP`

Removed 4 duplicate entries (`"rmse_real"`, `"rmse_imag"` × 2, and the
extra `"soc"` that was already covered by `"soc(%)"` mapping). Dict keys
are now unique.

---

#### P4-#14 · `src/data/pairing.py` · `pair_discharge_eis()`

Replaced the O(n)-Python-level `apply(lambda r: ... not in bad_set, axis=1)`
row filter with a vectorised `pd.MultiIndex.isin()` operation (implemented
as part of the P1-#2 fix). Typical speedup: 5–20× on large DataFrames.

---

### Files changed

| File | Fixes applied |
|------|---------------|
| `src/data/parse_mat.py` | P1-#1, P4-#10 |
| `src/data/dataset.py` | P1-#2A, P2-#3, P2-#4, P2-#5 |
| `src/data/pairing.py` | P1-#2B, P4-#14 |
| `src/data/split.py` | P3-#9 |
| `src/data/mendeley_dataset.py` | P4-#11, P4-#12, P4-#13 |
| `src/models/encoders.py` | P3-#8 |
| `src/training/multitask_trainer.py` | P3-#6 |
| `src/training/degradation_trainer.py` | P3-#7 |
| `scripts/smoke_test.py` | new — regression test for all P1/P2/P3 fixes |

---

### Smoke-test output (abbreviated)

```
CAPACITY SOURCE SUMMARY — 636/636 discharge cycles (100.0%) used .mat Capacity field
discharge shape : (100, 3)
soh_label       : 0.9014   (in [0,1])
discharge mean (z-scored): -0.0710
Batch discharge : (8, 100, 3)
Batch SOH max   : 0.9238   (<= 1.0)
EISEncoder raises ValueError on wrong input dim: PASS
SMOKE TEST PASSED — all checks OK
```

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
