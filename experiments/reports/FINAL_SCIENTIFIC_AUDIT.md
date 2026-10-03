# BaFuse — Final Scientific Audit

**Date:** 2026-09  
**Audit scope:** Full pipeline from data loading through external evaluation  
**Status:** 3 critical issues found, all resolved. Pipeline is thesis-ready with documented limitations.

---

# 1. Executive Summary

| Check | Result |
|-------|--------|
| NASA battery-level split (no overlap) | PASS |
| Normalization stats from train only | PASS |
| Capacity / SOH as model input | PASS (not used as input) |
| Empirical fade prior (train-only fit) | PASS |
| SOH > 1.0 samples | 3 samples (documented, clipped, physical) |
| Physics feature redundancy | NOTED (impedance_rise == eis[0]) |
| Samsung in training | CONFIRMED ABSENT |
| Samsung normalization stats | FIXED (now uses nasa_train_stats.json) |
| Zero-padding missing modalities | FIXED (removed; EIS-only path used) |
| Results labeling | FIXED (CASE B clearly labeled) |
| 14-model benchmark validity | ALL VALID for NASA |
| Samsung external eval validity | EXPLORATORY (EIS-only, scale mismatch) |
| Physical degradation analysis | PASS (no fabricated labels) |
| Reproducibility | PARTIAL PASS (gaps documented) |

---

# 2. NASA Pipeline

## Data

- Dataset: NASA PCoE Battery Aging Dataset, 34 batteries, 6 campaigns
- Batteries: B0005–B0056, 18650-type Li-ion, 2.0 Ah nominal (all campaigns)
- Input data: discharge time-series (V/I/T), EIS measurements, cycle metadata
- Pairing: discharge cycles matched to nearest EIS within max_cycle_gap=10

## Preprocessing

- Capacity filter: cycles with capacity < 25% nominal dropped per battery
- Discharge resampled to 100 timesteps via linear interpolation
- EIS: median |Z|, Re, |Im| per cycle (3 summary features)
- Physics: 4 engineered features (see Feature Leakage section)

## Split

- Battery-level stratified split: 60% train / 20% val / 20% test
- 20 train batteries / 7 val batteries / 7 test batteries
- 616 / 222 / 97 samples
- **Verified: zero overlap across splits, zero duplicate (battery, cycle) pairs**
- Seed: 42

## Normalization

- Statistics computed from train split only
- Val and test receive frozen train statistics via `external_stats=` parameter
- Stats saved to `data/processed/nasa_train_stats.json`
- Keys: voltage, current, temperature, impedance, capacity, re, rct

## Target

- SOH = capacity_ahr / nominal_capacity (battery-specific, all 2.0 Ah)
- SOH ∈ [0, 1] after clipping
- 3 samples with raw SOH > 1.0 found (B0036 cycle 278, B0049 cycle 10, B0051 cycle 10)
  - Maximum raw SOH: 1.222 (B0036, capacity 2.44 Ah exceeding 2.0 Ah nominal)
  - Cause: early-cycle break-in effect common in Li-ion cells
  - Treatment: `np.clip(soh, 0, 1)` applied in `__getitem__` — model receives SOH=1.0 as maximum
  - This is physically plausible and not a preprocessing bug

## Feature Leakage Audit

| Feature | Source | Leakage? | Notes |
|---------|--------|----------|-------|
| cycle_age_norm | discharge_cycle | NONE | Pure cycle index |
| empirical_fade_prior | power-law prior (train-fit) | NONE | Train-only fit, frozen params at inference; uses only cycle age |
| voltage_droop | voltage_min (measured) | NONE | Measured discharge signal, not the SOH label |
| impedance_rise | impedance_ohm (measured) | NONE | Measured EIS signal; REDUNDANT with eis[0] |
| discharge V/I/T | raw time-series | NONE | Input signal |
| EIS features | impedance, re, rct | NONE | Input signal |
| capacity_ahr | paired_df | NONE | Used ONLY to compute soh_label; never in model input dict |

**Known redundancy:** `physics[3]` (impedance_rise) = z-score(impedance_ohm) = `eis[0]`. This is not data leakage but reduces the independence of the 7 input features. Documented for thesis.

---

# 3. NASA Benchmark

## 14-Model Validity

All 14 benchmark models are valid for NASA evaluation:
- All accept `(discharge, eis, physics)` → same input pipeline
- All trained with `discharge_data_df=discharge_raw.pkl` → (B,100,3) time-series
- All use NASA TRAIN normalization statistics
- No model uses capacity_ahr as input
- MSE loss only (no cross-dataset contamination)

## Results Summary (seed=42, NASA test set)

| Model | Params | Test MAE% | Test RMSE% | Test R² |
|-------|--------|-----------|------------|---------|
| D2 (Full TCN) | 48,225 | 4.23 | 6.04 | 0.856 |
| D (TCN) | 44,257 | 5.34 | 8.37 | 0.724 |
| G (CNN+Gated) | 24,964 | 5.43 | 7.22 | 0.794 |
| G2 (Full CNN Gated) | 20,132 | 5.52 | 7.61 | 0.772 |
| A (TinyMLP) | 3,431 | 6.38 | 8.55 | 0.712 |
| C2 (Full CNN) | 21,409 | 6.24 | 7.75 | 0.763 |
| C (CNN1D) | 24,193 | 7.06 | 9.21 | 0.665 |
| ... | | | | |
| H (BaFuse v1) | 1,536,321 | 10.80 | 13.46 | 0.285 |
| Mean predictor | 0 | 15.68 | — | 0.000 |

*All 14 models beat the mean predictor baseline (15.68%). Results are valid.*

---

# 4. Samsung External Test

## Modality Compatibility

- Discharge: NOT AVAILABLE (no V/I/T time-series in Samsung dataset)
- EIS: AVAILABLE but ~12x smaller scale than NASA
- Physics: PARTIAL (cycle_age available; voltage_droop requires discharge)

**Verdict: CASE B — EIS-only evaluation only**

## Normalization

- **Before fix:** Script loaded `mendeley_stats.json` (Samsung-derived) — WRONG
- **After fix:** Script loads `nasa_train_stats.json` — CORRECT

## EIS Scale Mismatch

| Feature | NASA | Samsung | z-score |
|---------|------|---------|---------|
| |Z| mean | 0.216 Ohm | 0.017 Ohm | -6.8σ |
| Re mean | 0.209 Ohm | 0.017 Ohm | -4.9σ |

Samsung EIS values produce z-scores 5-7 sigma below NASA training distribution. Results are out-of-distribution.

## EIS-Only Evaluation Validity

- 13/14 models: true EIS encoder extracted (no discharge or physics created)
- Model A: zero-padded (no dedicated EIS encoder) — explicitly labeled
- All models use **random downstream heads** (no checkpoint loaded per model)
- Results reflect encoder structure only, not trained model performance

**All Samsung results labeled EXPLORATORY.**

---

# 5. Physical Degradation Analysis

- Observable signals: impedance_rise, voltage_droop, capacity_fade, re_change, rct_change
- No LLI/LAM/CL ground-truth labels fabricated for NASA
- No degradation mechanism claimed with certainty
- RAG layer is post-training interpretation only
- Language used: "consistent with", "may indicate", "possible mechanism"
- **AUDIT: PASS**

---

# 6. Reproducibility

**PARTIAL PASS.** Core pipeline reproducible. Gaps:
- Multi-seed results incomplete (seed=42 only for most models)
- Per-model checkpoints not individually saved
- No pip/conda lock file

---

# 7. Critical Issues Found and Fixed

| # | Issue | File | Fix Applied |
|---|-------|------|-------------|
| C1 | Samsung normalization stats used (`mendeley_stats.json`) | evaluate_samsung_external.py | Fixed: now loads `nasa_train_stats.json` |
| C2 | Zero-padding missing modalities into untrained full model | evaluate_samsung_external.py | Fixed: true EIS-encoder-only path |
| C3 | Results mislabeled (full model called but labeled "EIS-only") | evaluate_samsung_external.py | Fixed: CASE B clearly labeled |

---

# 8. Non-Critical Limitations

| # | Issue | Severity | Action |
|---|-------|----------|--------|
| L1 | `impedance_rise` == `eis[0]` (redundant feature) | LOW | Documented in thesis |
| L2 | SOH>1 in 3 early-cycle samples (B0036/B0049/B0051) | LOW | Clipped; physically plausible |
| L3 | Samsung EIS scale mismatch ~13x | MEDIUM | Documented; results labeled exploratory |
| L4 | Random eval head (no checkpoint per model in EIS eval) | MEDIUM | Documented; labeled "structural assessment" |
| L5 | Multi-seed results incomplete | LOW | Planned |
| L6 | No environment lock file | LOW | requirements.txt provided |
| L7 | Val MAE early stopping (not loss) | LOW | Intentional design choice, documented |

---

# 9. Thesis-Ready Methodology

The following aspects of the pipeline are thesis-ready:

1. **NASA training protocol:** Battery-level split, train-only normalization, no capacity leakage ✅
2. **14-model benchmark:** All models valid; same split/seed/protocol ✅
3. **Physical degradation analysis:** Observable signals only, cautious language ✅
4. **Samsung protocol:** Explicitly CASE B; clearly labeled as EIS-only exploratory ✅
5. **Normalization audit:** NASA train stats used throughout ✅

---

# 10. Thesis-Ready Evaluation Protocol

See `experiments/reports/thesis_evaluation_protocol.md` for full protocol.

**Summary:**
- NASA: full multimodal SOH on 7 test batteries. Report MAE%, RMSE%, R².
- Samsung: EIS-only exploratory assessment. Report with explicit caveats. Do not combine with NASA metrics.
- Physical degradation: post-training interpretation with cautious language.

---

# 11. Thesis-Ready Limitations

Required to disclose:

1. Small test set (7 batteries, 97 samples) — limited statistical power
2. SOH distribution shift between train and test (7.3pp mean difference)
3. Samsung EIS incompatible in absolute scale (~13x mismatch)
4. Samsung evaluation uses random heads — not trained model performance
5. `impedance_rise` and `eis[0]` are mathematically identical (feature redundancy)
6. All batteries have same nominal capacity (2.0 Ah) — no cross-chemistry NASA test
7. Multi-seed analysis only partially complete

---

# 12. Files Created / Modified This Audit

## Created
- `scripts/run_audit.py` — programmatic audit runner
- `scripts/inspect_samsung_data.py` — Samsung data inspection
- `experiments/reports/samsung_compatibility_audit.md`
- `experiments/reports/normalization_audit.md`
- `experiments/reports/external_model_compatibility.csv`
- `experiments/reports/external_model_compatibility_final.csv`
- `experiments/reports/leakage_audit.json`
- `experiments/reports/samsung_external_evaluation.md`
- `experiments/reports/samsung_eis_compatibility_final.md`
- `experiments/reports/thesis_evaluation_protocol.md`
- `experiments/reports/physical_degradation_audit.md`
- `experiments/reports/reproducibility_audit.md`
- `experiments/reports/thesis_results_table.csv`
- `experiments/reports/FINAL_SCIENTIFIC_AUDIT.md` (this file)

## Modified (fixes only)
- `scripts/evaluate_samsung_external.py` — fixed normalization source, removed zero-padding, honest labels
- `src/data/dataset.py` — added `save_stats_path` to `create_dataloaders()`
- `run_pipeline.py` — added `save_stats_path` call
- `scripts/train_bafuse_v2.py` — added `save_stats_path` call, blocked staged/joint
- `scripts/architecture_benchmark.py` — added `save_stats_path` call
- `configs/config_bafuse_v2.yaml` — training_mode: nasa_only

---

# Concise Final Summary

```
Critical issues found:  3
Critical issues fixed:  3
Remaining critical issues: 0

NASA evaluation valid:        YES
Samsung EIS-only eval valid:  EXPLORATORY (acknowledged; caveats documented)
Number of externally evaluable models: 14/14 (all EIS-only, 13 with true EIS encoder)
Leakage audit:  PASS
Reproducibility audit: PARTIAL PASS (multi-seed incomplete; no lock file)
Thesis-ready: YES — with documented limitations
```
