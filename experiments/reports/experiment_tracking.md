# BaFuse Experiment Tracking

**Last Updated:** 2026-09-21  
**Project:** Battery Fusion State-of-Health (SOH) Estimation with Degradation Analysis

---

## 1. Research Objective

**PRIMARY GOAL:** Develop a multimodal deep learning system that:
1. Estimates battery State-of-Health (SOH) from discharge curves, EIS measurements, and physics-informed features
2. Provides degradation indicators to support interpretation of battery aging mechanisms

**CRITICAL SCIENTIFIC DISTINCTIONS:**

- **TRACK 1 (SOH):** NASA PCoE is the PRIMARY dataset for training, validation, and testing
- **TRACK 2 (Degradation):** Physics-based indicators derived from NASA observables + literature interpretation
- **Samsung/Mendeley:** EXTERNAL TEST ONLY — NOT used for NASA training, normalization, or model selection
- **LLI/LAM/CL labels:** ECM-derived indicators (NOT direct physical ground truth)

---

## 2. Dataset Versions

### 2.1 NASA PCoE Battery Aging Dataset

**Source:** NASA Ames Prognostics Center of Excellence  
**Batteries:** 34 cells (B0005–B0056), 18650 Li-ion, 2.0 Ah nominal  
**Campaigns:** 6 different aging protocols  
**Total Samples:** 935 discharge-EIS paired cycles  

**Preprocessing Version:** `v1_current`
- Discharge resampled to 100 timesteps (V/I/T, 3 channels)
- EIS aggregated to 3 features: median |Z|, Re(Z), |Im(Z)| across frequency spectrum
- Physics features: 4 engineered (cycle_age_norm, empirical_fade_prior, voltage_droop, impedance_rise)
- Capacity filter: cycles with capacity < 25% nominal dropped per battery
- Pairing: discharge-EIS matched within max_cycle_gap=10
- Normalization: z-score per feature using TRAIN data only

**Files:**
- `data/processed/train.pkl` — 616 samples, 20 batteries
- `data/processed/val.pkl` — 222 samples, 7 batteries
- `data/processed/test.pkl` — 97 samples, 7 batteries
- `data/processed/paired.pkl` — 935 samples (all batteries)
- `data/processed/discharge_raw.pkl` — (B, 100, 3) time-series
- `data/processed/nasa_train_stats.json` — normalization statistics

**Battery-Level Split (seed=42):**
- **Train:** B0006, B0007, B0018, B0025, B0027, B0030, B0032, B0033, B0036, B0040, B0041, B0042, B0043, B0044, B0045, B0046, B0050, B0053, B0054, B0056
- **Val:** B0005, B0026, B0031, B0034, B0039, B0049, B0052
- **Test:** B0028, B0029, B0038, B0047, B0048, B0051, B0055

**Leakage Audit:** PASS (verified zero battery overlap, train-only normalization)

### 2.2 Samsung/Mendeley Degradation Dataset

**Source:** "Li-ion cells EIS dataset with fitting and degradation modes" (Mendeley Data)  
**Batteries:** 8 Samsung INR18650-30Q cells, 2.95 Ah nominal  
**Total Samples:** 488 EIS measurements  

**Features:**
- EIS: zmod_ohm, zphz_deg, zreal_ohm, zimg_ohm (4 direct EIS)
- ECM params: R_electrolyte, R_ct1, R_ct2, Zw, ocv_v (5 circuit-fitted)
- Labels: lli_pct, lam_pct, cl_pct (ECM-derived, NOT ground truth)
- SOH: soh_pct (from capacity)

**CRITICAL LABEL AUDIT:**
- **LLI:** 119/488 samples affected (range: -34.80% to 3516.42%, std=402%)
- **LAM:** 30/488 samples affected (negative values and >100%)
- **CL:** Clean range [0, 12.46%]
- **Cause:** ECM fitting instability, multiple local minima
- **Status:** Labels are MODEL-DERIVED, NOT direct physical measurements

**Files:**
- `data/mendeley_processed/mendeley_combined.pkl`
- `data/mendeley_processed/mendeley_train.pkl`
- `data/mendeley_processed/mendeley_val.pkl`
- `data/mendeley_processed/mendeley_test.pkl`
- `data/mendeley_processed/mendeley_stats.json` — NOT used for NASA

**Use:** EXTERNAL TEST ONLY (no NASA contamination)

---

## 3. Preprocessing Versions

### NASA v1_current (Active)

**ID:** `nasa_v1_current`  
**Date:** 2024-2026 (evolved during project)  
**Description:** Current production preprocessing

**Discharge:**
- Input: Raw V/I/T time-series from .mat files
- Resampling: Linear interpolation to 100 timesteps
- Normalization: z-score per channel (train-only stats)
- Output shape: (B, 100, 3)
- Time axis: Normalized to [0, 1] (absolute duration discarded)

**EIS:**
- Input: Full frequency spectrum (Re, Im per frequency)
- Aggregation: Median across frequencies
- Features: 3 scalars [|Z|_median, Re_median, |Im|_median]
- Normalization: z-score (train-only stats)
- Output shape: (B, 3)

**Physics:**
- 4 engineered features:
  1. `cycle_age_norm = discharge_cycle / N_ref` (N_ref from train power-law fit)
  2. `empirical_fade_prior = A * (cycle_age_norm)^b` (train-only fit)
  3. `voltage_droop = (voltage_min - cutoff) / (4.2 - cutoff)` (per-battery cutoff)
  4. `impedance_rise = z_score(impedance_ohm)` — **REDUNDANT with eis[0]**
- Normalization: None (already scaled/normalized)
- Output shape: (B, 4)

**Target:**
- SOH = capacity_ahr / 2.0 (all NASA cells 2.0 Ah nominal)
- Clipping: np.clip(soh, 0, 1) for 3 early-cycle samples >1.0
- Output: scalar ∈ [0, 1]

**Known Issues:**
- Time duration lost (normalized to [0,1])
- EIS frequency structure discarded (only median used)
- `impedance_rise` == `eis[0]` (feature redundancy)

**Status:** VALIDATED, BENCHMARKED (9 models tested)

### Candidate Preprocessing Ideas (NOT YET IMPLEMENTED)

**nasa_v2_duration:**
- Preserve discharge duration as explicit feature
- Add `discharge_duration_sec` to physics features
- Keep all other aspects same as v1

**nasa_v3_full_eis:**
- Use full EIS frequency spectrum instead of median
- Input: (B, n_freq, 2) for [Re, Im] at each frequency
- Requires frequency-aware encoder (1D-CNN or attention)

**nasa_v4_protocol_aware:**
- Add campaign/protocol indicators (one-hot or learned embedding)
- Separate handling per cutoff voltage group if variance is high

**Status:** CANDIDATES ONLY — require justification before implementation

---

## 4. SOH Experiments

### EXP-SOH-001: Architecture Benchmark (Phase 1)

**ID:** `exp_soh_001`  
**Date:** 2026-09  
**Track:** SOH Estimation  
**Status:** ✅ COMPLETED

**Dataset:** NASA v1_current  
**Split:** Battery-level (20/7/7)  
**Preprocessing:** nasa_v1_current  
**Models:** A, B, C, D, E, E2, F, G, H (9 architectures)  
**Seeds:** 42 (single seed only)  
**Training:** 100 epochs, early stop patience=15, lr=1e-3, wd=1e-4, batch=32  
**Loss:** MSE  
**Normalization:** Train-only stats  
**Leakage Check:** PASS

**Results (Test Set):**

| Model | Architecture | Params | Val MAE% | Test MAE% | Test RMSE% | Test R² | Best Epoch | Time(s) |
|-------|-------------|--------|----------|-----------|------------|---------|------------|---------|
| **D** | TCN | 44,257 | 2.82 | **5.56** | 7.91 | **0.753** | 16 | 103 |
| **G** | CNN+Gated | 24,964 | 3.14 | **5.13** | 7.50 | **0.778** | 18 | 68 |
| C | CNN1D | 24,193 | 3.16 | 5.65 | 8.20 | 0.735 | 24 | 90 |
| E | LSTM64 | 30,977 | 5.50 | 5.81 | 8.83 | 0.693 | 9 | 59 |
| F | Gated | 18,148 | 2.76 | 6.41 | 8.38 | 0.723 | 33 | 96 |
| B | SmallMLP | 16,545 | 3.12 | 6.45 | 8.52 | 0.713 | 23 | 73 |
| A | TinyMLP | 3,431 | 2.22 | 7.49 | 9.94 | 0.611 | 57 | 138 |
| E2 | LSTM128 | 85,505 | 4.50 | 7.38 | 10.14 | 0.595 | 31 | 166 |
| H | BaFuse v1 | 1,536,321 | 3.04 | 8.84 | 10.54 | 0.562 | 61 | 989 |

**Baseline:** Mean predictor test MAE = 15.68%

**Conclusion:**
- Models D and G are top performers (test MAE ~5.1-5.6%, R² ~0.75-0.78)
- All models beat baseline by wide margin
- Model H (BaFuse v1, 1.5M params) UNDERPERFORMS smaller architectures → likely overparameterized for 616 train samples
- **Single-seed results — variance unknown**

**Checkpoint:** `results/architecture_benchmark/` (individual model checkpoints TBD)

**Next Action:** Multi-seed evaluation on top-3 architectures (D, G, C)

---

### EXP-SOH-002: Multi-Seed Top-3 (PLANNED)

**ID:** `exp_soh_002`  
**Status:** 🔄 PLANNED  
**Priority:** HIGH

**Objective:** Quantify variance and confirm ranking stability

**Models:** D (TCN), G (CNN+Gated), C (CNN1D)  
**Seeds:** 42, 123, 2026  
**Other params:** Same as EXP-SOH-001

**Expected Output:**
- Mean ± std for MAE/RMSE/R² per model
- Statistical significance of differences
- Variance quantification

**Runs:** 3 models × 3 seeds = 9 training runs

---

### EXP-SOH-003: Preprocessing Ablation (FUTURE)

**ID:** `exp_soh_003`  
**Status:** 📋 CANDIDATE  
**Priority:** MEDIUM

**Objective:** Test whether duration/full-EIS/protocol-features improve SOH

**Variants:**
- nasa_v1_current (baseline)
- nasa_v2_duration (add discharge duration)
- nasa_v3_full_eis (frequency-aware EIS)
- nasa_v4_protocol_aware (campaign features)

**Models:** Best from EXP-SOH-002  
**Seeds:** 3

**Prerequisite:** Implement candidate preprocessing variants

**Runs:** 4 preprocessing × 1 model × 3 seeds = 12 training runs

---

### EXP-SOH-004: LOBO B0005/B0006/B0007/B0018 (FUTURE)

**ID:** `exp_soh_004`  
**Status:** 📋 CANDIDATE  
**Priority:** LOW (paper-comparison only)

**Objective:** Enable comparison with published papers using LOBO protocol

**Protocol:** Leave-One-Battery-Out on B0005, B0006, B0007, B0018  
**Model:** Best from EXP-SOH-002  
**Seeds:** 3

**Note:** Separate from main 34-cell benchmark

**Runs:** 4 batteries × 3 seeds = 12 training runs

---

## 5. Degradation Analysis Experiments

### Current Degradation Analysis Status

**Approach:** Physics + Literature-Based Indicators (NOT supervised learning on LLI/LAM/CL)

**NASA Observable Signals:**

| Signal | Available | Source | Use |
|--------|-----------|--------|-----|
| Capacity (Ah) | ✅ | Direct measurement | SOH, capacity fade |
| Discharge cycle | ✅ | Metadata | Cycle age |
| Impedance \|Z\| | ✅ | EIS | Impedance growth |
| Re (Ohm) | ✅ | EIS | Electrolyte resistance trend |
| Rct (Ohm) | ✅ | EIS | Charge-transfer resistance trend |
| Voltage min/max | ✅ | Discharge curve | Voltage droop, plateau analysis |
| Current (A) | ✅ | Discharge curve | Charge throughput |
| Temperature (°C) | ✅ | Discharge curve | Thermal stress |
| Discharge duration | ⚠️ | Derivable (not stored) | Time-to-cutoff |
| ICA/DVA | ❌ | Requires differentiation | NOT IMPLEMENTED |
| R_electrolyte | ❌ | Requires ECM fitting | NOT AVAILABLE |
| R_ct1/R_ct2 | ❌ | Requires ECM fitting | NOT AVAILABLE |
| **LLI labels** | ❌ | NOT AVAILABLE | NOT AVAILABLE |
| **LAM labels** | ❌ | NOT AVAILABLE | NOT AVAILABLE |
| **CL labels** | ❌ | NOT AVAILABLE | NOT AVAILABLE |

**NASA does NOT have LLI/LAM/CL ground truth.**

### EXP-DEG-001: Samsung EIS-Only Reference (COMPLETED)

**ID:** `exp_deg_001`  
**Date:** 2026-09  
**Track:** Degradation (Samsung-internal reference)  
**Status:** ✅ COMPLETED (EXPLORATORY)

**Objective:** Test whether Samsung EIS features contain predictive signal for ECM-derived LLI/LAM/CL

**Dataset:** Samsung/Mendeley (8 cells, 488 samples)  
**Split:** Cell-level (cells 1-5 train / 6-7 val / 8 test)  
**Model:** DegradationMLP (4,995 params)  
**Input:** 9 EIS features  
**Targets:** LLI, LAM, CL (ECM-derived)  
**Normalization:** Train-only (Samsung cells 1-5)  
**Seed:** 42  
**Training:** 100 epochs, early stop at 96, lr=1e-3  
**Leakage Check:** PASS (cell-level split)

**Results (Cell 8 Test):**

| Target | MAE | RMSE | R² | Interpretation |
|--------|-----|------|----|----------------|
| **LLI** | 52.36% | 74.77% | **-0.03** | No predictive power |
| **LAM** | 14.54% | 18.28% | **0.27** | Weak prediction |
| **CL** | 1.18% | 1.32% | **0.79** | Good prediction |

**Conclusion:**
- CL is predictable from EIS (R²=0.79)
- LLI has no predictive signal (R²=-0.03) — likely due to extreme label variance (std=402%)
- LAM has weak signal (R²=0.27)
- **Test set: 1 cell only (85 samples) — statistically insufficient**
- Labels are ECM-derived, NOT ground truth

**Checkpoint:** `experiments/checkpoints/best_degradation_quick_test.pth`

**Limitations:**
- 1-cell test set
- ECM-derived labels
- No NASA involvement

**Status:** EXPLORATORY REFERENCE ONLY (NOT final degradation system)

**Next Action:** LOCO cross-validation on Samsung (optional, separate from NASA)

### EXP-DEG-002: Samsung LOCO Cross-Validation (PLANNED)

**ID:** `exp_deg_002`  
**Status:** 📋 CANDIDATE  
**Priority:** LOW (Samsung-internal reference only)

**Objective:** Robust Samsung-internal evaluation with 8-fold LOCO

**Protocol:** Leave-One-Cell-Out (8 cells)  
**Model:** DegradationMLP  
**Metric:** Mean ± std across 8 held-out cells  
**Hyperparameters:** Fixed a priori (from EXP-DEG-001)

**Expected Output:**
- LLI: mean MAE ± std
- LAM: mean MAE ± std
- CL: mean MAE ± std

**Runs:** 8 LOCO folds

**Note:** This is a Samsung-internal experiment. Does NOT affect NASA degradation analysis.

### EXP-DEG-003: NASA Physics-Based Indicators (FUTURE)

**ID:** `exp_deg_003`  
**Status:** 📋 DESIGN PHASE  
**Priority:** HIGH

**Objective:** Derive physics-based degradation indicators from NASA observables

**Proposed Indicators:**

1. **Capacity Fade Trajectory**
   - Input: SOH over cycles
   - Method: Fit power-law A*(cycle/N_ref)^b
   - Output: Fade rate parameter b

2. **Impedance Growth Rate**
   - Input: |Z| over cycles
   - Method: Linear/exponential fit
   - Output: Growth rate (Ohm/cycle)

3. **Re Evolution**
   - Input: Re(Z) over cycles
   - Method: Trend analysis
   - Output: Electrolyte resistance change

4. **Rct Evolution**
   - Input: |Im(Z)| over cycles
   - Method: Trend analysis
   - Output: Charge-transfer resistance change

5. **Voltage Droop Progression**
   - Input: voltage_min over cycles
   - Method: Track minimum voltage drop
   - Output: Droop rate

**Literature Support:** Requires citations for each indicator

**Implementation:** `src/degradation_analysis.py` (physics-based rules)

**Validation:** Compare trends against literature expectations

**Status:** REQUIRES DESIGN AND LITERATURE REVIEW

---

## 6. Best SOH Configuration

### Current Best (Single-Seed)

**Model:** G (CNN+Gated) or D (TCN)  
**Test MAE:** 5.13% (G) / 5.56% (D)  
**Test R²:** 0.778 (G) / 0.753 (D)  
**Preprocessing:** nasa_v1_current  
**Seed:** 42

**Status:** PRELIMINARY (single seed only — variance unknown)

**Checkpoint:** `results/architecture_benchmark/` (specific path TBD)

**Next Required Step:** Multi-seed validation (EXP-SOH-002)

---

## 7. Current Degradation Analysis Status

**NASA Side:**
- Physics-based indicators: NOT YET IMPLEMENTED
- Observable signals: Available (impedance, Re, Rct, voltage, capacity)
- LLI/LAM/CL labels: NOT AVAILABLE (NASA does not provide these)

**Samsung Side:**
- EIS-only reference model: COMPLETED (exploratory)
- LOCO validation: PLANNED (optional)
- Label quality: ECM-derived (high variance in LLI/LAM)

**Final System Concept:**
- SOH: From NASA supervised learning
- Degradation indicators: From NASA physics-based analysis + literature
- External validation: Samsung as independent test (NOT training)

---

## 8. Failed / Rejected Experiments

### REJ-001: BaFuse v1 (Model H)

**Reason:** Severe underperformance (test MAE 8.84%, worse than simpler models)  
**Diagnosis:** 1.5M params overparameterized for 616 train samples  
**Params:** 1,536,321  
**Test R²:** 0.562  
**Status:** REJECTED

**Lesson:** Complexity does not guarantee performance on small datasets

---

## 9. Leakage and Reproducibility Audits

### Leakage Audits

**NASA Split Audit:** PASS  
- Zero battery overlap across train/val/test
- Zero duplicate (battery, cycle) pairs
- Train-only normalization
- No capacity_ahr in model inputs
- Empirical prior fit on train only

**Samsung Contamination Audit:** PASS  
- Samsung data NOT in NASA training
- Samsung data NOT in NASA normalization
- Samsung data NOT in NASA model selection

**Feature Leakage Audit:** PASS (with noted redundancy)  
- No target-derived inputs
- `impedance_rise` == `eis[0]` (redundant but not leaky)

**Audit Report:** `experiments/reports/FINAL_SCIENTIFIC_AUDIT.md`

### Reproducibility Status

**NASA Benchmark:**
- Seed: 42 (documented)
- Split: Deterministic (battery-level, seed=42)
- Normalization: Train-only stats saved
- Checkpoint: Per-model (partially saved)
- Config: Documented in `results/architecture_benchmark/config.json`

**Gaps:**
- Multi-seed results: NOT YET DONE
- Per-model checkpoints: NOT all saved
- Environment lock file: NOT PRESENT (only requirements.txt)

**Status:** PARTIAL REPRODUCIBILITY

**Audit Report:** `experiments/reports/reproducibility_audit.md`

---

## 10. Scientific Decisions

### SD-001: NASA is PRIMARY, Samsung is EXTERNAL TEST ONLY

**Date:** 2026-09  
**Decision:** NASA PCoE is the sole training/validation/model-selection dataset. Samsung/Mendeley is external test only.  
**Rationale:** Avoid cross-dataset contamination and maintain scientific rigor.  
**Status:** ENFORCED

### SD-002: ECM-Derived Labels Are NOT Ground Truth

**Date:** 2026-09  
**Decision:** Samsung LLI/LAM/CL are ECM-derived indicators, NOT direct physical measurements.  
**Rationale:** ECM fitting has high instability (LLI std=402%, 119/488 outliers).  
**Status:** DOCUMENTED

### SD-003: Physics-Based Degradation Analysis for NASA

**Date:** 2026-09  
**Decision:** NASA degradation indicators will be derived from physics-based rules + literature, NOT supervised learning on unavailable LLI/LAM/CL labels.  
**Rationale:** NASA does not have LLI/LAM/CL ground truth.  
**Status:** DESIGN PHASE

### SD-004: Battery-Level Split for NASA

**Date:** 2024-2026 (evolved)  
**Decision:** NASA uses battery-level split (no battery overlap across train/val/test).  
**Rationale:** Prevents cell-identity leakage.  
**Status:** VALIDATED

---

## 11. Current Status

**SOH Track:**
- ✅ NASA benchmark completed (9 models, single seed)
- ✅ Top performers identified: D (TCN), G (CNN+Gated)
- 🔄 Multi-seed validation: REQUIRED NEXT
- 📋 Preprocessing ablation: CANDIDATE

**Degradation Track:**
- ✅ Samsung EIS-only reference: COMPLETED (exploratory)
- 📋 NASA physics-based indicators: DESIGN PHASE
- 📋 Samsung LOCO: OPTIONAL

**Leakage/Reproducibility:**
- ✅ All audits: PASS
- ⚠️ Multi-seed: MISSING
- ⚠️ Checkpoints: PARTIALLY SAVED

---

## 12. Next Experiment

**IMMEDIATE PRIORITY:** EXP-SOH-002 (Multi-Seed Top-3)

**Objective:** Confirm ranking stability and quantify variance

**Execution Plan:**
1. Models: D, G, C
2. Seeds: 42 (reuse existing), 123, 2026
3. Same hyperparameters as EXP-SOH-001
4. Training: 9 runs total (3 models × 3 seeds)
5. Report: Mean ± std for MAE/RMSE/R²
6. Decision: Select final SOH architecture based on mean performance and stability

**Files to Modify:**
- `scripts/architecture_benchmark.py` (add multi-seed flag)
- `results/architecture_benchmark/benchmark_summary.csv` (append new rows)

**Expected Duration:** ~1-2 hours (9 runs × ~5-10 min each)

**After Completion:**
1. Update this tracking document
2. Select final SOH configuration
3. Decide whether preprocessing ablation (EXP-SOH-003) is justified
4. Begin NASA physics-based degradation indicator design (EXP-DEG-003)

---

**END OF EXPERIMENT TRACKING**
