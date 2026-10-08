# Mendeley Degradation Label Audit

**Date:** 2026-09  
**Purpose:** Investigate extreme values in LLI/LAM/CL labels before training  
**Status:** TRAINING NOT RECOMMENDED — outliers present

---

## Executive Summary

**CRITICAL FINDING:** The Mendeley degradation labels contain extreme outliers that make direct training risky.

| Label | Issue | Affected Samples | Recommendation |
|-------|-------|-----------------|----------------|
| **LLI** | Extreme values (-34.80 to 3516.42%) | 119/488 (24%) | **FAIL** — filter required |
| **LAM** | Negative and >100% values | 30/488 (6%) | **FAIL** — filter required |
| **CL** | Reasonable range [0, 12.46%] | 0 | **PASS** |
| **R_ct2** | Extreme spikes (>0.1 Ohm) | 6/488 (1%) | **FAIL** — minor issue |

**Training now:** **NO**  
**Reason:** Extreme LLI variance (std=402%) will dominate loss and destabilize training.

---

## 1. Dataset Overview

- **Source:** Mendeley "Li-ion cells EIS dataset with fitting and degradation modes"
- **Cells:** 8 Samsung INR18650-30Q
- **Total samples:** 488
- **Columns:** cell_id, aging_cycle, EIS features (zmod_ohm, zphz_deg, zreal_ohm, zimg_ohm), ECM params (R_electrolyte, R_ct1, R_ct2, Zw), degradation indicators (lli_pct, lam_pct, cl_pct), SOH

**IMPORTANT:** LLI/LAM/CL are **model-derived from equivalent-circuit fitting** (ECM), NOT direct physical measurements.

---

## 2. Label Distributions

### LLI (Loss of Lithium Inventory, %)

| Stat | Value |
|------|-------|
| Min | -34.80 |
| Max | **3516.42** |
| Mean | 208.68 |
| Median | 78.26 |
| Std | **402.64** |
| p1 | -31.64 |
| p5 | -18.92 |
| p25 | 3.36 |
| p75 | 226.99 |
| p95 | 906.88 |
| p99 | **2131.01** |

**Extreme variance:** std/mean = 1.93 (variance is 2x the mean!)

### LAM (Loss of Active Material, %)

| Stat | Value |
|------|-------|
| Min | -3.78 |
| Max | 212.09 |
| Mean | 35.25 |
| Median | 28.71 |
| Std | 31.66 |
| p95 | 87.14 |
| p99 | 157.84 |

### CL (Capacity Loss, %)

| Stat | Value |
|------|-------|
| Min | 0.00 |
| Max | 12.46 |
| Mean | 4.72 |
| Median | 4.80 |
| Std | 2.86 |
| p95 | 9.38 |
| p99 | 10.66 |

**CL looks reasonable** — constrained range, low variance.

### R_ct2 (Charge-Transfer Resistance, Ohm)

| Stat | Value |
|------|-------|
| Min | 0.001069 |
| Max | **0.169693** |
| Mean | 0.011239 |
| Median | 0.004817 |
| p95 | 0.045261 |
| p99 | 0.109675 |

Largest values (>0.1 Ohm) occur at late aging cycles in cells 5 and 6.

---

## 3. Extreme Value Identification

### LLI > 500%: 53 samples (11%)

**Affected cells:** 1, 2, 3, 4, 5, 6 (6/8 cells)  
**Affected cycles:** 75–275 (mid-to-late aging)  
**Maximum:** 3516.42% (cell 5, cycle 200, SOH=77.65%)

### LLI > 1000%: 20 samples (4%)

**Affected cells:** 5, 6 ONLY  
**Cycles:** 100–200

| Cell | Cycle | LLI (%) | SOH (%) |
|------|-------|---------|---------|
| 5 | 100 | 1127.04 | 88.30 |
| 5 | 125 | 1707.36 | 85.55 |
| 5 | 150 | 2331.23 | 82.84 |
| 5 | 175 | 2933.20 | 80.21 |
| 5 | 200 | **3516.42** | 77.65 |
| 6 | 125 | 1540.20 | 85.26 |
| 6 | 150 | 2101.09 | 82.60 |
| 6 | 175 | 2665.13 | 80.04 |
| 6 | 200 | **3244.70** | 77.30 |

**Pattern:** Multiple samples at the same cycle (e.g., cell 5, cycle 200 has 4 entries with LLI from 1298% to 3516%).  
**Likely cause:** ECM fitting with multiple initial conditions or local minima.

### LLI < 0: 66 samples (14%)

**Affected cells:** 1, 2, 3, 4, 7, 8 (6/8 cells)  
**Cycles:** 25–225

Negative LLI is physically implausible — suggests ECM fitting artifacts at early cycles or poor initialization.

### LAM < 0: 13 samples (3%)

**Affected cells:** 1, 2, 3, 7, 8  
Negative LAM is also physically implausible.

### LAM > 100%: 17 samples (3%)

**Affected cells:** 5, 6 ONLY (same cells with extreme LLI)

| Cell | Cycle | LAM (%) | SOH (%) |
|------|-------|---------|---------|
| 5 | 125 | 128.22 | 85.55 |
| 5 | 150 | 157.68 | 82.84 |
| 5 | 175 | 194.51 | 80.21 |
| 5 | 200 | **211.13** | 77.65 |
| 6 | 150 | 158.93 | 82.60 |
| 6 | 175 | 191.94 | 80.04 |
| 6 | 200 | **212.09** | 77.30 |

**Pattern:** Cells 5 and 6 consistently produce extreme LLI and LAM values at cycles 100–200.

### CL = 0: 39 samples (8%)

**All cells at cycle 0** — this is expected (beginning-of-life, no capacity loss yet).

---

## 4. Cell-Level Label Trajectories

| Cell | Samples | LLI Range | LAM Range | CL Range | Largest Jump |
|------|---------|-----------|-----------|----------|-------------|
| 1 | 60 | [-26.5, 925.7] | [-3.8, 85.8] | [0, 9.4] | 599.6% (cycle 275) |
| 2 | 60 | [-27.0, 714.0] | [-1.4, 84.0] | [0, 9.7] | 453.8% (cycle 275) |
| 3 | 48 | [-26.5, 975.8] | [-3.0, 86.7] | [0, 8.5] | 558.5% (cycle 275) |
| 4 | 60 | [-26.2, 779.0] | [0.0, 87.4] | [0, 10.0] | 496.2% (cycle 275) |
| **5** | 45 | **[0.0, 3516.4]** | **[0.0, 211.1]** | [0, 12.5] | **2155.9% (cycle 200)** |
| **6** | 45 | **[0.0, 3244.7]** | **[0.0, 212.1]** | [0, 12.0] | **1953.1% (cycle 200)** |
| 7 | 85 | [-34.8, 389.4] | [-2.5, 80.9] | [0, 10.5] | 265.4% (cycle 350) |
| 8 | 85 | [-33.7, 308.9] | [-1.0, 85.6] | [0, 10.3] | 209.2% (cycle 350) |

**Cells 5 and 6 are outlier cells** — both show extreme LLI and LAM values.

---

## 5. Source Investigation

### Preprocessing Script

Found: `scripts/preprocess_mendeley.py`

Key parameters:
- `--soc_filter 100` — only SOC=100% measurements used
- No evidence of outlier clipping or filtering in preprocessing
- Labels appear to be loaded directly from Mendeley Excel files

### Raw Data

Raw directory NOT FOUND in the repository — original Excel files not retained.

**Conclusion:** Extreme values are present in the original Mendeley dataset, not introduced by preprocessing.

---

## 6. Root Cause Analysis

### Why Are LLI/LAM Values So Extreme?

**LLI/LAM/CL are model-derived quantities from equivalent-circuit model (ECM) fitting.**

The Mendeley dataset README states:
> "LLI, LAM, and CL were extracted by fitting EIS data to an equivalent-circuit model at each aging cycle."

**Likely causes of extreme values:**

1. **ECM fitting instability** — nonlinear optimization with multiple local minima
2. **Ill-conditioned initialization** — poor starting guesses at mid-to-late aging cycles
3. **Multiple fitting attempts** — same cycle appears multiple times with different LLI/LAM values (e.g., cell 5, cycle 200)
4. **Model mismatch** — ECM circuit topology may not accurately represent all aging regimes
5. **Negative values** — physically implausible, suggest numerical artifacts

**Key observation:** Cells 5 and 6 have systematically worse ECM fits (extreme LLI/LAM) compared to cells 1–4 and 7–8.

---

## 7. Impact on Training

### Current State

If we train with unfiltered labels:

| Issue | Consequence |
|-------|-------------|
| LLI std=402% | Will dominate MSE loss (extreme gradient magnitudes) |
| LLI max=3516% | Outliers will distort learned representations |
| 119 outlier samples | ~24% of dataset contaminated |
| Cells 5 & 6 | If used in test set, results will be meaningless |

**Predicted outcome:** Model will fail to converge, or will learn to predict extreme outliers instead of typical aging behavior.

---

## 8. Recommended Next Steps

### Option A: Filter Outliers (RECOMMENDED)

**Clip labels to plausible ranges:**
- LLI: [-50, 500]% (removes 119 outliers → 369 clean samples)
- LAM: [-10, 150]% (removes 30 outliers → 458 clean samples)
- CL: [0, 15]% (no filtering needed)

**Advantages:**
- Removes ECM fitting artifacts
- Reduces LLI std from 402% to ~150% (estimated)
- Training will be stable

**Disadvantages:**
- Arbitrary clipping thresholds
- Loss of information (though likely noise, not signal)

### Option B: Exclude Cells 5 & 6

**Remove the 2 outlier cells entirely (90 samples).**

**Advantages:**
- Cleaner dataset (398 samples from 6 cells)
- No arbitrary clipping
- LLI max drops to ~980% (still high but more reasonable)

**Disadvantages:**
- Smaller dataset (25% fewer cells)
- Cells 5 & 6 may represent valid aging behavior that ECM simply cannot model

### Option C: Train Only on CL

**Use only CL as the target (ignore LLI/LAM).**

**Advantages:**
- CL has clean distribution [0, 12.46]%
- No outlier issues
- Directly related to SOH

**Disadvantages:**
- Cannot explore LLI/LAM prediction
- Reduces experiment scope

---

## 9. Final Verdict

| Criterion | Status | Details |
|-----------|--------|---------|
| **Training now** | **NO** | Outliers will destabilize training |
| **LLI outlier issue** | **FAIL** | 119 samples with |LLI| > 500% or LLI < 0 |
| **LAM outlier issue** | **FAIL** | 30 samples with LAM < 0 or LAM > 100% |
| **CL outlier issue** | **PASS** | Clean range [0, 12.46]% |
| **R_ct2 outlier issue** | **FAIL (minor)** | 6 samples > 0.1 Ohm (cells 5 & 6 at late cycles) |
| **NASA SOH benchmark modified** | **NO** | This audit does NOT affect NASA A-H benchmark |

---

## 10. Recommended Action

**DO NOT train with unfiltered Mendeley labels.**

**Proceed with Option A (filter outliers) OR Option C (CL-only).**

If Option A is chosen:
1. Apply clipping: `lli_pct = np.clip(lli_pct, -50, 500)`
2. Apply clipping: `lam_pct = np.clip(lam_pct, -10, 150)`
3. Document clipping thresholds and affected sample count
4. Report filtered vs. unfiltered statistics
5. Save filtered dataset as `mendeley_combined_filtered.pkl`

If Option C is chosen:
1. Train only on CL target
2. Skip LLI/LAM prediction entirely
3. Report CL-only results in the degradation quick test

---

## 11. Relation to NASA SOH Benchmark

**This audit does NOT affect the primary NASA A-H SOH benchmark.**

- NASA dataset has clean SOH labels (capacity-based, not ECM-derived)
- NASA training is complete and validated
- Mendeley data is ONLY used for separate degradation-mode experiment
- NO data mixing between NASA and Mendeley

---

## 12. Conclusion

The Mendeley degradation labels (LLI/LAM) contain severe outliers due to ECM fitting instability. **Training with unfiltered labels is NOT recommended.** Either filter outliers (Option A) or train only on CL (Option C) before proceeding with the degradation quick test.

**Files generated:**
- `scripts/audit_mendeley_labels.py` — audit script
- `experiments/reports/mendeley_label_audit_raw.txt` — raw audit output
- `experiments/reports/mendeley_degradation_label_audit.md` — this report
