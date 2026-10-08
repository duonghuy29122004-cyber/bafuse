# Degradation Quick Test — Separate Experiment

**Date:** 2026-09  
**Purpose:** Test whether EIS features can predict degradation-mode indicators

---

## IMPORTANT DISCLAIMERS

1. **This is a separate experiment** — does NOT modify NASA A-H SOH benchmark
2. **Samsung/Mendeley data NOT mixed with NASA training**
3. **LLI/LAM/CL are model-derived indicators** from equivalent-circuit fitting (ECM)
4. **NOT direct physical ground truth** — they are surrogate labels from circuit-fitting algorithms

---

## 1. Dataset

- **Source:** Mendeley "Li-ion cells EIS dataset with fitting and degradation modes"
- **Cells:** 8 Samsung INR18650-30Q cells
- **Total samples:** 488
- **Cell distribution:**

| Cell | Samples | Aging cycles |
|------|---------|--------------|
| 1 | 60 | 0-275 |
| 2 | 60 | 0-275 |
| 3 | 48 | 0-275 |
| 4 | 60 | 0-275 |
| 5 | 45 | 0-200 |
| 6 | 45 | 0-200 |
| 7 | 85 | 0-400 |
| 8 | 85 | 0-400 |

---

## 2. Features

**EIS features (9 total):**
- `zmod_ohm` — impedance magnitude |Z|
- `zphz_deg` — phase angle
- `zreal_ohm` — real part Re(Z)
- `zimg_ohm` — imaginary part Im(Z)
- `R_electrolyte` — electrolyte resistance (from ECM fitting)
- `R_ct1` — charge-transfer resistance 1
- `R_ct2` — charge-transfer resistance 2
- `Zw` — Warburg coefficient
- `ocv_v` — open-circuit voltage

**Targets (model-derived from ECM fitting, NOT direct physical measurements):**
- `lli_pct` — Loss of Lithium Inventory (%)
- `lam_pct` — Loss of Active Material (%)
- `cl_pct` — Capacity Loss (%)

**Label distributions:**

| Target | Min | Max | Mean | Std | p50 |
|--------|-----|-----|------|-----|-----|
| LLI (%) | -34.80 | 3516.42 | 208.68 | 402.64 | 78.26 |
| LAM (%) | -3.78 | 212.09 | 35.25 | 31.66 | 28.71 |
| CL (%) | 0.00 | 12.46 | 4.72 | 2.86 | 4.80 |

---

## 3. Cell-Level Split

**Train cells:** [1, 2, 3, 4, 5] (273 samples)  
**Validation cells:** [6, 7] (130 samples)  
**Test cells:** [8] (85 samples)

**Leakage audit:** PASS  
- train ∩ val: EMPTY
- train ∩ test: EMPTY
- val ∩ test: EMPTY

**Normalization:** All EIS features z-scored using TRAIN cells only.

---

## 4. Model Architecture

```
DegradationMLP:
  EIS features (9D)
      ↓
  Linear(9 → 64) + ReLU + Dropout(0.2)
      ↓
  Linear(64 → 64) + ReLU + Dropout(0.2)
      ↓
  Shared representation (64D)
      ├── Linear(64 → 1) → LLI
      ├── Linear(64 → 1) → LAM
      └── Linear(64 → 1) → CL
```

**Parameters:** 4,995  
**Loss:** MSE on all three targets (unweighted sum)  
**Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)  
**Early stopping:** Patience 15 on validation mean MAE  
**Best epoch:** 96  
**Training time:** 11.5s

---

## 5. Test Results

| Target | MAE | RMSE | R² |
|--------|-----|------|----|
| **LLI** | 52.36 | 74.77 | -0.0264 |
| **LAM** | 14.54 | 18.28 | 0.2670 |
| **CL** | 1.18 | 1.32 | 0.7926 |

**Interpretation:**
- LLI: Mean Absolute Error 52.36% (label range: -34.80 to 3516.42%)
- LAM: Mean Absolute Error 14.54% (label range: -3.78 to 212.09%)
- CL: Mean Absolute Error 1.18% (label range: 0 to 12.46%)

---

## 6. Limitations

1. **Very small test set:** Only 1 cell (85 samples) used for testing
2. **Labels are model-derived:** LLI/LAM/CL come from ECM fitting, NOT direct physical measurements
3. **No cross-validation:** Single train/val/test split due to limited cell count
4. **No frequency-aware encoding:** EIS features are aggregated scalars (no spectrum modeling)
5. **Extreme label variance:** LLI has very high variance (std=402.64%) relative to mean (208.68%)
6. **Negative values present:** LLI and LAM have negative values (likely ECM fitting artifacts)

---

## 7. Relation to Main BaFuse Project

**This experiment is COMPLETELY SEPARATE from the NASA A-H SOH benchmark.**

| Aspect | NASA SOH Benchmark | Degradation Quick Test |
|--------|-------------------|------------------------|
| Dataset | NASA PCoE (34 batteries) | Samsung/Mendeley (8 cells) |
| Input | Discharge + EIS + Physics | EIS only |
| Target | SOH (capacity fade) | LLI/LAM/CL (ECM-derived) |
| Models | A–H (14 architectures) | DegradationMLP (1 small MLP) |
| Status | Primary thesis results | Exploratory side experiment |

**NO data mixing:** Samsung data is NOT used to train NASA SOH models.  
**NO label transfer:** LLI/LAM/CL labels are NOT transferred to NASA dataset.

---

## 8. Conclusion

This quick test demonstrates that EIS features contain **some predictive signal** for the ECM-derived degradation-mode indicators (LLI/LAM/CL). However:

- Test set is too small (1 cell) for robust validation
- Labels are surrogate indicators from circuit fitting, not direct physical measurements
- Results are exploratory and NOT comparable to NASA SOH benchmark metrics

**Suitable for thesis as supporting evidence?** POSSIBLY — with explicit caveats about:
1. Model-derived labels (not ground truth)
2. Small test set (1 cell)
3. Separate experiment (not part of main BaFuse evaluation)

---

## 9. Saved Artifacts

- **Checkpoint:** `experiments/checkpoints/best_degradation_quick_test.pth`
- **This report:** `experiments/reports/degradation_quick_test.md`
- **Train normalization stats:** Saved in checkpoint under `train_stats`

---

## 10. Summary

- **Cells:** 8 (5 train / 2 val / 1 test)
- **Samples:** 488 (273 / 130 / 85)
- **LLI:** MAE=52.36  RMSE=74.77  R²=-0.0264
- **LAM:** MAE=14.54  RMSE=18.28  R²=0.2670
- **CL:** MAE=1.18  RMSE=1.32  R²=0.7926
- **Leakage:** PASS
- **SOH benchmark modified:** NO
