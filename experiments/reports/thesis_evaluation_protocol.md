# Thesis Evaluation Protocol

**Project:** BaFuse — Battery Fusion SOH Estimation  
**Date:** 2026-09

---

## 1. NASA Evaluation Protocol

**Dataset:** NASA PCoE Battery Aging Dataset (34 batteries, B0005–B0056)  
**Nominal capacity:** 2.0 Ah (18650 Li-ion, confirmed from README files)  
**Split:** Battery-level stratified split (60/20/20)

| Split | Batteries | Samples |
|-------|-----------|---------|
| Train | 20 batteries | 616 |
| Validation | 7 batteries | 222 |
| Test | 7 batteries | 97 |

**Verification:** No battery appears in more than one split. No duplicate (battery, cycle) pairs across splits. Verified programmatically.

**Model inputs:**
- Discharge: V/I/T time-series, resampled to 100 timesteps, 3 channels, z-scored with train statistics
- EIS: 3 scalar features [|Z|, Re, Rct], z-scored with train statistics  
- Physics: 4 engineered features [cycle_age_norm, empirical_fade_prior, voltage_droop, impedance_rise]

**Normalization:** All statistics computed from train split only. Validation and test use frozen train statistics. NASA train statistics saved to `data/processed/nasa_train_stats.json`.

**Target:** SOH = capacity_ahr / nominal_capacity ∈ [0, 1]. Values slightly above 1.0 (3 early-cycle samples) are clipped to 1.0 — this is physical behavior (break-in effect), not a preprocessing error.

**Loss (benchmark):** MSE only.  
**Metrics:** MAE%, RMSE% (×100 scale factor from [0,1]), R².  
**Seeds:** 42 (primary). Multi-seed planned for top architectures.  
**Early stopping:** Patience 15 on validation MAE.

---

## 2. Samsung External Evaluation Protocol

**Dataset:** Samsung INR18650-30Q cells (8 cells, 2.95 Ah nominal)  
**Source:** "Li-ion cells EIS dataset with fitting and degradation modes" (Mendeley Data)  
**Use:** EXTERNAL TEST ONLY — never used in training, validation, normalization, or model selection.

**What Samsung provides:**
- EIS measurements (8 features per aging cycle): Zmod, Zphz, Zreal, Zimg, R_electrolyte, R_ct1, R_ct2, Zw
- SOH label (from soh_pct / 100)
- Aging cycle index

**What Samsung does NOT provide:**
- Discharge time-series (V/I/T) — absent from this dataset
- Physics features that depend on discharge (voltage_droop, voltage_min)

**Consequence:** The full NASA multimodal model cannot be evaluated on Samsung. CASE B applies.

**What is evaluated:** EIS encoder sub-path only, using 3 Samsung EIS features mapped to NASA EIS features.

**Normalization for Samsung:** NASA training statistics from `nasa_train_stats.json` are applied. Samsung values are systematically ~12x smaller than NASA values, producing out-of-distribution z-scores (~-5 to -7 sigma from NASA training distribution).

**Current limitation:** The `evaluate_eis_only()` function uses randomly initialized downstream heads (no saved checkpoint is loaded per model). Reported metrics reflect EIS encoder structural capacity, not trained NASA model performance.

---

## 3. Why Samsung is EIS-Only

Samsung cells were measured using EIS-only protocol (no discharge time-series recorded). The NASA multimodal architecture requires:
1. Discharge V/I/T time-series → DischargeEncoder (LSTM/CNN/TCN)
2. EIS features → EISEncoder (MLP)
3. Physics features (including voltage_droop from discharge data) → PhysicsEncoder

Without discharge data, modalities 1 and 3 cannot be provided. Zero-padding would be scientifically invalid because models were not trained with zero inputs. Therefore, only the EIS encoder branch can be used.

---

## 4. What Samsung Results Can and Cannot Demonstrate

**CAN demonstrate:**
- Whether the EIS encoder produces any coherent representation of Samsung aging
- Whether cross-dataset EIS structure is partially preserved
- Approximate trend analysis (whether predictions correlate with aging cycle)

**CANNOT demonstrate:**
- Full model generalization (discharge/physics branches not evaluated)
- Quantitative accuracy comparable to NASA test metrics
- That the trained NASA model achieves a specific MAE on Samsung data (no checkpoint loaded in current implementation)
- That NASA-trained models generalize to Samsung chemistry (scale mismatch prevents this claim)

---

## 5. Recommended Thesis Wording

### For NASA results:
> "Evaluation was performed on NASA PCoE Battery Aging Dataset using battery-level split. Models were trained exclusively on training batteries [list] and evaluated on held-out test batteries [list]. Normalization statistics were computed from training data only. Full multimodal input (discharge V/I/T, EIS, physics-informed features) was used. NASA test results: [table]."

### For Samsung results:
> "An independent external assessment was conducted on Samsung INR18650-30Q cells from the Mendeley dataset. Samsung data was not used at any stage of training, validation, or model selection. Because Samsung does not provide discharge time-series compatible with the NASA training pipeline, evaluation was limited to the EIS encoder sub-path (EIS-only, Case B). Samsung EIS values are systematically ~12x smaller in absolute scale than NASA training data, producing out-of-distribution inputs when NASA normalization statistics are applied. The reported Samsung results are therefore exploratory and cannot be directly compared to NASA test metrics. Samsung external results: [table with explicit 'EIS-only' and 'exploratory' labels]."

### Combining in a single table:
```
| Dataset  | Eval Type              | Model | MAE%  | RMSE% | R²    | Note                    |
|----------|------------------------|-------|-------|-------|-------|-------------------------|
| NASA     | Full multimodal        | D2    | 4.23  | 6.04  | 0.856 | Trained on NASA train   |
| NASA     | Full multimodal        | G     | 5.43  | 7.22  | 0.794 | Trained on NASA train   |
| Samsung  | EIS-only (exploratory) | all   | N/A   | N/A   | N/A   | See caveats in Sec. X   |
```

---

## 6. Limitations

1. **Small test set:** Only 7 test batteries (97 samples). Results may not generalize to all NASA battery types.
2. **SOH distribution shift:** Test set mean SOH (66.5%) differs from train mean (73.8%) by 7.3 percentage points. Models may perform differently in degraded-SOH regime.
3. **Samsung external:** EIS-only, scale mismatch, random evaluation head — not comparable to NASA metrics.
4. **Single test cell:** Samsung evaluation uses only cell 2 (60 samples) — insufficient for statistical significance.
5. **Impedance redundancy:** Physics feature `impedance_rise` is mathematically identical to `eis[0]`. This represents a minor input redundancy that inflates apparent physics feature diversity.
6. **No multi-seed results reported** for all 14 models — only seed=42. Multi-seed analysis planned.
