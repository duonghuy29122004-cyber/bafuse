# Samsung External Evaluation — Scientific Report

**Date:** 2026-09  
**Status:** CASE B — EIS-only evaluation. Full multimodal not possible.

---

## 1. Verdict: Full Multimodal Evaluation Is NOT Possible

Samsung/Mendeley does **not** provide discharge time-series (V/I/T/t).  
The NASA model requires discharge as a mandatory modality.  
**Voltage droop** (a physics feature) cannot be computed without discharge data.

Therefore, the full NASA multimodal architecture:

```
Discharge (B,100,3) + EIS (B,3) + Physics (B,4) → Fusion → SOH
```

**CANNOT** be evaluated on Samsung data.

---

## 2. What Can Be Evaluated

### Samsung EIS-Only External Test

Using only the EIS encoder branch:

```
Samsung EIS (B,8) → EIS Encoder → EIS Embedding (B,64) → Linear → SOH
```

This evaluates:
- Whether the EIS encoder can produce useful representations from Samsung EIS
- Whether EIS features correlate with SOH in a cross-dataset setting

This does **NOT** evaluate the full trained NASA model.

---

## 3. Critical Bugs Found and Fixed

### Bug 1: Wrong Normalization Statistics

**Old behavior:**
```python
nasa_stats_path = Path("data/mendeley_processed/mendeley_stats.json")
```
This loaded Samsung-derived statistics and applied them as normalization.

**Fix:**
- `create_dataloaders()` now saves NASA training stats to `data/processed/nasa_train_stats.json`
- `evaluate_samsung_external.py` loads from this file
- If file missing, recomputes from NASA `train.pkl`

### Bug 2: Zero-Padded Full Model Evaluation

**Old behavior:**
```python
disc_zero = torch.zeros(B, 100, 3)   # fabricated discharge
phys_zero = torch.zeros(B, 4)        # fabricated physics
out = model(disc_zero, eis_tensor, phys_zero)
```

Models were **not** trained with zero discharge or physics. This is out-of-distribution inference with no scientific basis. Results were meaningless.

**Fix:**
Removed entirely. Replaced with true EIS-encoder-only path.

### Bug 3: Results Mislabeled

**Old behavior:** Script labeled results as "EIS-only path" but ran full model.

**Fix:** Script now explicitly documents:
- `evaluation_type: "Samsung EIS-only external test (CASE B — not full multimodal)"`
- All caveats printed in the summary table

---

## 4. Domain Gap

| Feature | NASA mean | Samsung mean | Ratio | Impact |
|---------|-----------|-------------|-------|--------|
| EIS \|Z\| (Ohm) | 0.216 | 0.017 | 13× | Z-scores ~-10 to -13 |
| EIS Re (Ohm) | 0.209 | 0.017 | 13× | Out of training distribution |
| EIS \|Im\| (Ohm) | 0.016 | 0.002 | 9× | Out of training distribution |

Samsung cells are Samsung INR18650-30Q (2.95 Ah). NASA cells are 18650-type Li-ion (2.0 Ah). The impedance values differ systematically due to different cell chemistry, size, and measurement protocol. Applying NASA normalization statistics to Samsung values produces inputs far outside the training distribution.

**This is an inherent limitation of cross-dataset EIS comparison, not a bug.**

---

## 5. Evaluation Protocol Summary

| Item | Status |
|------|--------|
| Samsung used for training | NOT USED ✅ |
| Samsung used for validation | NOT USED ✅ |
| Samsung used for early stopping | NOT USED ✅ |
| Samsung used for model selection | NOT USED ✅ |
| Samsung used for normalization fitting | NOT USED ✅ |
| NASA stats used for normalization | YES ✅ (after fix) |
| Discharge fabricated | NOT FABRICATED ✅ (after fix) |
| Physics fabricated | NOT FABRICATED ✅ (after fix) |
| Results labeled correctly | YES ✅ (after fix) |
| Samsung SOH used only for eval metrics | YES ✅ |

---

## 6. Commands

```bash
# Step 1: Ensure NASA training stats are saved (run training first)
python run_pipeline.py --data_dir "5. BatteryDataSet" --epochs 50

# Step 2: Run Samsung external evaluation
python scripts/evaluate_samsung_external.py

# Step 3: With a checkpoint
python scripts/evaluate_samsung_external.py \
    --checkpoint checkpoints/best_model.pth \
    --out_dir results/samsung_external_eval
```

---

## 7. Scientific Limitations

1. Samsung discharge time-series (V/I/T) not available — full multimodal evaluation impossible
2. Samsung EIS absolute scale ~13× different from NASA — domain gap is large
3. Evaluation uses random linear head — not the trained NASA SOH head (which expects 128D fused input)
4. Only 1 Samsung test cell (cell 2, 60 samples) — insufficient for statistical significance
5. EIS features are aggregate per aging cycle — temporal detail lost

---

## 8. Recommended Language for Reports

**Correct:**
> "Samsung EIS-only external evaluation was conducted to assess EIS encoder transferability. Samsung cells were not used in any stage of model training, validation, or selection."

**Incorrect (do not use):**
> ~~"The full BaFuse model was externally validated on Samsung cells."~~
> ~~"Samsung provides independent confirmation of model performance."~~
