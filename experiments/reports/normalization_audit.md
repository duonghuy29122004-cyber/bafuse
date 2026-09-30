# Normalization Audit

**Date:** 2026-09  
**Protocol requirement:** NASA train statistics must be used for ALL splits (NASA val, NASA test, Samsung external test). Samsung statistics must NEVER be used for model evaluation.

---

## 1. NASA Normalization Statistics

Computed by `BaFuseDataset._compute_normalization_stats()` from `train.pkl`:

| Key | Mean | Std | Used for |
|-----|------|-----|---------|
| `voltage` | 3.4293 | 0.1126 | Discharge V channel |
| `current` | -1.7763 | 0.4844 | Discharge I channel |
| `temperature` | 29.1589 | 10.1273 | Discharge T channel |
| `impedance` | 0.2163 | 0.0293 | EIS feature 0 (|Z|) |
| `re` | 0.2090 | 0.0394 | EIS feature 1 (Re) |
| `rct` | 0.0164 | 0.0237 | EIS feature 2 (|Im|) |
| `capacity` | 1.4766 | 0.2909 | (internal only) |

**Status:** These stats are computed from NASA train only. ✅  
**Where stored:** In `train_dataset.stats` dict in memory during training. **NOT persisted to disk during standard training.**

### Problem identified

The current `evaluate_samsung_external.py` loads:
```python
nasa_stats_path = Path("data/mendeley_processed/mendeley_stats.json")
```
This is **Samsung-derived** statistics. The script itself logs a warning:
> "Note: these are Mendeley stats, not NASA."

This is a confirmed normalization protocol violation.

---

## 2. Samsung (Mendeley) Normalization Statistics

Computed by `compute_mendeley_stats()` from Samsung train cells:

| Key | Mean (Ohm) | Std (Ohm) | Notes |
|-----|----------|---------|-------|
| `zmod_ohm` | 0.0167 | 0.0007 | ~13x smaller than NASA impedance |
| `zreal_ohm` | 0.0166 | 0.0007 | ~13x smaller than NASA re |
| `zimg_ohm` | 0.0019 | 0.0005 | ~9x smaller than NASA rct |

**Status:** MUST NOT be used for normalization in model evaluation. ❌

---

## 3. Correct Normalization Flow

```
NASA train.pkl
      |
      v
BaFuseDataset._compute_normalization_stats()
      |
      v
NASA training stats (freeze)
      |
      +------> NASA val/test  (apply NASA stats)   ✅
      |
      +------> Samsung EIS-only eval               ⚠️ Problem: NASA stats
                                                    use 'impedance' key (~0.216 Ohm)
                                                    Samsung zmod_ohm is ~0.017 Ohm
                                                    — fundamentally different scale
```

### Issue: EIS key name mismatch

NASA stats use keys: `'impedance'`, `'re'`, `'rct'`  
Samsung EIS columns: `'zmod_ohm'`, `'zreal_ohm'`, `'zimg_ohm'`, `'zphz_deg'`, `'R_electrolyte'`, `'R_ct1'`, `'R_ct2'`, `'Zw'`

These are different physical quantities measured by different instruments with different protocols. Applying NASA z-score stats to Samsung values would produce inputs in the range `[-25, -10]` (far outside [−3, 3] training distribution).

**Conclusion:** EIS cross-dataset normalization is inherently problematic due to scale mismatch. The external evaluation must document this limitation explicitly.

---

## 4. Required Fix: Save NASA Stats During Training

Add to `run_pipeline.py` and `scripts/train_bafuse_v2.py`:

```python
# After create_dataloaders()
import json
nasa_stats = train_loader.dataset.stats
with open("data/processed/nasa_train_stats.json", "w") as f:
    json.dump(nasa_stats, f, indent=2)
```

This allows `evaluate_samsung_external.py` to load the correct stats.

---

## 5. Normalization Audit Checklist

| Check | Status | Notes |
|-------|--------|-------|
| NASA train stats computed from train split only | ✅ | `_compute_normalization_stats()` |
| NASA val stats use train stats | ✅ | `external_stats=train_dataset.stats` |
| NASA test stats use train stats | ✅ | `external_stats=train_dataset.stats` |
| Samsung not used for normalization fitting | ✅ | Never called `_compute_normalization_stats()` on Samsung |
| Samsung stats (`mendeley_stats.json`) NOT used for evaluation | ❌ | **BUG: current `evaluate_samsung_external.py` loads mendeley_stats.json** |
| NASA stats persisted to disk for external eval | ❌ | **MISSING: not saved during training** |
| EIS scale compatibility acknowledged | ❌ | **NOT documented in current script** |
