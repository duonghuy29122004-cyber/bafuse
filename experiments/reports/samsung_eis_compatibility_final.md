# Samsung EIS Compatibility — Final Report

**Date:** 2026-09  
**Conclusion:** INCOMPATIBLE for quantitative comparison; EXPLORATORY at best

---

## 1. Samsung EIS Features Available

| Column | Description | Unit | Available |
|--------|-------------|------|-----------|
| `zmod_ohm` | EIS impedance magnitude |Z| | Ohm | YES |
| `zreal_ohm` | EIS real part Re(Z) | Ohm | YES |
| `zimg_ohm` | EIS imaginary part Im(Z) | Ohm | YES |
| `zphz_deg` | EIS phase angle | degrees | YES |
| `R_electrolyte` | ECM ohmic resistance (from fitting) | Ohm | YES |
| `R_ct1` | ECM charge-transfer resistance 1 | Ohm | YES |
| `R_ct2` | ECM charge-transfer resistance 2 | Ohm | YES |
| `Zw` | ECM Warburg coefficient | — | YES |

## 2. NASA EIS Features Expected by Models

| Feature | Source | Unit |
|---------|--------|------|
| `impedance_ohm` (eis[0]) | median |Z| across NASA EIS frequencies | Ohm |
| `re_ohm` (eis[1]) | median Re(Z) across NASA EIS frequencies | Ohm |
| `rct_ohm` (eis[2]) | median |Im(Z)| across NASA EIS frequencies | Ohm |

## 3. Scale Mismatch — Critical Finding

| Feature | NASA mean | Samsung mean | Ratio | z-score of Samsung vs NASA |
|---------|-----------|-------------|-------|---------------------------|
| |Z| (Ohm) | 0.216 | 0.017 | ~12.6x | **-6.8 sigma** |
| Re (Ohm) | 0.209 | 0.017 | ~12.3x | **-4.9 sigma** |
| |Im| (Ohm) | 0.016 | 0.002 | ~8x | ~-0.6 sigma |

The first two EIS features fall **6-7 standard deviations below** the NASA training distribution when Samsung values are z-scored with NASA statistics. This is far outside the range the models were trained on.

**Reason for scale difference:**
- Samsung INR18650-30Q: 2.95 Ah, 18650 cylindrical, different electrolyte and electrode composition
- NASA cells: 2.0 Ah, 18650 cylindrical, different chemistry and cycling protocol
- EIS impedance values are inherently cell-chemistry-dependent

## 4. Measurement Protocol Differences

| Aspect | NASA | Samsung |
|--------|------|---------|
| EIS measurement | Periodic during aging, various frequencies | At SOC=100% per aging cycle |
| Frequency representation | Median across spectrum (3 summary features) | Median of full spectrum (8 features after aggregation) |
| Representation type | Summary scalars (not full spectrum) | Summary scalars (not full spectrum) |

## 5. Feature Mapping Used in Evaluation

```
Samsung zmod_ohm  → NASA impedance statistics  (mapping: closest equivalent)
Samsung zreal_ohm → NASA re statistics         (mapping: closest equivalent)
Samsung zimg_ohm  → NASA rct statistics        (mapping: closest equivalent)
```

This mapping is the best available approximation, but:
- Physical quantities are not identical
- Absolute scale differs 5-13x
- Samsung values produce severely out-of-distribution z-scores

## 6. Conclusion

**INCOMPATIBLE** for quantitative accuracy comparison.

**EXPLORATORY** use only:
- Can observe whether models produce any SOH-related variation across Samsung aging cycles
- Can assess whether EIS encoder architecture captures some structure in Samsung EIS
- Cannot claim the NASA training distribution applies to Samsung data
- Cannot interpret Samsung MAE/RMSE values as equivalent to NASA test performance

**Required disclaimer in any report using these results:**
> "Samsung EIS features are systematically ~12x smaller in absolute scale compared to NASA training data. After applying NASA z-score normalization, Samsung values fall 5-7 standard deviations below the training distribution. Reported Samsung EIS-only results are exploratory and cannot be directly compared to NASA internal test metrics."
