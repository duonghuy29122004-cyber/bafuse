# Samsung/Mendeley Compatibility Audit

**Date:** 2026-09  
**Auditor:** BaFuse project automated inspection (`scripts/inspect_samsung_data.py`)  
**Purpose:** Determine which NASA model modalities can be evaluated on Samsung data

---

## 1. Samsung Dataset Facts

**Dataset:** "Li-ion cells EIS dataset with fitting and degradation modes" (Mendeley)  
**Cells:** 8 Samsung INR18650-30Q cells, nominal capacity 2.95 Ah  
**Measurements per cycle:** EIS at SOC=100%, one row per (cell_id, aging_cycle)  
**Test split used:** Cell 2 only (60 samples, aging_cycle 0–275)

### Actual columns in `mendeley_test.pkl`

```
cell_id, aging_cycle, zmod_ohm, zphz_deg, zreal_ohm, zimg_ohm,
soh_pct, ocv_v, lam_pct, lli_pct, cl_pct,
R_electrolyte, R_ct1, R_ct2, Zw, soh_norm
```

### What Samsung DOES provide

| Column | Description | Available |
|--------|-------------|-----------|
| `aging_cycle` | Cycle index (0–400) | YES |
| `zmod_ohm` | EIS impedance magnitude (Ohm) | YES |
| `zphz_deg` | EIS phase angle (degrees) | YES |
| `zreal_ohm` | EIS real part Re(Z) (Ohm) | YES |
| `zimg_ohm` | EIS imaginary part Im(Z) (Ohm) | YES |
| `R_electrolyte` | ECM ohmic resistance (Ohm) | YES |
| `R_ct1` | ECM charge-transfer resistance 1 (Ohm) | YES |
| `R_ct2` | ECM charge-transfer resistance 2 (Ohm) | YES |
| `Zw` | ECM Warburg coefficient | YES |
| `ocv_v` | Open-circuit voltage at SOC=100% | YES |
| `soh_pct` / `soh_norm` | State-of-health label | YES (eval-only) |

### What Samsung DOES NOT provide

| Column | Description | Available |
|--------|-------------|-----------|
| `voltage_v` | Discharge voltage time-series | **MISSING** |
| `current_a` | Discharge current time-series | **MISSING** |
| `temperature_c` | Temperature time-series | **MISSING** |
| `time_s` | Time time-series | **MISSING** |
| `voltage_min` | Minimum discharge voltage | **MISSING** |
| `voltage_mean` | Mean discharge voltage | **MISSING** |

---

## 2. NASA Model Input Requirements

The NASA-trained models use three input modalities:

### Discharge (B, 100, 3) — time-series
- Channel 0: Voltage (V), z-scored with NASA `stats['voltage']`
- Channel 1: Current (A), z-scored with NASA `stats['current']`
- Channel 2: Temperature (°C), z-scored with NASA `stats['temperature']`
- **Source:** discharge_raw.pkl, resampled to 100 time-steps

### EIS (B, 3) — scalar summary features
- Feature 0: `impedance_ohm` — |Z| median (Ohm), z-scored with NASA `stats['impedance']`  
  NASA range: **0.01–0.35 Ohm** (mean≈0.216)
- Feature 1: `re_ohm` — Re(Z) median (Ohm), z-scored with NASA `stats['re']`  
  NASA range: **0.01–0.35 Ohm** (mean≈0.209)
- Feature 2: `rct_ohm` — |Im(Z)| median (Ohm), z-scored with NASA `stats['rct']`  
  NASA range: **0.001–0.10 Ohm** (mean≈0.016)

### Physics (B, 4) — engineered features
- Feature 0: `cycle_age_norm` = `discharge_cycle / N_ref`
- Feature 1: `empirical_fade_prior` = `A * (age/N_ref)^b` — population-level prior
- Feature 2: `voltage_droop` = `(voltage_min - cutoff_v) / (4.2 - cutoff_v)`  
  **Requires `voltage_min` from discharge — NOT available in Samsung**
- Feature 3: `impedance_rise` = `(impedance_ohm - mean_imp) / std_imp`

---

## 3. Compatibility Table

| Modality | NASA | Samsung | Compatible? | Reason |
|----------|------|---------|-------------|--------|
| **Discharge V/I/T** | YES | **NO** | **NO** | Samsung has no discharge time-series |
| **Discharge time shape** | (B,100,3) | N/A | **NO** | Cannot be constructed |
| **EIS zmod_ohm** | YES | YES | PARTIAL | Scale mismatch: Samsung ~0.017 Ohm vs NASA ~0.216 Ohm |
| **EIS zreal_ohm** | YES | YES | PARTIAL | Same scale mismatch issue |
| **EIS zimg_ohm** | YES | YES | PARTIAL | Same scale mismatch issue |
| **EIS zphz_deg** | YES | YES | PARTIAL | Different physical context |
| **Physics cycle_age** | YES | YES (aging_cycle) | YES | Directly available |
| **Physics voltage_droop** | YES | **NO** | **NO** | Requires voltage_min from discharge |
| **Physics imp_rise** | YES | PARTIAL | PARTIAL | Can approximate from EIS if BOL defined |
| **Physics fade_prior** | YES | PARTIAL | PARTIAL | Can use cycle age only |
| **SOH label** | YES | YES | YES (eval-only) | Used only after prediction for metrics |

---

## 4. EIS Scale Mismatch — Critical Issue

The NASA EIS values and Samsung EIS values are on completely different absolute scales:

| Feature | NASA mean (Ohm) | Samsung mean (Ohm) | Ratio |
|---------|-----------------|-------------------|-------|
| `zmod_ohm` / `impedance_ohm` | 0.2163 | 0.0167 | 13× |
| `zreal_ohm` / `re_ohm` | 0.2090 | 0.0166 | 13× |
| `zimg_ohm` / `rct_ohm` | 0.0164 | 0.0019 | 9× |

**This is a fundamental domain gap** due to different cell sizes, chemistries, and measurement protocols.  
Even if EIS features could be mapped, applying NASA normalization statistics to Samsung values would produce out-of-distribution inputs far outside the training range.

---

## 5. Current Implementation Issue — Identified Bug

In `evaluate_samsung_external.py`, `evaluate_with_benchmark_model()`:

```python
disc_zero  = torch.zeros(len(samsung_df), 100, 3, device=device)   # PROBLEMATIC
phys_zero  = torch.zeros(len(samsung_df), 4, device=device)         # PROBLEMATIC
```

**These zeros are NOT a valid representation of missing data for these models.**  
The models were trained with non-zero discharge and physics inputs. Passing zeros is an artificial input that was never seen during training. This is not a legitimate missing-modality evaluation — it is untested out-of-distribution inference with no scientific basis.

**Additionally:**
```python
nasa_stats_path = Path("data/mendeley_processed/mendeley_stats.json")
```
This loads **Samsung-derived** normalization statistics, not NASA statistics. This violates the protocol: Samsung data must not influence normalization used for model evaluation.

---

## 6. Verdict

**CASE B applies:** Samsung CANNOT support full multimodal evaluation of the NASA model.

The following modalities are definitively unavailable:
- Discharge time-series (V/I/T) — **MISSING**, cannot be fabricated
- Voltage droop physics feature — **MISSING**, depends on discharge data

The following evaluation is scientifically valid:
- **Samsung EIS-only evaluation** using an EIS-only sub-path (not the full multimodal model)
- This must be explicitly labeled as "Samsung EIS-only external evaluation"
- It must NOT be labeled as "Full BaFuse external validation"

---

## 7. Required Corrections

1. Remove `disc_zero` and `phys_zero` from full-model evaluation — invalid
2. Remove use of `mendeley_stats.json` for normalization — wrong stats source
3. Save NASA training stats in a dedicated file during training
4. Implement EIS-only evaluation path using only the EIS encoder branch
5. Clearly label all results as "EIS-only" in all reports and output files
