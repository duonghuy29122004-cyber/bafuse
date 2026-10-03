# Physical Degradation Analysis Audit

**Date:** 2026-09

---

## 1. Observable Signals Used

| Signal | Source | Computation | Notes |
|--------|--------|-------------|-------|
| `impedance_rise` | paired_df `impedance_ohm` | (Z - train_mean) / train_std | z-scored; BOL computed from train stats |
| `voltage_droop` | paired_df `voltage_min` | (Vmin - Vcutoff) / (4.2 - Vcutoff) | Per-battery cutoff from README |
| `cycle_age_norm` | paired_df `discharge_cycle` | cycle / N_ref | N_ref from power-law fit |
| `empirical_fade_prior` | cycle age only | A*(age/N_ref)^b | Population prior, no per-sample capacity |
| `capacity_fade` | SOH over cycles | 1 - SOH_current / SOH_BOL | Derivable from trajectory |
| `re_change` | `re_ohm` across cycles | relative change from BOL | In degradation analyzer |
| `rct_change` | `rct_ohm` across cycles | relative change from BOL | In degradation analyzer |

---

## 2. What Each Signal Measures

| Signal | What it measures | Physical interpretation |
|--------|-----------------|------------------------|
| `impedance_rise` | Increase in |Z| vs. training-population mean | May indicate SEI growth, electrolyte degradation, or increased contact resistance |
| `voltage_droop` | How far Vmin falls from cutoff voltage | May indicate capacity fade, active material loss, or increased internal resistance |
| `capacity_fade` | Reduction in dischargeable capacity vs. nominal | Direct measure of SOH decline; aggregates multiple mechanisms |
| `re_change` | Increase in real part of impedance (electrolyte resistance proxy) | May indicate electrolyte degradation, SEI growth, or contact resistance increase |
| `rct_change` | Increase in |Im(Z)| (charge-transfer resistance proxy) | May indicate interfacial degradation, active material loss, or passivation layer growth |
| `cycle_age_norm` | Normalized cycle count | Proxy for cumulative aging; does not identify mechanism |

---

## 3. What Can and Cannot Be Concluded

| Observation | What can be said | What CANNOT be said |
|-------------|-----------------|---------------------|
| Impedance rises over cycling | "Consistent with SEI growth or electrolyte degradation" | "Confirmed SEI growth mechanism" |
| Voltage droop increases | "Associated with capacity fade; may indicate active material loss" | "Particle cracking confirmed" |
| Re increases rapidly | "May indicate electrolyte resistance growth" | "Conductivity loss confirmed at X%" |
| All signals increase monotonically | "Consistent with progressive degradation" | "Specific mechanism identified without diagnostic testing" |

---

## 4. Audit Findings

### No LLI/LAM/CL ground truth fabricated
- NASA dataset does not contain LLI/LAM/CL labels
- No supervised degradation targets are used in training
- `BaFuseDataset.__getitem__()` returns only: discharge, eis, physics, soh_label, battery_id, cycle_idx
- No degradation labels appear in any model input

### Degradation analysis is post-training only
- `src/degradation_analysis.py` contains `PhysicalSignalExtractor` and `BatteryDegradationAnalyzer`
- These are called AFTER training, not during
- They use observable signals (impedance, voltage, capacity) from the data
- Results are fed to the RAG/literature layer for interpretation

### RAG layer is downstream interpretation only
- `rag/knowledge_base.py`: static literature entries, no training dependency
- `rag/retriever.py`: `DegradationExplainer.explain_from_signals()` maps observed changes to plausible mechanisms
- No mechanism label is used as a training signal
- Disclaimers are explicit in code and output

### Language used in code
- "consistent with"
- "may indicate"
- "possible contributing mechanism"
- "plausible candidates supported by published literature"

### Confirmed: No unsupported causal claims in code
- `src/degradation_analysis.py` uses `BatteryDegradationAnalyzer.analyze_battery()` which returns `"may indicate"` phrasing
- `rag/retriever.py` disclaimer: "presence cannot be confirmed without dedicated diagnostics"

---

## 5. Non-Issues

- `capacity_ahr` used in `fit_aging_prior()` — only from train data, only to fit population-level prior
- `voltage_min` used in `voltage_droop` — measured observable signal, not the target
- `impedance_ohm` used in `impedance_rise` AND `eis[0]` — redundancy noted; not leakage

---

## 6. Audit Verdict

**PASS.** Physical degradation analysis is implemented as a post-training interpretation layer using observable measured signals. No LLI/LAM/CL labels are fabricated, no degradation mechanism is claimed with certainty, and no degradation labels are used as supervised targets.
