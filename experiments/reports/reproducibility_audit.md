# Reproducibility Audit

**Date:** 2026-09

---

## 1. Random Seeds

| Location | Value | Status |
|----------|-------|--------|
| `configs/config_bafuse_v2.yaml`: `seed: 42` | 42 | EXISTS |
| `scripts/architecture_benchmark.py`: `--seeds 42` default | 42 | EXISTS |
| `src/data/split.py`: `random_state=42` default | 42 | EXISTS |
| Multi-seed runs (42, 123, 2026) | planned | PARTIALLY DONE (A-H only) |

---

## 2. Configuration Files

| File | Status | Notes |
|------|--------|-------|
| `configs/config.yaml` | EXISTS | v1 backward-compat config |
| `configs/config_bafuse_v2.yaml` | EXISTS | v2 primary config, training_mode=nasa_only |

---

## 3. Normalization Statistics

| File | Status | Contents |
|------|--------|---------|
| `data/processed/nasa_train_stats.json` | EXISTS | 7 keys: voltage, current, temperature, impedance, capacity, re, rct |
| `data/mendeley_processed/mendeley_stats.json` | EXISTS | Samsung analysis stats (NOT used for model normalization) |

---

## 4. Checkpoints

| File | Status | Notes |
|------|--------|-------|
| `checkpoints/best_model.pth` | EXISTS | Last trained checkpoint |
| `experiments/degradation/best_degradation.pth` | EXISTS | Stage-1 Mendeley checkpoint (DEPRECATED under new direction) |

---

## 5. Data Split Files

| File | Status |
|------|--------|
| `data/processed/train.pkl` | EXISTS |
| `data/processed/val.pkl` | EXISTS |
| `data/processed/test.pkl` | EXISTS |
| `data/processed/paired.pkl` | EXISTS |
| `data/processed/discharge_raw.pkl` | EXISTS |
| `data/mendeley_processed/mendeley_test.pkl` | EXISTS |

---

## 6. Package Requirements

| Package | Required | Version in requirements.txt | Notes |
|---------|----------|------------------------------|-------|
| torch | YES | >=2.0.0 | Core ML |
| numpy | YES | >=1.23.0 | Arrays |
| pandas | YES | >=1.5.0 | Data |
| scipy | YES | >=1.10.0 | Stats, KS test |
| scikit-learn | YES | >=1.2.0 | Split, metrics |
| matplotlib | YES | >=3.7.0 | Plots |
| pyyaml | YES | >=6.0 | Config |
| openpyxl | YES | >=3.1.0 | Samsung Excel parsing |
| sympy | YES | >=1.12 | Torch dependency |
| streamlit | Optional | >=1.20.0 | Dashboard only |

---

## 7. Model Configuration

All 14 benchmark models defined in `src/models/benchmark_models.py` with explicit `get_model(name)` registry. Model architectures are deterministic given input dimensions (d_in=3, e_in=3, p_in=4). Parameter counts verified:

| Model | Params |
|-------|--------|
| A | 3,431 |
| B | 16,545 |
| C | 24,193 |
| D | 44,257 |
| E | 30,977 |
| E2 | 85,505 |
| F | 18,148 |
| G | 24,964 |
| C2 | 21,409 |
| D2 | 48,225 |
| E3 | 57,409 |
| E4 | 127,553 |
| E5 | 54,625 |
| G2 | 20,132 |

---

## 8. Output Report Locations

| Report | Path |
|--------|------|
| Benchmark summary | `results/architecture_benchmark/benchmark_summary.csv` |
| Samsung external eval | `results/samsung_external_eval/samsung_external_eval.json` |
| Cross-dataset comparison | `results/cross_dataset/all_architectures_comparison.csv` |
| Samsung compatibility audit | `experiments/reports/samsung_compatibility_audit.md` |
| Normalization audit | `experiments/reports/normalization_audit.md` |
| External compatibility | `experiments/reports/external_model_compatibility_final.csv` |
| Thesis protocol | `experiments/reports/thesis_evaluation_protocol.md` |
| Physical degradation audit | `experiments/reports/physical_degradation_audit.md` |
| Final audit | `experiments/reports/FINAL_SCIENTIFIC_AUDIT.md` |

---

## 9. Missing Items / Open Reproducibility Gaps

| Item | Status | Priority |
|------|--------|---------|
| Multi-seed (42, 123, 2026) for all 14 models | NOT DONE (seed=42 only) | HIGH |
| Dataset version/hash documentation | MISSING | MEDIUM |
| Exact NASA PCoE dataset download URL | In README | LOW |
| Conda/pip environment lock file | Not present (`requirements.txt` only) | MEDIUM |
| Checkpoint for each of the 14 models | Only latest best_model.pth | HIGH |

---

## 10. Reproducibility Verdict

**PARTIAL PASS.** The core pipeline (data, split, normalization, training) is reproducible. Key gaps:
- Multi-seed results not yet complete
- Per-model checkpoints not saved (only one best_model.pth)
- No environment lock file (requirements.txt has version ranges only)

Another researcher following the README can reproduce the basic split and training, but cannot reproduce the exact reported numbers without the same checkpoint and seed.
