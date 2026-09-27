# BaFuse v2 — Battery Fusion Estimation

**Multimodal multi-task battery State-of-Health estimation**  
**NASA PCoE (discharge + EIS + physics -> SOH) + Mendeley EIS (-> LLI/LAM/CL)**

> Version: v2.1 | Smoke-test: 17/17 PASS | Python 3.10 | PyTorch

---

## Overview

BaFuse combines three complementary battery signals to estimate State-of-Health (SOH),
and additionally investigates battery degradation modes using a second dataset.

| Task | Dataset | Input | Output |
|------|---------|-------|--------|
| SOH regression | NASA PCoE (34 batteries, B0005-B0056) | Discharge + EIS + Physics | SOH ∈ [0,1] |
| Degradation mode estimation | Mendeley (8 Samsung INR18650-30Q cells) | EIS only | LLI / LAM / CL (model-derived estimates) |

**IMPORTANT:** LLI/LAM/CL labels are model-derived from equivalent-circuit fitting,
not physical ground truth. All results refer to them as "model-derived degradation-mode estimates".

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess Mendeley dataset
python scripts/preprocess_mendeley.py

# 3. Stage-1: pre-train EIS encoder on Mendeley
python scripts/train_degradation.py --epochs 50

# 4. Stage-2: train full BaFuseV2 on NASA (staged mode)
python scripts/train_bafuse_v2.py --training_mode staged --epochs 100

# 5. (Optional) Original NASA-only pipeline — backward compatible
python run_pipeline.py --data_dir "5. BatteryDataSet" --epochs 50
```

---

## Project Structure

```
bafuse/
├── 5. BatteryDataSet/                  # NASA PCoE dataset (.mat files)
│   ├── 1. BatteryAgingARC-FY08Q4/      # B0005-B0007, B0018
│   ├── 2. BatteryAgingARC_25_26_27_28_P1/
│   ├── 3. BatteryAgingARC_25-44/
│   ├── 4. BatteryAgingARC_45_46_47_48/
│   ├── 5. BatteryAgingARC_49_50_51_52/
│   └── 6. BatteryAgingARC_53_54_55_56/
├── Lithium-ion cells EIS dataset.../   # Mendeley dataset (Excel files)
│   ├── ExperimentalDATA/               # EISexpCell01.xlsx ... Cell08.xlsx
│   └── OutputDATA/                     # Circuit_parameter_Cell01.xlsx ...
├── src/
│   ├── data/
│   │   ├── parse_mat.py                NASA .mat parser
│   │   ├── pairing.py                  Discharge-EIS pairing
│   │   ├── split.py                    Battery-level stratified split
│   │   ├── dataset.py                  BaFuseDataset (PyTorch)
│   │   └── mendeley_dataset.py         Mendeley parser + MendeleyDataset
│   ├── models/
│   │   ├── encoders.py                 DischargeEncoder, EISEncoder, PhysicsEncoder
│   │   ├── fusion.py                   CrossAttentionFusion, WeightedFusion, ConcatFusion
│   │   ├── bafuse.py                   BaFuse v1 (backward compatible)
│   │   ├── bafuse_v2.py                BaFuseV2 (multi-task + ablation flags)
│   │   └── degradation_head.py         DegradationHead (LLI/LAM/CL)
│   ├── training/
│   │   ├── degradation_trainer.py      Stage-1 Mendeley pre-training
│   │   └── multitask_trainer.py        Stage-2/3 interleaved joint training
│   ├── losses.py                       SoHPredictionLoss, DegradationLoss, MultiTaskLoss
│   ├── train.py                        Original training loop (backward compat)
│   ├── evaluate.py                     Test evaluation
│   └── visualization.py                7 plot functions
├── experiments/
│   ├── run_ablation.py                 A1-A7 ablation runner
│   └── evaluate_ablation.py            Ablation summary + plots
├── evaluation/
│   ├── metrics.py                      MAE / RMSE / R²
│   └── ablation_metrics.py             AblationAnalyzer, delta-RMSE contribution
├── rag/
│   ├── knowledge_base.py               9 degradation mechanism entries (3 per mode)
│   └── retriever.py                    DegradationExplainer (downstream only)
├── configs/
│   ├── config.yaml                     v1 config (backward compat)
│   └── config_bafuse_v2.yaml           v2 config
├── scripts/
│   ├── preprocess_mendeley.py          Mendeley preprocessing
│   ├── train_degradation.py            Stage-1 training
│   ├── train_bafuse_v2.py              Stage-2/3 training
│   ├── smoke_test.py                   17-check regression test
│   └── audit_splits.py                 Split overlap + baseline audit
└── app/
    └── dashboard.py                    Streamlit dashboard
```

---

## Model Architecture

### BaFuseV2

```
NASA PCoE
    |
    +------------------+------------------+
    |                  |                  |
Discharge (B,100,3)  EIS_NASA (B,3)  Physics (B,4)
    |                  |                  |
DischargeEncoder    EISEncoder(3)   PhysicsEncoder
(LSTM 3-layer)      (MLP)           (MLP 3-layer)
    |                  |                  |
   (B,64)            (B,64)            (B,64)
    |                  |                  |
    +------------------+------------------+
                       |
            CrossAttentionFusion (4 heads)
                       |
                   fused (B,128)
                       |
             +---------+---------+
             |                   |
          SOH Head         Degradation Head
          (MLP 3-layer)    (MLP, input=128)
             |                   |
         soh_pred (B,1)   [LLI, LAM, CL] (B,3)
                                 ^
                                 |
               Mendeley EIS (B,8)
               -> mendeley_eis_encoder(8)
               -> degradation_head_eis_only
```

**Trainable params:** ~1,536,711

### Modality flags for ablation (eval-time zero-masking)
```python
model.set_modality_flags(use_discharge=False, use_eis=True, use_physics=True)
```

---

## Datasets

### NASA PCoE Battery Dataset

- **Cells:** 34 batteries (B0005-B0056), 18650-type Li-ion, 2.0 Ah nominal
- **Campaigns:** 6 aging test campaigns, different cutoff voltages
- **Cycles:** ~168 cycles per battery on average
- **Split:** 20 train / 7 val / 7 test (battery-level, no overlap)
- **Source:** https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-center/publications/

```
Split statistics:
  Train: 616 samples, 20 batteries, mean SOH = 73.8%
  Val:   222 samples,  7 batteries, mean SOH = 75.7%
  Test:   97 samples,  7 batteries, mean SOH = 66.5%
```

### Mendeley EIS Degradation Dataset

- **Cells:** 8 Samsung INR18650-30Q cells, 2.95 Ah nominal
- **Protocol:** EIS measurements at SOC=100% across aging cycles
- **Labels:** LLI, LAM, CL — model-derived from equivalent-circuit fitting
- **Split:** 6 train / 1 val / 1 test (cell-level)
- **Source:** https://data.mendeley.com

---

## Features

### Discharge features (per sample)
After z-score normalisation, resampled to 100 time-steps:
- Voltage_norm, Current_norm, Temperature_norm

### EIS features (NASA)
All three z-scored with train statistics:
- `|Z|_norm` — impedance magnitude
- `Re_norm`  — electrolyte resistance proxy (median Re across frequencies)
- `Rct_norm` — charge-transfer proxy (median |Im| across frequencies)

> Note: Re and Rct are engineered proxy features, not true impedance spectroscopy
> quantities (true Re/Rct require high-frequency Nyquist intercept / ECM fitting).

### Physics features (4D, leak-free)
| Feature | Formula |
|---------|---------|
| `cycle_age_norm` | `cycle / N_ref` (N_ref = 90th percentile of train ages) |
| `empirical_fade_prior` | `A * (cycle/N_ref)^b` — power-law fit on TRAIN population only |
| `voltage_droop` | `(Vmin - Vcutoff) / (4.2 - Vcutoff)` — per-battery cutoff |
| `impedance_rise` | `(Z - Z_mean) / Z_std` — z-scored with train stats |

### SOH label
```
SOH = capacity_ahr / BATTERY_NOMINAL_CAPACITY[battery_id]
    ∈ [0, 1]   (clipped)

Reporting: MAE% = MAE * 100,  RMSE% = RMSE * 100
```

---

## Training

### Training modes

| Mode | Description |
|------|-------------|
| `nasa_only` | SOH only, no degradation (original BaFuse behaviour) |
| `staged` | Load Stage-1 EIS weights, then joint NASA+Mendeley training |
| `joint` | Both tasks from scratch simultaneously |

### Multi-task loss (staged/joint)
```
L = lambda_soh * L_soh   +   lambda_deg * L_deg
  =     1.0    * MSE(SOH)  +      0.5   * weighted_MSE(LLI, LAM, CL)
```

Batches from both loaders are **interleaved per step** (not sequential per epoch).
Single `backward()` + `optimizer.step()` combines both gradients.

### Optimizer & schedule
```
Optimizer:    Adam (lr=5e-4, weight_decay=1e-5)
LR schedule:  Linear warmup (5 epochs) + Cosine annealing
Early stop:   patience=15 on val MAE
Grad clip:    max_norm=1.0
```

---

## Ablation Study

Seven configurations trained independently with the same split and seed:

| ID | Modalities | Description |
|----|-----------|-------------|
| A1 | D + E + P | Full model (baseline) |
| A2 | E + P | No discharge |
| A3 | D + P | No EIS |
| A4 | D + E | No physics |
| A5 | D | Discharge only |
| A6 | E | EIS only |
| A7 | P | Physics only |

Contribution score (delta-RMSE, explicitly defined):
```
contribution_pct(m) = 100 * (RMSE_without_m - RMSE_full) / sum(RMSE_without_* - RMSE_full)
```
This is an ablation-based contribution, NOT causal importance.

```bash
python experiments/run_ablation.py --config configs/config_bafuse_v2.yaml
python experiments/evaluate_ablation.py
```

---

## RAG Explanation

The RAG layer is a **downstream explanation component only** — it does not affect training or predictions.

```python
from rag.retriever import DegradationExplainer

explainer = DegradationExplainer(top_k=2)
result = explainer.explain({"LLI": 32.5, "LAM": 15.0, "CL": 2.1})
print(explainer.format_explanation(result))
```

Output distinguishes:
1. Estimated degradation-mode value
2. Plausible mechanisms (from knowledge base)
3. Supporting literature (with citations)
4. Explicit disclaimer: model-derived estimates, not ground truth

---

## Current Results

### Baselines (constant mean predictor)
| | Val | Test |
|---|---|---|
| MAE | 9.22% | 13.86% |
| RMSE | 11.63% | 17.92% |
| R² | 0.000 | 0.000 |

### BaFuseV2 staged training (best val checkpoint)
| | Val |
|---|---|
| MAE | ~4.9% |
| RMSE | ~7.2% |
| R² | ~0.59 |

*Model beats mean predictor baseline: MAE -47%, R² 0 -> 0.59.*

### v1 5-fold CV (for reference)

| Config | MAE | Std | R² |
|--------|-----|-----|-----|
| (a) baseline | 7.947% | 2.226% | 0.174 |
| (b) +per-battery cutoff | 7.437% | 1.462% | 0.322 |
| **(c) +within-battery monotonicity [BEST]** | **7.191%** | **0.668%** | **0.377** |
| (d) +CNN1D physics (worse) | 8.243% | 1.240% | 0.259 |

---

## Key Design Decisions

**What works:**
- Per-battery cutoff voltage in physics features (not fixed 2.7V)
- Within-battery monotonicity regularisation (reduces variance 54%)
- Separate Mendeley EIS encoder (8D) vs NASA EIS encoder (3D)
- SOH in [0,1] with MAE% = MAE*100 for reporting
- True joint loss via batch interleaving (not sequential loops)

**What does not help:**
- CNN1D for 4D physics features (no natural ordering, MLP is better)
- SOH > 1.0 values (some capacity readings exceed nominal — clipped to 1.0)

**Known limitations:**
- Only 3/34 batteries have >= 50 cycles (B0005/6/7) — insufficient for all degradation stages
- Test set SOH distribution differs from train (7.3 pp mean shift)
- EIS contributes ~2.5% in v1 (NASA EIS range is small ~0.03 Ohm over lifetime)
- Mendeley LLI/LAM/CL are model-derived, not physical ground truth

---

## Regression Test

```bash
python scripts/smoke_test.py
# Expected: 17/17 PASS, EXIT=0
```

Checks: capacity from .mat field (100%), SOH in [0,1], discharge z-scored,
EIS finite, discharge shape (100,3), DataLoader collate, EISEncoder ValueError.

---

## References

- NASA PCoE Battery Dataset: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-center/publications/
- Mendeley EIS dataset: "Li-ion cells EIS dataset with fitting and degradation modes"
- Birkl et al. (2017). Degradation diagnostics for lithium ion cells. *J. Power Sources*, 341, 373-386.
- Vetter et al. (2005). Ageing mechanisms in lithium-ion batteries. *J. Power Sources*, 147, 269-281.

---

## License

MIT
