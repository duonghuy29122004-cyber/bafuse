# BaFuse v2 — Kien Truc Mo Hinh

> Phien ban: v2.1 | Cap nhat: 2026-09 | Smoke-test: 17/17 PASS

---

## 1. Tong quan kien truc

BaFuse v2 la mo hinh **multimodal multi-task** ket hop:

- **3 nguon du lieu** (discharge, EIS, physics) tu NASA PCoE --> du doan SOH
- **Degradation head** su dung EIS tu Mendeley --> uoc tinh LLI / LAM / CL

```
NASA PCoE
   |
   +------------------+------------------+
   |                  |                  |
   v                  v                  v
Discharge (B,100,3)  EIS_NASA (B,3)  Physics (B,4)
   |                  |                  |
DischargeEncoder  EISEncoder(3)    PhysicsEncoder
   |                  |                  |
latent_d (B,64)  latent_e (B,64)  latent_p (B,64)
   |                  |                  |
   +------------------+------------------+
                       |
              CrossAttentionFusion
                       |
                   fused (B,128)
                       |
             +---------+---------+
             |                   |
          SOH Head         Degradation Head
             |                   |
         soh_pred (B,1)   [LLI, LAM, CL] (B,3)
                                 ^
                                 |
                   Mendeley EIS (B,8)
                   --> mendeley_eis_encoder(8)
                   --> degradation_head_eis_only
```

**Diem khac biet so voi v1:**
- Them `DegradationHead` + `mendeley_eis_encoder` cho Mendeley path
- `BaFuseV2` ho tro `use_discharge/eis/physics` flags cho ablation (zero-masking)
- `from_bafuse_v1_checkpoint()` cho phep chuyen trong so tu v1
- SOH label: [0,1] thay vi [0,100] (fix scale bug)

---

## 2. Encoders

### 2.1 DischargeEncoder (LSTM)

```
Input:  (B, 100, 3)   [Voltage_norm, Current_norm, Temp_norm]
           |
        LSTM(input=3, hidden=256, num_layers=3, dropout=0.2, batch_first=True)
           |
        h_last = hidden state layer cuoi, time-step cuoi: (B, 256)
           |
        Linear(256->64) + LayerNorm(64) + ReLU
           |
Output: (B, 64)
```

Discharge tensor da duoc **z-score theo tung kenh** (voltage/current/temp)
truoc khi resample ve 100 diem (P2-#3 fix v2.1).

Input co the la (B, seq_len, 3) hoac (B, 3) -- encoder tu dieu chinh.

### 2.2 EISEncoder (MLP hoac CNN1D)

**NASA path** (`num_frequencies=3`, MLP):
```
Input:  (B, 3)   [|Z|_norm, Re_norm, Rct_norm]   -- ca 3 da z-score (P2-#4 fix)
           |
        Linear(3->128) + ReLU + Dropout(0.2)
           |
        Linear(128->64) + LayerNorm(64) + ReLU
           |
Output: (B, 64)
```

**Mendeley path** (`num_frequencies=8`, MLP):
```
Input:  (B, 8)   [Zmod, Zreal, Zimg, Zphz, R_e, R_ct1, R_ct2, Zw] -- da z-score
           |
        Linear(8->128) + ReLU + Dropout(0.2)
           |
        Linear(128->64) + LayerNorm(64) + ReLU
           |
Output: (B, 64)
```

`mendeley_eis_encoder` la instance rieng biet voi `eis_encoder` cua NASA path.
Hai encoder KHONG chia trong so. Stage-1 chi train `mendeley_eis_encoder`.

**P3-#8 fix (v2.1):** Bo han _mlp_fallback() lazy-init cu (layer tao trong forward()
se bi optimizer bo qua). Gio raise ValueError ro rang neu input dim sai.

**CNN1D path** (num_frequencies > 8): dung Conv1d tren spectrum day du.
Khong su dung trong pipeline nay (NASA EIS chi co 3 features, Mendeley 8).

### 2.3 PhysicsEncoder (MLP)

```
Input:  (B, 4)
   [cycle_age_norm, empirical_fade_prior, voltage_droop, impedance_rise]
           |
        Linear(4->256) + ReLU + Dropout(0.2)
           |
        Linear(256->128) + ReLU + Dropout(0.2)
           |
        Linear(128->64) + LayerNorm(64) + ReLU
           |
Output: (B, 64)
```

**4 physics features (leak-free sau v1 fixes):**

| Feature | Cong thuc | Ghi chu |
|---------|-----------|---------|
| `cycle_age_norm` | `cycle / N_ref` | N_ref = percentile 90 cua train |
| `empirical_fade_prior` | `A * (cycle/N_ref)^b` | Fit power-law tren TRAIN POPULATION, khong dung capacity cua mau cu the |
| `voltage_droop` | `(Vmin - Vcutoff) / (4.2 - Vcutoff)` | Vcutoff theo tung battery (P1-#2) |
| `impedance_rise` | `(Z - Z_mean) / Z_std` | Z_mean/std tu train set |

**voltage_droop dung cutoff dung cua tung battery** (BATTERY_CUTOFF_VOLTAGE dict).
Vi du B0053 co cutoff=2.0V, truoc day dung 2.7V nhan tao gay "droop" gia.

### 2.4 PhysicsEncoderCNN (Experimental)

Thu nghiem CNN1D thay MLP. Ket qua: MAE kem hon 1.05% so voi MLP.
Physics features khong co thu tu tu nhien -> CNN1D khong co loi the.
**Khong dung trong pipeline chinh.**

---

## 3. CrossAttentionFusion

Ba latent vector (moi cai 64-dim) duoc stack thanh 3 "tokens", roi dung
Multi-Head Self-Attention (4 heads) de moi modalite "hoi" thong tin tu 2 modalite con lai.

```
discharge_latent (B, 64) --+
eis_latent       (B, 64) --+--> stack --> (B, 3, 64)
physics_latent   (B, 64) --+
                                   |
                    MultiheadAttention(embed=64, num_heads=4, dropout=0.1)
                                   |
                    LayerNorm(attended + residual)
                                   |
                    reshape --> (B, 192)
                                   |
                    Linear(192->128) + ReLU + Dropout(0.1)
                                   |
Output: fused (B, 128)  +  attention_weights (B, 4, 3, 3)
```

Attention weights co the dung de phan tich contribution cua tung modalite.

**Cac fusion khac (khong dung trong pipeline chinh):**
- `WeightedFusion`: 3 learnable scalar weights (softmax)
- `ConcatFusion`: cat(latents) -> Linear, baseline don gian nhat

**Ablation support:** Khi `use_discharge=False`, `latent_d` bi thay bang zeros
truoc khi vao fusion. Cho phep eval-time ablation khong can retrain.

---

## 4. SOH Prediction Head

```
fused (B, 128)
    |
Linear(128->256) + ReLU + Dropout(0.2)
    |
Linear(256->128) + ReLU + Dropout(0.2)
    |
Linear(128->1)
    |
Output: soh_pred (B, 1)   -- range xap xi [0, 1]
```

**SOH scale (v2.1 fix):**
- Internal representation: SOH = capacity_ahr / nominal_capacity_Ah ∈ [0, 1]
- Reporting: MAE% = MAE * 100, RMSE% = RMSE * 100
- Nominal capacity lay tu `BATTERY_NOMINAL_CAPACITY` dict (P1-#2 fix)
- Truoc v2.1: dung SOH ∈ [0, 100] -> MSE loss ~ 5923, model khong hoi tu

---

## 5. DegradationHead (Mendeley path)

```
Input:  (B, input_dim)   -- fused (B,128) hoac eis_latent (B,64)
           |
        Linear(input_dim->hidden_dim) + LayerNorm + ReLU + Dropout
           |
        Linear(hidden_dim->hidden_dim//2) + ReLU + Dropout
           |
   +-------+-------+
   |       |       |
 head_LLI head_LAM head_CL
   |       |       |
(B,1)   (B,1)   (B,1)
   |       |       |
   +-------+-------+
           |
Output: deg_pred (B, 3) = [LLI, LAM, CL]
```

**Hai degradation head:**
- `degradation_head`: input_dim=128 (fused), dung khi co ca 3 modalite
- `degradation_head_eis_only`: input_dim=64 (eis latent), dung cho Mendeley-only path

**QUAN TRONG:** LLI/LAM/CL la "model-derived degradation-mode estimates" tu
ECM fitting (khong phai ground truth vat ly). Khong duoc goi la "ground truth".

---

## 6. Loss Functions

### 6.1 SoHPredictionLoss (NASA batches)

```
L_soh = MSE(pred, target)                    -- tren SOH [0,1]
      + 0.1 * relu(-pred) + relu(pred - 1.0) -- physics penalty [0,1] (v2.1 fix)
      + 0.05 * monotonicity_loss             -- within-battery only (bug fix v1)
```

**Monotonicity fix (v1 bug):** Chi so sanh (i,j) trong cung battery_id.
Cross-battery comparison la sai logic vi aging rate khac nhau.

### 6.2 DegradationLoss (Mendeley batches)

```
L_deg = w_LLI * MSE(LLI_pred, LLI_target)
      + w_LAM * MSE(LAM_pred, LAM_target)
      + w_CL  * MSE(CL_pred,  CL_target)
```

Default: w_LLI = w_LAM = w_CL = 1.0 (configurable trong config.yaml).

### 6.3 Combined loss (joint training)

```
L_total = lambda_soh * L_soh + lambda_deg * L_deg
```

Default: lambda_soh=1.0, lambda_deg=0.5.

**P3-#6 fix (v2.1):** Truoc day dung 2 loop tach biet (NASA xong roi Mendeley).
Gio dung `itertools.cycle` interleave batch theo tung step, mot `backward()` + `optimizer.step()`
cho ca hai loss. Tranh gradient cac nhau giua 2 task.

---

## 7. Training Strategy (3 stages)

### Stage 1 — Degradation pre-training (Mendeley)

```
Freeze: tat ca tru mendeley_eis_encoder + degradation_head_eis_only
Train:  Mendeley EIS (8D) --> mendeley_eis_encoder --> degradation_head_eis_only
Loss:   L_deg (LLI/LAM/CL)
```

**P3-#7 fix (v2.1):** Freeze dung `startswith("mendeley_eis_encoder.")` thay vi
substring `"eis_encoder" in name` (da vo tinh match ca NASA eis_encoder).

### Stage 2 — Weight transfer

```
Stage-1 checkpoint: eis_encoder.* --> mendeley_eis_encoder.*
                    (74/74 params compatible, 0 skipped)
NASA eis_encoder (3D input) giu trong so random (khong tuong thich 8D)
```

### Stage 3 — Full BaFuseV2 training (NASA + optional Mendeley)

```
Training mode "staged":
  Interleave NASA batches (SOH loss) + Mendeley batches (deg loss)
  Single backward + optimizer.step() per step
  L = 1.0 * L_soh + 0.5 * L_deg
```

---

## 8. Data Pipeline

### 8.1 NASA PCoE preprocessing

```
parse_all_mat_files()
    |  P1-#1 fix: capacity tu .mat field (len>0), ko dung coulomb-counting
    |  100% cycles dung .mat Capacity field (0 fallback)
    v
pair_discharge_eis()
    |  P1-#2 fix: filter threshold 25% nominal_capacity(battery_id) (khong co dinh 0.5Ah)
    |  Vectorised MultiIndex.isin() filter (nhanh ~5-20x)
    v
split_by_battery()
    |  P3-#9 fix: stratify ca split thu 2 (val/test), graceful fallback
    v
BaFuseDataset.__init__()
    |  P2-#5 fix: pre-filter data_df neu discharge_data_df cung cap (tranh shape mixing)
    v
BaFuseDataset.__getitem__()
    |  P2-#3 fix: z-score ts_raw V/I/T truoc resample
    |  P2-#4 fix: z-score ca re_ohm, rct_ohm (them stats 're','rct')
    |  P1-#2 fix: SOH = capacity / get_nominal_capacity(battery_id)
    v
Output shapes:
    discharge: (B, 100, 3)   -- z-scored
    eis:       (B, 3)         -- z-scored
    physics:   (B, 4)         -- normalized
    soh_label: (B,)            -- [0, 1]
```

### 8.2 Mendeley preprocessing

```
build_mendeley_dataframe()
    |  P4-#11: log match rate per cell, warning <90%
    v
_aggregate_eis_per_cycle()   -- median per aging_cycle, SOC=100% only
    v
inner_join(eis_agg, circ_raw) on (cell_id, aging_cycle)
    v
MendeleyDataset.__getitem__()
Output shapes:
    eis:      (B, 8)  -- [Zmod, Zreal, Zimg, Zphz, Re, Rct1, Rct2, Zw], z-scored
    soh:      (B,)    -- [0,1] (soh_pct/100)
    lli/lam/cl: (B,)  -- model-derived, z-scored voi train stats
```

---

## 9. Ablation Framework (A1-A7)

7 cau hinh ablation, retrain rieng biet, cung split/seed:

| ID | Modalities | Ghi chu |
|----|-----------|---------|
| A1_full | D + E + P | Baseline day du |
| A2_no_discharge | E + P | Bo discharge |
| A3_no_eis | D + P | Bo EIS |
| A4_no_physics | D + E | Bo physics |
| A5_discharge_only | D | Chi discharge |
| A6_eis_only | E | Chi EIS |
| A7_physics_only | P | Chi physics |

**Contribution score (DELTA_RMSE):**
```
delta_RMSE(m) = RMSE_without_m - RMSE_full
contribution_pct(m) = 100 * delta_RMSE(m) / sum(delta_RMSE(all))
```

Nay la "ablation-based contribution", KHONG phai causal importance.

---

## 10. RAG Explanation Layer (downstream, khong train)

```
{LLI: x, LAM: y, CL: z}
        |
DegradationExplainer.explain()
        |
Knowledge base (9 entries: 3 LLI, 3 LAM, 3 CL)
Moi entry: mechanism, indicators, conditions, reference
        |
Output:
  - estimated_modes: {LLI, LAM, CL}
  - severity: negligible/low/moderate/high
  - explanations: top-k mechanisms per mode
  - disclaimer: "model-derived estimates, not ground truth"
```

RAG KHONG tham gia training, KHONG anh huong prediction.
Chi la tang giai thich downstream.

---

## 11. Tong so tham so (BaFuseV2)

| Component | Params |
|-----------|--------|
| DischargeEncoder (LSTM h=256, 3 layers) | ~1,115,648 |
| EISEncoder NASA (3->128->64) | ~8,384 |
| EISEncoder Mendeley (8->128->64) | ~8,960 |
| PhysicsEncoder MLP (4->256->128->64) | ~50,560 |
| CrossAttentionFusion | ~74,240 |
| SOH Head (128->256->128->1) | ~66,177 |
| DegradationHead (fused, 128->128->64->3) | ~16,707 |
| DegradationHead EIS-only (64->64->32->3) | ~4,099 |
| **Tong BaFuseV2** | **~1,536,711** |

---

## 12. Ket qua training (v2.1, sau 14-bug fix)

### 12.1 Data split (34 batteries, 935 pairs)

```
Train: 20 batteries (616 samples)  mean SOH = 73.8%
Val:    7 batteries (222 samples)  mean SOH = 75.7%  KS p=0.0002 vs train
Test:   7 batteries  (97 samples)  mean SOH = 66.5%  KS p=0.0000 vs train
```

Battery overlap: NONE (tat ca 3 split hoan toan tach biet).

Mean shift train->test: **7.3 pp** (test battery degraded nhieu hon).
Day la nguyen nhan chinh khien test MAE > val MAE.

### 12.2 Baseline (mean predictor)

| | Val | Test |
|---|---|---|
| MAE | 9.22% | 13.86% |
| RMSE | 11.63% | 17.92% |
| R² | 0.000 | 0.000 |

### 12.3 BaFuseV2 (staged training, epoch best)

| | Val (best epoch ~9) |
|---|---|
| MAE | ~4.9% |
| RMSE | ~7.2% |
| R² | ~0.59 |
| vs baseline | MAE giam ~47%, R² tu 0 -> 0.59 |

*Test set metrics se cap nhat sau khi training hoan thanh.*

### 12.4 Lich su 5-fold CV (v1, truoc v2 changes)

| Config | MAE (all) | Std | R² |
|--------|-----------|-----|-----|
| (a) baseline (co bug) | 7.947% | 2.226% | 0.174 |
| (b) +cutoff_fix | 7.437% | 1.462% | 0.322 |
| (c) +mono_fix [BEST v1] | 7.191% | **0.668%** | 0.377 |
| (d) +cnn1d_physics | 8.243% | 1.240% | 0.259 |

---

## 13. Cau truc thu muc (v2)

```
bafuse/
├── src/
│   ├── data/
│   │   ├── parse_mat.py          NASA .mat parser (P1-#1 fix)
│   │   ├── pairing.py            discharge-EIS pairing (P1-#2, P4-#14)
│   │   ├── split.py              battery-level split (P3-#9)
│   │   ├── dataset.py            BaFuseDataset (P2-#3,4,5; P1-#2)
│   │   └── mendeley_dataset.py   Mendeley parser + LOOCV (P4-#11,12,13)
│   ├── models/
│   │   ├── encoders.py           DischargeEncoder, EISEncoder (P3-#8), PhysicsEncoder
│   │   ├── fusion.py             CrossAttentionFusion, WeightedFusion, ConcatFusion
│   │   ├── bafuse.py             BaFuse v1 (backward compat)
│   │   ├── bafuse_v2.py          BaFuseV2 (multi-task, ablation flags)
│   │   └── degradation_head.py   DegradationHead (LLI/LAM/CL)
│   ├── training/
│   │   ├── degradation_trainer.py  Stage-1 Mendeley pre-training (P3-#7)
│   │   └── multitask_trainer.py    Stage-2/3 interleaved joint training (P3-#6)
│   ├── losses.py                 SoHPredictionLoss + DegradationLoss + MultiTaskLoss
│   ├── train.py                  Original NASA training loop (backward compat)
│   ├── evaluate.py               Test set evaluation
│   ├── ablation.py               Ablation study
│   └── visualization.py          7 plot functions
├── experiments/
│   ├── run_ablation.py           A1-A7 ablation runner
│   └── evaluate_ablation.py      CSV/JSON/plot summary
├── evaluation/
│   ├── metrics.py                compute_metrics (MAE/RMSE/R²)
│   └── ablation_metrics.py       AblationAnalyzer, DELTA_RMSE contribution
├── rag/
│   ├── knowledge_base.py         9 degradation mechanism entries
│   └── retriever.py              DegradationExplainer
├── configs/
│   ├── config.yaml               v1 config (backward compat)
│   └── config_bafuse_v2.yaml     v2 config (staged training, degradation)
├── scripts/
│   ├── preprocess_mendeley.py    Mendeley preprocessing CLI
│   ├── train_degradation.py      Stage-1 CLI
│   ├── train_bafuse_v2.py        Stage-2/3 CLI
│   ├── smoke_test.py             17-check regression test
│   └── audit_splits.py           Split overlap + baseline audit
└── app/
    └── dashboard.py              Streamlit visualization
```

---

## 14. Lenh chay (v2)

```bash
# 1. Preprocess Mendeley
python scripts/preprocess_mendeley.py

# 2. Stage-1: Degradation pre-training
python scripts/train_degradation.py --epochs 50

# 3. Stage-2: BaFuseV2 staged training
python scripts/train_bafuse_v2.py --training_mode staged --epochs 100

# 4. Original NASA-only pipeline (backward compat)
python run_pipeline.py --data_dir "5. BatteryDataSet" --epochs 50

# 5. Ablation (A1-A7)
python experiments/run_ablation.py --config configs/config_bafuse_v2.yaml

# 6. Ablation results + plots
python experiments/evaluate_ablation.py

# 7. Regression smoke-test
python scripts/smoke_test.py
```

---

## 15. Cac bug da sua (v2.1)

| # | File | Noi dung |
|---|------|----------|
| P1-#1 | parse_mat.py | Fix capacity extraction: len>0 thay vi >=min_len |
| P1-#2 | dataset.py, pairing.py | BATTERY_NOMINAL_CAPACITY dict, /2.0 -> per-cell, relative threshold |
| P2-#3 | dataset.py | Z-score ts_raw V/I/T truoc resample |
| P2-#4 | dataset.py | Z-score re_ohm + rct_ohm (them stats) |
| P2-#5 | dataset.py | Pre-filter data_df trong __init__ tranh shape mixing |
| P3-#6 | multitask_trainer.py | Interleave joint loss voi itertools.cycle |
| P3-#7 | degradation_trainer.py | startswith() thay vi substring match |
| P3-#8 | encoders.py | Bo lazy-init _mlp_fallback, raise ValueError |
| P3-#9 | split.py | Stratify ca split thu 2, graceful fallback |
| P4-#10 | parse_mat.py | Proxy-feature disclaimer trong docstring |
| P4-#11 | mendeley_dataset.py | Log match rate per cell |
| P4-#12 | mendeley_dataset.py | leave_one_cell_out_splits() 8-fold LOOCV |
| P4-#13 | mendeley_dataset.py | Dedup COLUMN_MAP |
| P4-#14 | pairing.py | Vectorised MultiIndex.isin() filter |
