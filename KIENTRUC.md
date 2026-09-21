# BaFuse - Kien Truc Mo Hinh va Ket Qua

---

## 1. Tong quan kien truc

BaFuse la mo hinh **multimodal fusion** ket hop 3 nguon du lieu doc lap de du doan
State of Health (SoH) cua pin lithium-ion:

```
Discharge curve   (100 x 3)  -->  DischargeEncoder  -->  latent_d (64-dim)  --+
EIS features      (3,)        -->  EISEncoder        -->  latent_e (64-dim)  --+--> CrossAttentionFusion --> SoH Head --> SoH (%)
Physics features  (4,)        -->  PhysicsEncoder    -->  latent_p (64-dim)  --+
```

Mo hinh co **3 tang chinh**: Encoders (ma hoa tung modalite doc lap) →
Fusion (ket hop thong tin) → Prediction Head (du doan scalar).

---

## 2. Encoders - Phien ban hien tai

### 2.1 DischargeEncoder (LSTM)

Xu ly chuoi thoi gian discharge curve.

```
Input:  (B, 100, 3)   -- 100 time-steps x [Voltage, Current, Temperature]
           |
        LSTM(input=3, hidden=256, num_layers=3, dropout=0.2)
           |
        h_last = hidden state cua layer cuoi, epoch cuoi: (B, 256)
           |
        Linear(256 -> 64) + LayerNorm(64) + ReLU
           |
Output: (B, 64)   -- latent vector
```

**Ly do chon LSTM:** Discharge curve co cau truc sequential ro rang --
voltage giam dan, drop nhanh cuoi chu ky. LSTM hoc duoc pattern nay
tot hon MLP (khong co memory) hay Transformer (overfit khi data nho).

**Lich su thay doi:**

| Phien ban | hidden_size | num_layers | Ly do thay doi |
|-----------|-------------|------------|----------------|
| v1 (dau)  | 128         | 2          | Default skeleton |
| v2 (hien tai) | 256    | 3          | Discharge gach 70% signal sau khi go leakage, can capacity lon hon |

### 2.2 EISEncoder (MLP adaptive)

Xu ly 3 scalar features tu do EIS: [|Z| (Ohm), Re(Z) (Ohm), |Im(Z)| (Ohm)].

```
Input:  (B, 3)
           |
        Linear(3 -> 128) + ReLU + Dropout(0.2)
           |
        Linear(128 -> 64) + LayerNorm(64) + ReLU
           |
Output: (B, 64)
```

**Luu y:** EISEncoder co kha nang xu ly ca spectrum day du (CNN1D qua Conv1d)
neu input co nhieu frequency points (num_freq > 8). Trong NASA PCoE,
EIS duoc rut gon thanh 3 scalar --> dung MLP path.

### 2.3 PhysicsEncoder (MLP) -- Phien ban chinh

Xu ly 4 physics-informed features:

```
[cycle_age_norm, empirical_fade_prior, voltage_droop, impedance_rise]

Input:  (B, 4)
           |
        Linear(4 -> 256) + ReLU + Dropout(0.2)
           |
        Linear(256 -> 128) + ReLU + Dropout(0.2)
           |
        Linear(128 -> 64) + LayerNorm(64) + ReLU
           |
Output: (B, 64)
```

**Lich su thay doi cua physics features (quan trong):**

| Phien ban | Features | Van de |
|-----------|----------|--------|
| v1 | cycle_age, **capacity_fade**, voltage_droop, impedance_rise | **DATA LEAKAGE**: capacity_fade = 2 - capacity_ahr, ma soh_label = capacity_ahr/2 * 100 --> tuong quan r = -1.0 |
| v2 (hien tai) | cycle_age_norm, **empirical_fade_prior**, voltage_droop, impedance_rise | Prior duoc fit tu train population (power-law), khong dung capacity thuc cua tung mau |

**voltage_droop -- Lich su thay doi:**

| Phien ban | Cong thuc | Van de |
|-----------|-----------|--------|
| v1 | (voltage_min - 2.7) / 1.5 | Dung cutoff co dinh 2.7V cho tat ca battery -- sai voi B0053 (cutoff = 2.0V), B0041 (2.0V)... |
| v2 (hien tai) | (voltage_min - cutoff_battery) / (4.2 - cutoff_battery) | Dung cutoff dung theo README cua tung campaign |

**Per-battery cutoff voltages (tu NASA PCoE README):**

| Battery | Cutoff (V) | Campaign |
|---------|-----------|----------|
| B0005 | 2.7 | 1 |
| B0006 | 2.5 | 1 |
| B0007 | 2.2 | 1 |
| B0018 | 2.5 | 1 |
| B0025 | 2.0 | 2/3 |
| B0026 | 2.2 | 2/3 |
| B0027 | 2.5 | 2/3 |
| B0028 | 2.7 | 2/3 |
| B0029 | 2.0 | 3 |
| B0033 | 2.0 | 3 |
| B0034 | 2.2 | 3 |
| B0036 | 2.7 | 3 |
| B0038 | 2.2 | 3 |
| B0039 | 2.5 | 3 |
| B0040 | 2.7 | 3 |
| B0041-B0052 | 2.0/2.2/2.5/2.7 (theo so) | 3/4/5 |
| B0053 | **2.0** | 6 -- day la ly do voltage_min ~1.97V, khong phai suy giam |
| B0054 | 2.2 | 6 |
| B0055 | 2.5 | 6 |
| B0056 | 2.7 | 6 |

### 2.4 PhysicsEncoderCNN (Experimental -- Task 3)

Thu nghiem dung CNN1D thay the cho MLP:

```
Input:  (B, 4)
           |
        unsqueeze(1) --> (B, 1, 4)
           |
        Conv1d(in=1,  out=16, k=2, pad=1) + ReLU  --> (B, 16, 5)
           |
        Conv1d(in=16, out=32, k=2, pad=0) + ReLU  --> (B, 32, 4)
           |
        AdaptiveAvgPool1d(2) --> (B, 32, 2)
           |
        Flatten --> (B, 64)
           |
        Linear(64 -> 64) + LayerNorm + ReLU
           |
Output: (B, 64)   -- shape giong het PhysicsEncoder
```

Params: 1,458,321 (it hon MLP 37,168 params do bo phan physics encoder nho hon).

---

## 3. Fusion -- CrossAttentionFusion (phien ban chinh)

Ba latent vector duoc stack thanh sequence 3 "tokens", sau do dung
Multi-Head Self-Attention de moi modalite co the "hoi" thong tin tu 2 modalite con lai.

```
discharge_latent (B, 64) --+
eis_latent       (B, 64) --+--> stack --> (B, 3, 64)
physics_latent   (B, 64) --+
                                   |
                    MultiheadAttention(embed=64, heads=4, dropout=0.1)
                                   |
                           (B, 3, 64) attended
                                   |
                    LayerNorm(attended + tokens)  -- residual connection
                                   |
                    reshape --> (B, 192)
                                   |
                    Linear(192 -> 128) + ReLU + Dropout(0.1)
                                   |
Output: fused (B, 128)  +  attention_weights (B, 4, 3, 3)
```

**Attention weights** co the dung de phan tich dong gop cua tung modalite --
moi head hoc mot aspect khac nhau ve moi quan he giua 3 nguon tin hieu.

**Cac fusion khac co san (khong dung trong pipeline chinh):**

- **WeightedFusion**: 3 learnable scalar weights (softmax normalized),
  combined = w_d * latent_d + w_e * latent_e + w_p * latent_p
  --> giai thich duoc nhung kem flexible.

- **ConcatFusion**: cat(latent_d, latent_e, latent_p) --> Linear(192->256->128)
  --> baseline don gian nhat, khong co cross-modality interaction.

---

## 4. SoH Prediction Head

```
fused (B, 128)
    |
Linear(128 -> 256) + ReLU + Dropout(0.2)
    |
Linear(256 -> 128) + ReLU + Dropout(0.2)
    |
Linear(128 -> 1)
    |
Output: soh_pred (B, 1)   -- range [0, 100]
```

---

## 5. Loss Function

```
Loss = MSE(pred, target)
     + 0.1 * physics_penalty
     + 0.05 * monotonicity_loss
```

**Physics penalty:** `relu(-pred) + relu(pred - 100)` -- phat du doan ngoai [0, 100].

**Monotonicity loss -- Lich su thay doi:**

| Phien ban | Logic | Van de |
|-----------|-------|--------|
| v1 | Pairwise (i,j) trong toan bo batch -- cross-battery | **Bug logic**: battery A cycle 50 co the co SoH cao hon battery B cycle 10 (aging rate khac nhau). Penalty sai |
| v2 (hien tai) | Chi so sanh cap (i,j) trong cung battery_id | Dung ve mat vat ly: monotonicity chi co nghia trong trajectory cua 1 battery |

**Ket qua sau khi sua bug monotonicity:**
- MAE from 7.44% (config b) --> **7.19%** (config c)
- std from 1.46% --> **0.67%** (on dinh hon rat nhieu)

---

## 6. Training Configuration

```yaml
optimizer:        Adam
learning_rate:    0.0005     # Giam tu 0.001 (tranh spike epoch 1-2)
weight_decay:     1e-5
batch_size:       32
max_epochs:       100
patience:         20         # early stopping theo val MAE (khong dung val loss)
```

**LR Schedule -- Lich su thay doi:**

| Phien ban | Schedule | Van de |
|-----------|----------|--------|
| v1 | CosineAnnealingLR bat dau ngay | Loss spike manh epoch 1-3 (MAE 73% -> dot ngot 7%) |
| v2 (hien tai) | Linear warmup 10 epochs + Cosine decay | On dinh, khong spike |

```
Epoch 1-10: LR tang dan tu 0 --> 0.0005  (linear)
Epoch 10+:  LR giam dang cosine xuong 5% LR ban dau
```

**Early stopping:**
- v1: theo `val_loss` --> noisy khi val set nho (103 samples / 4 batches)
- v2: theo `val_MAE` --> on dinh hon

**Logging bug da sua:** Truoc day sau khi load best checkpoint, log van in metrics
cua epoch cuoi (khong phai epoch tot nhat). Da sua bang cach luu `best_val_metrics`
tai thoi diem save checkpoint.

---

## 7. Data Pipeline (anh huong kien truc)

### 7.1 Sequence resampling

Discharge curve goc co 200-500 time-steps. Resampled ve 100 diem deu bang `np.interp`:

```
Input:  (N, 3) -- N co the la 200-500
           |
        np.interp (linear interpolation)
           |
Output: (100, 3) -- dong nhat, nhanh ~6x
```

Dieu nay quyet dinh `seq_len = 100` trong DischargeEncoder input.

### 7.2 Feature dimensions

```
discharge:  (B, 100, 3)   seq_len=100, features=[V, I, T]
eis:        (B, 3)         [|Z|_norm, Re_Ohm, |Im|_Ohm]
physics:    (B, 4)         [cycle_age_norm, empirical_prior, voltage_droop, impedance_rise]
```

### 7.3 Normalization

- v1: Moi split (train/val/test) tu tinh mean/std cua chinh no --> distribution shift
- v2: Val va test dung `external_stats = train_dataset.stats` --> chuan hoa dong nhat

---

## 8. Tong so tham so

| Component | Params |
|-----------|--------|
| DischargeEncoder (LSTM 3-layer, h=256) | ~1,115,648 |
| EISEncoder (MLP 3->128->64) | ~8,384 |
| PhysicsEncoder MLP (4->256->128->64) | ~50,560 |
| CrossAttentionFusion (64-dim, 4 heads) | ~74,240 |
| SoH Head (128->256->128->1) | ~66,177 |
| **Tong (MLP physics)** | **~1,495,489** |
| PhysicsEncoderCNN (thay the) | ~2,192 (nho hon 48x) |
| **Tong (CNN1D physics)** | **~1,458,321** |

---

## 9. Ket qua theo tung giai doan

### 9.1 Truoc khi sua bug (co data leakage)

Khi con `capacity_fade` trong physics features:

```
Test MAE:        ~9.0%     (gia -- phan lon la leakage)
Test R^2:        ~0.87     (gia -- phan lon la leakage)
Physics contrib: 79.1%     (bat thuong -- MLP nho ma anh huong lon)
Discharge:       20.9%
EIS:              0.7%
```

### 9.2 Sau khi sua leakage (single split 60/20/20)

```
Test MAE:        6.97%
Test RMSE:       10.82%
Test R^2:        0.529
Test MAPE:       13.68%

Per-battery:
  B0005: MAE=3.45%  R^2=0.811   -- pin du data (142 cycles)
  B0048: MAE=5.44%  R^2=0.268
  B0053: MAE=6.95%  R^2=-21.1   -- outlier: SoH range 4.5%, cutoff 2.0V
  B0026: MAE=11.95% R^2=-2.96   -- chi 9 cycles
  B0039: MAE=16.28% R^2=0.016
  B0045: MAE=19.71% R^2=-10.1
  B0051: MAE=25.22% R^2=-0.34   -- chi 6 cycles

Modality contributions:
  Discharge: 70.4%
  Physics:   27.1%
  EIS:        2.5%
```

### 9.3 5-fold Cross-validation -- Baseline (config a)

```
(a) baseline  -- fixed 2.7V cutoff, cross-battery monotonicity, MLP physics

Fold 1: MAE=5.71%  R^2=0.641
Fold 2: MAE=9.40%  R^2=0.113
Fold 3: MAE=11.56% R^2=-0.699
Fold 4: MAE=7.07%  R^2=0.382
Fold 5: MAE=5.99%  R^2=0.434

ALL:  MAE=7.947 +/- 2.226 %   RMSE=11.142 +/- 2.208 %   R^2=0.174 +/- 0.468
SUF:  MAE=3.775 +/- 0.802 %   (B0005, B0006, B0007 -- battery du data)
```

### 9.4 5-fold CV -- Sau khi sua voltage_droop (config b)

```
(b) +cutoff_fix  -- dung cutoff dung theo tung battery

Fold 1: MAE=6.24%  R^2=0.586
Fold 2: MAE=8.01%  R^2=0.363
Fold 3: MAE=9.92%  R^2=-0.197
Fold 4: MAE=7.22%  R^2=0.391
Fold 5: MAE=5.79%  R^2=0.467

ALL:  MAE=7.437 +/- 1.462 %   RMSE=10.360 +/- 1.264 %   R^2=0.322 +/- 0.271
SUF:  MAE=3.961 +/- 0.441 %

Cai thien so voi baseline: MAE -0.51%, std -0.76%, R^2 +0.148
```

### 9.5 5-fold CV -- Sau khi sua monotonicity loss (config c)  **[BEST]**

```
(c) +mono_fix  -- within-battery monotonicity only

Fold 1: MAE=6.88%  R^2=0.579
Fold 2: MAE=8.50%  R^2=0.316
Fold 3: MAE=6.91%  R^2=0.227
Fold 4: MAE=7.02%  R^2=0.370
Fold 5: MAE=6.64%  R^2=0.393

ALL:  MAE=7.191 +/- 0.668 %   RMSE=10.112 +/- 0.518 %   R^2=0.377 +/- 0.116
SUF:  MAE=4.559 +/- 1.080 %

Cai thien so voi (b): MAE -0.246%, std -0.794% (giam ~54%), R^2 +0.055
--> Tac dong chinh cua monotonicity fix la GIAM VARIANCE (model on dinh hon)
```

### 9.6 5-fold CV -- CNN1D physics encoder (config d)

```
(d) +cnn1d_physics  -- thay MLP bang CNN1D cho physics encoder

Fold 1: MAE=7.31%  R^2=0.555
Fold 2: MAE=10.01% R^2=0.191
Fold 3: MAE=9.47%  R^2=-0.120
Fold 4: MAE=7.06%  R^2=0.397
Fold 5: MAE=7.36%  R^2=0.272

ALL:  MAE=8.243 +/- 1.240 %   RMSE=10.893 +/- 0.799 %   R^2=0.259 +/- 0.226
SUF:  MAE=5.284 +/- 1.093 %

So voi (c): MAE +1.052% (kem hon) --> CNN1D KHONG GIUP CAI THIEN
```

---

## 10. Bang tong hop ket qua

```
==========================================================================
Config                     MAE (all)         MAE (>=50 cy)     R^2 (all)
--------------------------------------------------------------------------
(a) baseline               7.947 +/- 2.226%  3.775 +/- 0.802%  0.174
(b) +cutoff_fix            7.437 +/- 1.462%  3.961 +/- 0.441%  0.322
(c) +mono_fix [BEST]       7.191 +/- 0.668%  4.559 +/- 1.080%  0.377
(d) +cnn1d_physics         8.243 +/- 1.240%  5.284 +/- 1.093%  0.259
==========================================================================
```

---

## 11. Ket luan kien truc

### Dieu gi giup ich

1. **Per-battery cutoff voltage** (config b vs a): Giam MAE 0.51%, quan trong cho
   battery co cutoff thap (B0053 = 2.0V) vi voltage_droop truoc do bi sai lech
   co dinh.

2. **Within-battery monotonicity** (config c vs b): Khong cai thien MAE nhieu
   (chi -0.25%) nhung **giam std tu 1.46% xuong 0.67% (giam 54%)**. Day la
   tac dong chinh: model on dinh hon qua cac fold, it phu thuoc may man
   vao battery nao roi vao test set.

3. **DischargeEncoder lon hon** (256 hidden, 3 layers): Can thiet sau khi go
   leakage vi discharge phai ganh ~70% signal.

4. **Linear warmup LR**: Tranh spike MAE=73% epoch 1-2.

### Dieu KHONG giup ich

- **CNN1D physics encoder**: MAE kem hon MLP 1.05%. Ly do: 4 physics features
  khong co thu tu tu nhien (khong phai sequence), CNN1D khong co loi the gi
  so voi MLP khi xu ly vector nho nhu vay. MLP van la lua chon tot hon.

### Han che cua dataset

- Chi 3/34 battery co du >= 50 cycles (B0005/6/7) --> MAE tren nhom nay chi
  4.04% +/- 0.66% trong khi nhom <20 cycles la 10.87%.
- Fold 3 R^2 am o nhieu config vi test set khong chua battery nao "sufficient".
- EIS chi dong gop ~2.5% vi NASA PCoE co impedance range nho (~0.03 Ohm
  qua vong doi), discharge curve da chim nhieu thong tin hon.

---

## 12. Cau truc thu muc model

```
src/models/
├── encoders.py     DischargeEncoder, EISEncoder, PhysicsEncoder, PhysicsEncoderCNN
├── fusion.py       CrossAttentionFusion, WeightedFusion, ConcatFusion
└── bafuse.py       BaFuse (unified model, chon encoder/fusion qua flag)
```

**Flags quan trong:**

```python
BaFuse(
    physics_encoder_type = "mlp"             # hoac "cnn1d"
    fusion_method        = "cross_attention"  # hoac "weighted", "concat"
)
```
