# BaFuse — Tài liệu chi tiết dự án

## 1. Mục tiêu dự án

**BaFuse** (Battery Fusion Estimation) là một hệ thống ước lượng **State of Health (SoH)** của pin lithium-ion dùng trong xe điện (EV), sử dụng phương pháp **multimodal fusion** — tức là kết hợp đồng thời nhiều nguồn tín hiệu khác nhau thay vì chỉ dùng một loại dữ liệu.

**SoH** (State of Health) là tỷ lệ dung lượng hiện tại so với dung lượng ban đầu của pin:

```
SoH (%) = (Dung lượng hiện tại / Dung lượng định mức) × 100
        = (capacity_ahr / 2.0 Ahr) × 100
```

Pin mới: SoH = 100%. Pin đến cuối vòng đời (End-of-Life, EOL): SoH ≈ 70% (mất 30% dung lượng).

**Câu hỏi đặt ra:** Nếu chỉ nhìn vào đường cong discharge (voltage giảm theo thời gian), EIS (tổng trở nội), và các đặc trưng vật lý về sự lão hóa — có thể dự đoán SoH chính xác không?

**Ba nguồn tín hiệu được fuse:**

| Nguồn | Ý nghĩa |
|---|---|
| Discharge curve | Điện áp/dòng điện/nhiệt độ theo thời gian trong chu kỳ xả |
| EIS (Electrochemical Impedance Spectroscopy) | Tổng trở điện hóa nội tại của pin |
| Physics-informed features | Các đặc trưng suy ra từ mô hình lão hóa vật lý |

---

## 2. Dataset — NASA PCoE Battery Aging Dataset

### Nguồn gốc

Dataset từ **NASA Prognostics Center of Excellence (PCoE)**, gồm **34 viên pin lithium-ion 18650** được lão hóa nhân tạo trong phòng thí nghiệm. Mỗi pin trải qua hàng trăm chu kỳ charge–discharge–impedance cho đến khi đạt EOL (SoH = 70%).

**Cấu trúc thư mục:**

```
5. BatteryDataSet/
├── 1. BatteryAgingARC-FY08Q4/     ← Campaign 1: B0005, B0006, B0007, B0018
├── 2. BatteryAgingARC_25_26_27_28_P1/
├── 3. BatteryAgingARC_25-44/       ← B0025–B0044
├── 4. BatteryAgingARC_45_46_47_48/
├── 5. BatteryAgingARC_49_50_51_52/
└── 6. BatteryAgingARC_53_54_55_56/
```

Mỗi file `.mat` là một viên pin, lưu dữ liệu dạng struct của MATLAB.

### Điều kiện thí nghiệm (Campaign 1 — tiêu biểu nhất)

- **Charge:** CC ở 1.5A đến 4.2V, sau đó CV đến khi dòng giảm xuống 20mA
- **Discharge:** CC ở 2A đến khi điện áp chạm ngưỡng cắt (2.7V với B0005)
- **Impedance:** EIS quét tần số từ 0.1 Hz đến 5 kHz
- **Dung lượng định mức:** 2 Ahr
- **EOL criterion:** Mất 30% dung lượng → capacity còn 1.4 Ahr

---

## 3. Cấu trúc dữ liệu thô trong file .mat

Mỗi file `.mat` chứa một struct nhiều cấp. Đọc bằng `scipy.io.loadmat()` ra Python:

```
B0005 (struct)
  └── cycle (array of N structs, N = số chu kỳ tổng cộng)
        ├── type: "charge" | "discharge" | "impedance"
        ├── ambient_temperature: nhiệt độ phòng (°C)
        ├── time: thời điểm bắt đầu chu kỳ
        └── data (struct, khác nhau theo type)
```

### 3.1 Chu kỳ Discharge — `data` fields

| Field | Đơn vị | Ý nghĩa |
|---|---|---|
| `Voltage_measured` | V | Điện áp đầu cực pin theo thời gian |
| `Current_measured` | A | Dòng xả (âm = xả) |
| `Temperature_measured` | °C | Nhiệt độ pin |
| `Time` | s | Vector thời gian |
| `Capacity` | Ahr | **Dung lượng tích lũy** — tăng dần từ 0 đến max trong chu kỳ |

Mỗi chu kỳ discharge có khoảng **200–400 time-steps** tùy tốc độ lấy mẫu.

### 3.2 Chu kỳ Impedance (EIS) — `data` fields

| Field | Đơn vị | Ý nghĩa |
|---|---|---|
| `Battery_impedance` | Ω (phức) | Tổng trở đo trực tiếp từ dữ liệu raw |
| `Rectified_impedance` | Ω (phức) | Tổng trở sau hiệu chỉnh và làm mượt |
| `Re` | Ω | Điện trở điện giải (electrolyte resistance) |
| `Rct` | Ω | Điện trở truyền điện tích (charge transfer resistance) |

**Quan trọng:** `Battery_impedance` là **số phức** (complex number) — phần thực là điện trở (Re), phần ảo là reactance (liên quan đến Rct).

### 3.3 Mô hình lão hóa quan sát được

Qua hàng trăm chu kỳ, có thể thấy rõ:
- **Điện áp cuối discharge giảm dần** — pin yếu hơn không giữ được điện áp cao
- **Dung lượng giảm dần** — từ 2.0 Ahr xuống 1.4 Ahr khi đến EOL
- **Tổng trở tăng dần** — pin già có điện trở nội lớn hơn, tỏa nhiệt nhiều hơn

---

## 4. Xử lý dữ liệu thô

### 4.1 Parse file .mat

Dùng `scipy.io.loadmat()` để đọc file MATLAB. Tuy nhiên cấu trúc nested struct rất phức tạp — cần truy cập đúng cách:

```python
mat_data = sio.loadmat("B0005.mat", squeeze_me=False)
battery_struct = mat_data["B0005"]
cycles = battery_struct["cycle"][0, 0]   # array of cycle structs

for cycle_idx in range(cycles.size):
    cycle = cycles.flat[cycle_idx]       # numpy.void (structured scalar)
    cycle_type = str(cycle["type"].flat[0]).strip()
    if cycle_type == "discharge":
        data = cycle["data"].flat[0]
        voltage = data["Voltage_measured"].flatten()
        capacity = data["Capacity"].flatten()
        # ...
```

**Vấn đề kỹ thuật gặp phải:** MATLAB struct khi load vào Python tồn tại dưới dạng `numpy.void` — không thể truy cập như dict thông thường, phải dùng `.flat[0]` để lấy scalar từ array kích thước `(1,1)`.

### 4.2 Tính capacity cho mỗi chu kỳ

Mỗi chu kỳ discharge được đại diện bằng **một giá trị capacity** duy nhất — đây chính là proxy của SoH:

```python
# Nếu có field Capacity (tích lũy theo thời gian):
cycle_capacity = max(|Capacity_ts|)   # giá trị max = tổng dung lượng xả được

# Nếu không có (một số battery trong dataset):
# Tích phân Coulomb: ∫|I| dt / 3600
dt = diff(time_data)
cycle_capacity = sum(|current| * dt / 3600)
```

### 4.3 Lỗi phát hiện trong quá trình parse

**Lỗi 1 — Impedance là số phức nhưng code xử lý như số thực:**

```python
# Code ban đầu — SAI
impedance_val = float(np.nanmedian(val))   # ComplexWarning!
# → val là mảng số phức, nanmedian tự cắt phần ảo trước khi tính
# → impedance_range = (-1161, 1.47e14) Ω — vô lý
```

```python
# Code sửa — ĐÚNG
mag  = np.abs(val)                     # |Z| = sqrt(Re² + Im²)
real = np.clip(val.real, 0, None)      # Re(Z) ≥ 0
imag = np.abs(val.imag)                # |Im(Z)|
# Lọc hard: [0, 1000] Ω
# Lọc soft: percentile [1%, 99%]
# Sau đó lấy median
# → impedance_range = (0.15, 648) Ω — hợp lý
```

**Lỗi 2 — Parse trùng lặp:**

B0025–B0028 xuất hiện trong cả thư mục `2.` lẫn `3.` (NASA lưu trùng). Code ban đầu parse cả 2 → 835,422 records thay vì 770,070.

```python
# Sửa: dùng set để track battery_id đã parse
seen_ids = set()
for mat_file in sorted(mat_files):
    battery_id = mat_file.stem
    if battery_id in seen_ids:
        continue   # skip duplicate
    seen_ids.add(battery_id)
```

### 4.4 Ghép cặp Discharge ↔ EIS

Mỗi sample cho model cần cả 2 loại dữ liệu. Chu kỳ trong NASA PCoE thường theo thứ tự:

```
discharge (cycle N) → charge (N+1) → impedance (N+2)
```

Nên EIS đo sau discharge khoảng 1–2 cycle. Logic ghép cặp:

```
Với mỗi discharge cycle N:
    Tìm EIS cycle trong [N, N+10]  ← max_cycle_gap = 10
    Nếu không tìm thấy → bỏ qua
```

**Lỗi ban đầu:** Code cũ dùng fallback "tìm EIS gần nhất bất kỳ" nếu không có trong window → có cặp lệch đến 39 cycle. Sửa: enforce cứng `max_cycle_gap`, không fallback.

```
Trước: cycle_gap max = 39,  mean = -0.12 (có cả gap âm)
Sau:   cycle_gap max = 10,  mean = 1.02,  98% pairs có gap = 1
```

**Lọc cycle lỗi thiết bị:**

```
Trước: capacity_range min = 0.0001 Ahr  ← lỗi thiết bị đo
Sau:   lọc bỏ capacity < 0.5 Ahr → 185 cycle bị loại trên 13 battery
       capacity_range min = 0.52 Ahr
```

### 4.5 Kết quả sau xử lý

```
Batteries:        34 (không trùng lặp)
Discharge records: 770,070
EIS records:       1,953
Pairs sau lọc:     2,537
  - capacity ≥ 0.5 Ahr
  - cycle_gap ≤ 10
  - impedance trong [0.15, 648] Ω
```

---

## 5. Feature Engineering

### 5.1 Discharge features (từ time-series)

Mỗi discharge cycle có ~200–400 time-steps. Để đưa vào LSTM, resample về **100 điểm** đồng nhất:

```python
# Resample về 100 điểm bằng nội suy tuyến tính
x_old = linspace(0, 1, N_original)
x_new = linspace(0, 1, 100)
for each feature (V, I, T):
    resampled = interp(x_new, x_old, feature_values)
```

**Lý do resample:** Sequence gốc 500 điểm → LSTM mất ~4 phút/epoch. Sau resample 100 điểm → ~45 giây/epoch (nhanh ~6 lần).

Features đưa vào LSTM: `[Voltage_measured, Current_measured, Temperature_measured]`

### 5.2 EIS features (3 features scalar)

```
impedance_ohm = median(|Z|) qua tất cả frequency points
re_ohm        = median(Re(Z)) — điện trở điện giải
rct_ohm       = median(|Im(Z)|) — proxy của Rct
```

Cả 3 được normalize bằng train-set statistics.

### 5.3 Physics-informed features (4 features)

Đây là phần có thay đổi lớn nhất trong quá trình phát triển.

**Phiên bản đầu tiên (có leakage):**

```python
cycle_age_norm  = cycle / 150
capacity_fade   = 2.0 - capacity_ahr      ← DATA LEAKAGE!
voltage_droop   = (voltage_min - 2.7) / 1.5
impedance_rise  = (impedance - mean) / std
```

**Vấn đề phát hiện:** `capacity_fade = 2.0 - capacity_ahr` và `soh_label = (capacity_ahr/2.0) × 100` đều tính từ cùng `capacity_ahr`. Tương quan tuyến tính r ≈ -1.0. Model chỉ cần học một phép nhân là ra đáp án — không học gì từ discharge hay EIS.

**Bằng chứng:** Ablation study cho thấy physics đóng góp 79.1% — bất thường vì physics chỉ là MLP nhỏ nhận 4 số.

**Phiên bản sửa (không leakage):**

```python
cycle_age_norm    = cycle / N_ref
empirical_prior   = A × (cycle / N_ref)^b   ← fit từ train population, không dùng capacity thực
voltage_droop     = (voltage_min - 2.7) / 1.5
impedance_rise    = (impedance - train_mean) / train_std
```

**Empirical aging prior** được fit bằng log-log regression trên toàn bộ tập train:

```
fade_fraction = A × (age / N_ref)^b
log(fade) = log(A) + b × log(age/N_ref)
→ fit tuyến tính trên log-log space
→ A = 0.225, b = 0.103, N_ref = 488 cycles (lấy từ tập train cụ thể)
```

Prior này là ước lượng **trung bình của toàn bộ battery trong train** — không biết capacity thực của mẫu đang predict → không leak.

### 5.4 SoH Label

```python
soh = (capacity_ahr / 2.0) × 100   # clip vào [0, 100]
```

---

## 6. Train/Val/Test Split

### Vấn đề đặc biệt với battery dataset

Không thể split ngẫu nhiên theo từng cycle vì:
- Các cycle của cùng một battery **có correlation cao** (cùng aging trajectory)
- Nếu battery B0005 xuất hiện cả trong train lẫn test → **data leakage**

**Giải pháp: Battery-level split**

```
Mỗi battery được gán vào đúng một split:
  Train: 20 batteries (60% số lượng pairs)
  Val:   7 batteries  (20%)
  Test:  7 batteries  (20%)
```

**Stratification:** Sắp xếp battery theo mean capacity → chia round-robin để mỗi split có đại diện từ battery mới (SoH cao) đến pin cũ (SoH thấp).

```
Train: 629 samples  | Val: 103 samples | Test: 223 samples
```

### Lỗi normalization (phát hiện sau)

Code ban đầu mỗi split tự normalize theo stats của chính nó:

```python
# SAI
train_dataset = BaFuseDataset(train_df, normalize=True)  # stats từ train
val_dataset   = BaFuseDataset(val_df,   normalize=True)  # stats từ VAL
test_dataset  = BaFuseDataset(test_df,  normalize=True)  # stats từ TEST
```

Nghĩa là cùng một giá trị impedance = 0.25 Ω sẽ cho ra số normalize khác nhau tùy split. Model train trên phân phối của train nhưng test nhận phân phối khác → distribution shift.

```python
# ĐÚNG
train_dataset = BaFuseDataset(train_df, normalize=True)
val_dataset   = BaFuseDataset(val_df,   external_stats=train_dataset.stats)
test_dataset  = BaFuseDataset(test_df,  external_stats=train_dataset.stats)
```

---

## 7. Kiến trúc Model — BaFuse

```
Discharge (100×3) ──→ DischargeEncoder (LSTM) ──→ latent_d (64-dim)  ─┐
EIS (3)           ──→ EISEncoder (MLP)         ──→ latent_e (64-dim)  ─┤─→ CrossAttentionFusion ──→ SoH head ──→ SoH (%)
Physics (4)       ──→ PhysicsEncoder (MLP)     ──→ latent_p (64-dim)  ─┘
```

### 7.1 DischargeEncoder — LSTM

Xử lý chuỗi thời gian (100 time-steps × 3 features):

```python
LSTM(input_size=3, hidden_size=256, num_layers=3, dropout=0.2)
→ lấy hidden state cuối cùng h[-1]
→ Linear(256 → 64) + LayerNorm + ReLU
```

**Lý do chọn LSTM:** Discharge curve có cấu trúc sequential rõ ràng — voltage giảm dần, cuối discharge voltage drop nhanh. LSTM học được pattern này tốt hơn MLP hay CNN 1D.

**Thay đổi trong quá trình:** Ban đầu `hidden_size=128, num_layers=2`. Sau khi phát hiện discharge cần gánh ~70% signal (sau khi gỡ leakage), tăng lên `hidden_size=256, num_layers=3`.

### 7.2 EISEncoder — MLP

EIS chỉ có 3 scalar features (không phải spectrum đầy đủ):

```python
Linear(3 → 128) + ReLU + Dropout(0.2)
→ Linear(128 → 64) + LayerNorm + ReLU
```

### 7.3 PhysicsEncoder — MLP

```python
Linear(4 → 256) + ReLU + Dropout(0.2)
→ Linear(256 → 128) + ReLU + Dropout(0.2)
→ Linear(128 → 64) + LayerNorm + ReLU
```

### 7.4 CrossAttentionFusion

Ba latent vector (mỗi cái 64-dim) được stack thành sequence 3 tokens:

```python
tokens = stack([latent_d, latent_e, latent_p], dim=1)   # (B, 3, 64)
attended, attn_weights = MultiheadAttention(tokens, tokens, tokens, num_heads=4)
attended = LayerNorm(attended + tokens)    # residual connection
flat = attended.reshape(B, 3×64=192)
fused = Linear(192 → 128) + ReLU          # fusion_dim = 128
```

Attention cho phép mỗi modality "hỏi" thông tin từ 2 modality còn lại. Trọng số attention có thể dùng để phân tích đóng góp.

### 7.5 SoH Prediction Head

```python
Linear(128 → 256) + ReLU + Dropout(0.2)
→ Linear(256 → 128) + ReLU + Dropout(0.2)
→ Linear(128 → 1)
```

Output: 1 số thực dự đoán SoH (%).

**Tổng số parameters: 1,495,489 (~1.5M)**

---

## 8. Loss Function

```python
Loss = MSE(pred, target)
     + λ_physics × physics_penalty    # phạt pred ngoài [0, 100]
     + λ_smooth × monotonicity_loss   # khuyến khích SoH giảm theo cycle_age
```

**Physics penalty:** `ReLU(-pred) + ReLU(pred - 100)` — đảm bảo dự đoán trong khoảng vật lý hợp lệ.

**Monotonicity loss:** Với cặp (i, j) mà `cycle_age[i] < cycle_age[j]`, phạt nếu `pred[i] < pred[j]` (pin trẻ hơn được dự đoán SoH thấp hơn pin già).

---

## 9. Training

### Cấu hình

```yaml
optimizer:    Adam
lr:           0.0005   (giảm từ 0.001 sau khi phát hiện spike)
weight_decay: 1e-5
batch_size:   32
max_epochs:   100
patience:     20       (early stopping theo val MAE)
```

### LR Schedule — Linear Warmup + Cosine Decay

**Vấn đề phát hiện:** Training log epoch 1–2 cho thấy MAE = 73% → đột ngột drop về 7% ở epoch 5–6. Nguyên nhân: LR bắt đầu cao (0.001) ngay từ epoch 1 → gradient lớn → loss spike trong khi model chưa "warm up".

```
Trước: CosineAnnealingLR từ đầu → spike mạnh epoch 1-3
Sau:   Linear warmup 10 epochs (LR: 0 → 0.0005) → Cosine decay
```

```python
def lr_lambda(epoch):
    if epoch < warmup_epochs:             # 10 epochs đầu
        return (epoch + 1) / warmup_epochs  # tăng tuyến tính
    progress = (epoch - warmup) / (total - warmup)
    return max(0.05, 0.5 * (1 + cos(π × progress)))  # cosine decay
```

### Early Stopping

Ban đầu track `val_loss` → noisy với val set nhỏ (103 samples / 4 batches). Sửa sang track `val_MAE` — ổn định hơn.

**Logging bug phát hiện:** Sau khi early stopping và load best checkpoint, log vẫn in metrics của epoch cuối (không phải epoch tốt nhất). Sửa bằng cách lưu `best_val_metrics` riêng tại thời điểm save checkpoint.

---

## 10. Đánh giá — Từ Single Split đến Cross-Validation

### Vấn đề với single split

Lần đầu đánh giá dùng 1 lần chia cố định (60/20/20). Kết quả per-battery:

```
B0053: MAE=6.95%, R²=-21    ← outlier cực đoan
```

B0053 có SoH range chỉ 4.5% (64.5–69%) và discharge xuống 1.97V (thay vì 2.5V thông thường) → operating window hoàn toàn khác → model không thể học pattern → R² rất âm. Một battery này làm méo hoàn bộ metric tổng.

### Giải pháp: 5-fold Battery-Group CV

34 battery được chia thành 5 nhóm stratified by mean capacity. Mỗi fold: train 27–28 batteries, test 6–7 batteries.

**Phân loại battery theo trajectory:**

| Nhóm | Tiêu chí | Battery | Cycles |
|---|---|---|---|
| Sufficient | ≥ 50 cycles | B0005, B0006, B0007 | 142–168 |
| Intermediate | 20–49 cycles | B0018, B0033, B0034... | 20–56 |
| Insufficient | < 20 cycles | B0025–B0032, B0038–B0044... | 1–19 |

**Kết quả 5-fold CV:**

| Fold | Test batteries chính | MAE | R² | Ghi chú |
|---|---|---|---|---|
| 1 | B0005, B0045, B0055, B0048 | 6.94% | 0.541 | Có B0005 (sufficient) |
| 2 | B0033, B0036, B0056, B0046 | 12.01% | -0.158 | Không có sufficient |
| 3 | B0018, B0034, B0049, B0053 | 7.64% | 0.291 | B0053 outlier có mặt |
| 4 | B0006, B0054, B0042, B0051 | 7.09% | 0.350 | Có B0006 (sufficient) |
| 5 | B0007, B0047, B0040, B0043 | 5.75% | 0.449 | Có B0007 (sufficient) |
| **Mean ± std** | | **7.88 ± 2.15 %** | **0.29 ± 0.24** | |

**Tách theo nhóm:**

```
Battery sufficient (≥50 cycles):     MAE = 4.04 ± 0.66 %
Battery insufficient (<20 cycles):   MAE = 10.87 ± 1.41 %
```

Fold 2 có R² âm hoàn toàn do test set không có battery nào sufficient — đây là giới hạn của dataset (chỉ 3/34 battery có đủ trajectory dài), không phải lỗi model.

---

## 11. Kết quả Modality Contribution

Sau khi gỡ data leakage:

| Modality | Trước (có leakage) | Sau khi sửa |
|---|---|---|
| Discharge | 20.9% | **70.4%** |
| Physics | **79.1%** ← giả | **27.1%** |
| EIS | 0.7% | **2.5%** |

**Về EIS ~2.5%:** Điều tra chi tiết cho thấy:
- EIS–SoH correlation = -0.37 (có tín hiệu thật)
- EIS-only sklearn MAE = 9.56%, Discharge-only = 7.21%
- Impedance thay đổi rất ít qua vòng đời (~0.03 Ω range với nhiều battery)

Kết luận: Trong NASA PCoE, discharge curve đã chứa đủ tín hiệu để dự đoán SoH. EIS có tín hiệu nhưng yếu hơn → attention mechanism tự học gán trọng số thấp cho EIS. Đây là **finding học thuật hợp lệ**, không cần "ép" EIS phải đóng góp nhiều.

---

## 12. Tổng kết các thay đổi theo thứ tự thời gian

```
[1] Implement toàn bộ skeleton → chạy được lần đầu

[2] Phát hiện Smart App Control chặn torch.dll
    → tắt SAC trong Windows Security → torch 2.14.0 chạy được

[3] Chạy lần đầu → 5 lỗi dữ liệu:
    ├── Impedance phức bị xử lý sai  → _safe_impedance_scalar()
    ├── Cycle lỗi capacity ≈ 0      → filter < 0.5 Ahr
    ├── Cycle_gap max = 39          → enforce max_cycle_gap = 10
    ├── Parse trùng battery         → seen_ids dedup
    └── Sequence 500 điểm chậm     → resample về 100 điểm (×6 tốc độ)

[4] Phát hiện data leakage: capacity_fade ↔ soh_label r ≈ -1.0
    Bằng chứng: physics=79% trong ablation
    → Xóa capacity_fade, thay bằng empirical aging prior
    → Physics contribution giảm xuống 27% (thật)

[5] Phát hiện normalization skew
    → val/test dùng external_stats từ train

[6] Điều tra EIS ~0%:
    → EIS có tín hiệu (r=-0.37) nhưng yếu hơn discharge
    → Finding học thuật: không phải lỗi

[7] Fix LR schedule: linear warmup 10 epochs + cosine decay
    → Không còn spike MAE=73% ở epoch 1-2

[8] Tăng DischargeEncoder: hidden 128→256, layers 2→3
    → Discharge cần capacity lớn hơn sau khi gỡ leakage

[9] Fix logging bug: in best-checkpoint metrics, không phải last-epoch

[10] Implement 5-fold CV với battery trajectory grouping
     → MAE = 7.88 ± 2.15 % (sufficient group: 4.04 ± 0.66 %)
```

---

## 13. Các file chính trong codebase

```
src/data/
  parse_mat.py     → Load .mat, trích xuất discharge + EIS, xử lý impedance phức
  pairing.py       → Ghép cặp discharge ↔ EIS, lọc cycle lỗi
  split.py         → Battery-level stratified split
  dataset.py       → PyTorch Dataset, resample, physics prior, normalization
src/models/
  encoders.py      → DischargeEncoder (LSTM), EISEncoder (MLP), PhysicsEncoder (MLP)
  fusion.py        → CrossAttentionFusion, WeightedFusion, ConcatFusion
  bafuse.py        → Unified model
src/
  losses.py        → SoH loss + physics bound + monotonicity
  train.py         → Training loop, warmup LR, early stopping by val MAE
  evaluate.py      → Test metrics, ablation, MC Dropout uncertainty

scripts/
  validate_pipeline.py  → Kiểm tra data quality (assertions)
  cross_validate.py     → 5-fold CV với LSTM size comparison
  investigate_eis.py    → Phân tích tín hiệu EIS
  inspect_data.py       → Debug .mat file structure

run_pipeline.py          → Entry point chính
configs/config.yaml      → Hyperparameters
```

---

## 14. Kết quả cuối cùng

| Metric | Single split | 5-fold CV (đáng tin hơn) |
|---|---|---|
| MAE | 6.97% | **7.88 ± 2.15 %** |
| RMSE | 10.82% | **10.70 ± 1.70 %** |
| R² | 0.529 | **0.29 ± 0.24** |
| MAE (sufficient batteries) | 3.45% (B0005) | **4.04 ± 0.66 %** |
