# BaFuse: Multimodal Fusion for EV Battery State-of-Health Estimation

---

## 1. Mục tiêu đề tài

### 1.1 Vấn đề cần giải quyết

Pin lithium-ion trong xe điện (EV) bị lão hóa dần theo thời gian: dung lượng giảm,
điện trở nội tăng, hiệu suất giảm. Nếu không giám sát đúng, pin có thể bị hư hỏng
đột ngột, gây mất an toàn hoặc làm giảm khoảng chạy của xe.

**State of Health (SoH)** là chỉ số định lượng mức độ lão hóa của pin:

```
SoH (%) = (Dung lượng hiện tại / Dung lượng định mức ban đầu) × 100
        = (capacity_ahr / 2.0 Ahr) × 100
```

- Pin mới xuất xưởng: SoH = 100%
- End-of-Life (EOL): SoH = 70% (mất 30% dung lượng)

### 1.2 Hạn chế của các phương pháp đơn lẻ

| Phương pháp | Ưu điểm | Nhược điểm |
|---|---|---|
| Discharge curve | Phản ánh trực tiếp suy giảm dung lượng | Bỏ qua thay đổi điện trở nội |
| EIS (tổng trở) | Phản ánh thay đổi điện trở nội | Thiếu thông tin suy giảm dung lượng |
| Physics model | Có lý thuyết nền tảng | Chỉ là mô hình xấp xỉ, sai số cao |

### 1.3 Giải pháp: Multimodal Fusion

**BaFuse** kết hợp đồng thời cả 3 nguồn tín hiệu bằng **Cross-Attention Fusion**,
giữ lại thông tin bổ sung từ từng modalité và loại trừ nhược điểm của từng nguồn.

---

## 2. Dữ liệu thô (NASA PCoE Battery Aging Dataset)

### 2.1 Nguồn gốc

Dataset từ **NASA Prognostics Center of Excellence**, gồm **34 viên pin lithium-ion
18650** được lão hóa nhân tạo trong phòng thí nghiệm. Mỗi pin được chạy qua hàng
trăm chu kỳ charge–discharge–impedance đến khi đạt EOL.

- **Dung lượng định mức:** 2 Ahr
- **Ngưỡng EOL:** SoH = 70% (còn 1.4 Ahr)
- **Nhiệt độ:** Phòng (24°C), lạnh (4°C), cao (43–44°C) tùy campaign
- **Tổng số pin:** 34 (B0005–B0056, một số số không liên tục)

### 2.2 Cấu trúc file .mat

Mỗi file `.mat` (MATLAB) là một viên pin, chứa struct nhiều cấp:

```
B0005 (struct)
  └── cycle (array of N cycles)
        ├── type: "charge" | "discharge" | "impedance"
        ├── ambient_temperature (°C)
        ├── time (MATLAB datenum)
        └── data (struct, khác nhau theo type)
```

### 2.3 Dữ liệu thô trong chu kỳ DISCHARGE

| Field | Đơn vị | Mô tả |
|---|---|---|
| `Voltage_measured` | V | Điện áp đầu cực pin theo thời gian |
| `Current_measured` | A | Dòng xả (âm: đang xả) |
| `Temperature_measured` | °C | Nhiệt độ pin |
| `Time` | s | Vector thời gian của chu kỳ |
| `Capacity` | Ahr | **Dung lượng tích lũy** tăng dần từ 0 đến giá trị max |

**Ý nghĩa vật lý:** Khi pin lão hóa, Capacity max giảm dần qua các chu kỳ.
Đây chính là giá trị dùng để tính SoH label.

**Ví dụ trajectory của B0005:**

```
Chu kỳ 1:    Capacity max = 1.85 Ahr  -->  SoH = 92.7%
Chu kỳ 50:   Capacity max = 1.72 Ahr  -->  SoH = 86.0%
Chu kỳ 100:  Capacity max = 1.55 Ahr  -->  SoH = 77.5%
Chu kỳ 168:  Capacity max = 1.36 Ahr  -->  SoH = 68.0%  -->  EOL
```

### 2.4 Dữ liệu thô trong chu kỳ IMPEDANCE (EIS)

| Field | Đơn vị | Mô tả |
|---|---|---|
| `Battery_impedance` | Ω (phức) | Tổng trở từ 0.1 Hz đến 5 kHz |
| `Rectified_impedance` | Ω (phức) | Tổng trở sau hiệu chỉnh |
| `Re` | Ω | Điện trở điện giải (electrolyte resistance) |
| `Rct` | Ω | Điện trở truyền điện tích (charge transfer resistance) |

**Lưu ý quan trọng:** `Battery_impedance` là **số phức** (complex number).
Phần thực Re(Z) là điện trở, phần ảo Im(Z) liên quan đến hiện tượng điện hóa
như phổ Warburg và Rct.

**Các điều chỉnh đã thực hiện khi xử lý:**
- Tính magnitude `|Z| = sqrt(Re² + Im²)` TRƯỚC khi tính median
- Lọc cứng vật lý: `[0, 1000] Ω`
- Lọc mềm: percentile `[1%, 99%]` để loại outlier
- Re được clip về >= 0 (điện trở điện giải không âm)

### 2.5 Cutoff voltage khác nhau giữa các battery

Đây là điều ít tài liệu đề cập: mỗi campaign thí nghiệm dùng **ngưỡng cắt khác nhau**
khi xả pin. Điều này ảnh hưởng trực tiếp đến voltage_droop feature.

| Battery | Cutoff (V) | Battery | Cutoff (V) |
|---|---|---|---|
| B0005 | 2.7 | B0029, B0033, B0041, B0045, B0049, B0053 | 2.0 |
| B0006 | 2.5 | B0030, B0034, B0042, B0046, B0050, B0054 | 2.2 |
| B0007 | 2.2 | B0031, B0036, B0043, B0047, B0051, B0055 | 2.5 |
| B0018 | 2.5 | B0032, B0040, B0044, B0048, B0052, B0056 | 2.7 |
| B0025 | 2.0 | B0038 | 2.2 |
| B0026 | 2.2 | B0039 | 2.5 |
| B0027 | 2.5 | | |
| B0028 | 2.7 | | |

---

## 3. Định nghĩa Features

### 3.1 Discharge Features — từ time-series

Mỗi chu kỳ discharge có ~200–500 time-steps. Dữ liệu được **resample về 100 điểm**
bằng nội suy tuyến tính trước khi đưa vào LSTM.

**3 features tại mỗi time-step:**

| Feature | Ký hiệu | Đơn vị | Mô tả |
|---|---|---|---|
| Voltage | `voltage_v` | V | Điện áp đầu cực tại từng thời điểm |
| Current | `current_a` | A | Dòng xả |
| Temperature | `temperature_c` | °C | Nhiệt độ pin |

**Lý do chọn 3 features này:**
- Voltage phản ánh mức độ suy giảm dung lượng (pin yếu, voltage giảm nhanh hơn)
- Current phục vụ Coulomb counting
- Temperature ảnh hưởng đến hiệu suất và tốc độ lão hóa

### 3.2 EIS Features — scalar

| Feature | Ký hiệu | Đơn vị | Công thức | Mô tả |
|---|---|---|---|---|
| Impedance magnitude | `impedance_ohm` | Ω | median(\|Z\|) qua tất cả tần số | Tổng trở toàn phần |
| Electrolyte resistance | `re_ohm` | Ω | median(Re(Z)) | Điện trở dung dịch điện giải, tăng nhẹ khi lão hóa |
| Charge transfer resistance | `rct_ohm` | Ω | median(\|Im(Z)\|) | Điện trở giao diện điện cực–điện giải, tăng rõ khi lão hóa |

**Normalized:** `impedance_ohm` được z-score theo train stats.
`re_ohm` và `rct_ohm` giữ nguyên (đơn vị Ω, giá trị nhỏ ~0.01–0.3 Ω).

### 3.3 Physics-Informed Features — tính toán

Đây là phần có nhiều thay đổi nhất trong quá trình phát triển.

#### 3.3.1 Feature 1: cycle_age_norm

```
cycle_age_norm = discharge_cycle / N_ref
```

- `discharge_cycle`: số thứ tự của chu kỳ trong dataset (0, 1, 2, ...)
- `N_ref`: giá trị percentile 90 của chu kỳ trong tập train (thay đổi theo fold)
- Ý nghĩa: "pin này đã xả được bao nhiêu % vòng đời dự kiến"

#### 3.3.2 Feature 2: empirical_fade_prior (Physics-Informed)

Đây là điểm then chốt phân biệt BaFuse với các mô hình thông thường:

**Công thức:**
```
empirical_fade_prior = A × (cycle_age / N_ref)^b
```

**Cách tính A, b:**
Được fit bằng **log-log regression** trên toàn bộ tập TRAIN (không dùng tập test):

```python
fade_fraction = (100 - SoH) / 100         # từ tập train
age_norm      = cycle_age / N_ref

# Log-log linear fit:
# log(fade) = log(A) + b × log(age_norm)
b, log_A = polyfit(log(age_norm), log(fade), 1)
A = exp(log_A)
```

**Kết quả fit trên tập train thực tế (tùy fold):**
- A ≈ 0.21–0.24 (mức độ fade tại N_ref cycles)
- b ≈ 0.10–0.14 (tốc độ fade — gần tuyến tính với log)
- N_ref ≈ 480–530 cycles (percentile 90 của tập train)

**Lý do đây là physics-informed:**
- Prior này được xây dựng từ domain knowledge (pin Li-ion lão hóa theo power-law)
- Được calibrate trên tập train, KHÔNG dùng capacity thực của từng mẫu cụ thể
- Tại inference: chỉ cần biết `cycle_age` là tính được prior
- Không có data leakage vì prior không biết SoH thực của mẫu đang predict

**So sánh với phiên bản cũ (có leakage):**
```python
# PHIÊN BẢN CŨ — CÓ LEAKAGE
capacity_fade = 2.0 - capacity_ahr   # feature
soh_label     = (capacity_ahr / 2.0) × 100  # label
# --> capacity_fade = 2 - 2×(soh_label/100) --> tương quan -1.0

# PHIÊN BẢN HIỆN TẠI — KHÔNG LEAKAGE
empirical_fade_prior = A × (cycle_age / N_ref)^b  # chỉ dùng cycle_age
```

#### 3.3.3 Feature 3: voltage_droop

```
voltage_droop = (voltage_min - cutoff_v) / (4.2 - cutoff_v)
```

- `voltage_min`: điện áp thấp nhất trong chu kỳ discharge
- `cutoff_v`: ngưỡng cắt điện áp của battery cụ thể (lấy từ README)
- `4.2 - cutoff_v`: toàn bộ cửa sổ discharge (từ full charge đến cutoff)

**Ý nghĩa:** Do pin lão hóa, đường cong voltage giảm nhanh hơn và chạm ngưỡng
sớm hơn. `voltage_droop` đo mức độ "trượt" của điện áp so với ngưỡng cắt.
Giá trị gần 0 = pin còn khỏe, giá trị âm nhiều = pin lão hóa.

**Tại sao phải dùng cutoff riêng theo battery:**
B0053 được xả xuống 2.0V (thay vì 2.7V của B0005). Nếu dùng cutoff cố định 2.7V,
B0053 luôn có voltage_min = 1.97V < 2.7V → voltage_droop rất âm dù pin còn khỏe,
gây hiệu ứng giả.

#### 3.3.4 Feature 4: impedance_rise

```
impedance_rise = (impedance_ohm - train_mean_imp) / train_std_imp
```

Z-score của tổng trở so với phân phối trong tập train.
Giá trị dương lớn = tổng trở cao = pin lão hóa.

---

## 4. Input / Output của mô hình

### 4.1 Input

```
discharge:  Tensor (B, 100, 3)
            B = batch size
            100 = số time-steps (sau resample)
            3 = [voltage_v, current_a, temperature_c]

eis:        Tensor (B, 3)
            [impedance_ohm_norm, re_ohm, rct_ohm]

physics:    Tensor (B, 4)
            [cycle_age_norm, empirical_fade_prior,
             voltage_droop, impedance_rise]
```

### 4.2 Output

```
soh_pred:           Tensor (B, 1)        -- SoH dự đoán trong [0, 100]
discharge_latent:   Tensor (B, 64)       -- latent vector của discharge branch
eis_latent:         Tensor (B, 64)       -- latent vector của EIS branch
physics_latent:     Tensor (B, 64)       -- latent vector của physics branch
fused:              Tensor (B, 128)      -- fused representation sau attention
attention_weights:  Tensor (B, 4, 3, 3) -- attention map (4 heads, 3×3 modalities)
```

---

## 5. Xử lý dữ liệu

### 5.1 Parse file .mat

```
File .mat (MATLAB format)
  --> scipy.io.loadmat()
  --> Duyệt struct cycle[], lọc chỉ lấy 'discharge' và 'impedance'
  --> Trích xuất time-series
  --> Tính cycle_capacity = max(Capacity_ts) hoặc Coulomb counting
```

**Vấn đề kỹ thuật đã giải quyết:**
- MATLAB struct trong Python tồn tại dưới dạng `numpy.void` (structured scalar)
- Phải dùng `.flat[0]` để lấy scalar từ array `(1,1)`
- Số phức trong impedance: xử lý bằng `np.abs()` trước

### 5.2 Lọc dữ liệu

**Lọc 1 — Cycle dung lượng bất thường:**
```
Loại bỏ: capacity_ahr < 0.5 Ahr
Lý do:   Lỗi thiết bị đo, không phải suy giảm pin thật
Kết quả: 185 cycle bị loại trên 13 battery
```

**Lọc 2 — Impedance outlier:**
```
Hard filter:  |Z| ngoài [0, 1000] Ω
Soft filter:  percentile [1%, 99%]
Clip:         Re(Z) >= 0
Lý do:        Số phức + chia gần 0 ở tần số thấp tạo giá trị vô lý
Kết quả:      impedance_range từ (-1161, 1.47e14) --> (0.15, 648) Ω
```

**Lọc 3 — Loại battery duplicate:**
```
B0025–B0028 có trong cả thư mục 2 và 3 --> chỉ parse 1 lần
Kết quả: 835,422 --> 770,070 discharge records (chính xác)
```

### 5.3 Ghép cặp Discharge ↔ EIS

Pattern trong NASA PCoE:
```
Discharge (cycle N) --> Charge (N+1) --> Impedance (N+2)
```

**Quy tắc ghép:**
```
Với mỗi discharge cycle N:
  Tìm EIS trong window [N, N+10]  (max_cycle_gap = 10)
  Nếu không có --> bỏ qua (không fallback vô hạn)

Kết quả: 72 discharge cycle không ghép được --> loại bỏ
cycle_gap thực tế: 98% gap=1, max=2 (rất sạch)
```

### 5.4 Resample discharge curve

```
Thô: ~200–500 time-steps (khác nhau giữa các battery/campaign)
Resample:   np.interp trên 100 điểm đều
Kết quả:    Đồng nhất, tốc độ training nhanh ~6 lần (45 giây/epoch thay vì 4 phút)
```

### 5.5 Train/Val/Test Split

**Nguyên tắc: Battery-level split** (không trộn các battery giữa các split):
```
Train: 20 battery (629 samples)
Val:    7 battery (103 samples)
Test:   7 battery (223 samples)
```

**Stratification:** Sắp xếp battery theo mean capacity → chia round-robin đảm bảo
mỗi split có đại diện từ pin mới (SoH cao) đến pin cũ (SoH thấp).

**Normalization:** Train stats (mean/std) được tính một lần từ tập train,
áp dụng cho cả val và test (tránh distribution shift).

---

## 6. Kiến trúc mô hình

### 6.1 Tổng quan

```
Discharge (100×3) --> DischargeEncoder (LSTM)   --> latent_d (64) --+
EIS (3)           --> EISEncoder (MLP)           --> latent_e (64) --+--> CrossAttentionFusion --> SoH Head --> SoH%
Physics (4)       --> PhysicsEncoder (MLP)       --> latent_p (64) --+
```

### 6.2 DischargeEncoder — LSTM

```
Input:  (B, 100, 3)
LSTM(input=3, hidden=256, num_layers=3, dropout=0.2, batch_first=True)
  --> lấy h_last = hidden state cuối cùng: (B, 256)
Linear(256->64) + LayerNorm(64) + ReLU
Output: (B, 64)
```

Tổng params: ~1,115,648

### 6.3 EISEncoder — MLP

```
Input: (B, 3)
Linear(3->128) + ReLU + Dropout(0.2)
Linear(128->64) + LayerNorm(64) + ReLU
Output: (B, 64)
```

Tổng params: ~8,384

### 6.4 PhysicsEncoder — MLP

```
Input: (B, 4)
Linear(4->256) + ReLU + Dropout(0.2)
Linear(256->128) + ReLU + Dropout(0.2)
Linear(128->64) + LayerNorm(64) + ReLU
Output: (B, 64)
```

Tổng params: ~50,560

### 6.5 CrossAttentionFusion

```
Stack 3 latent vectors --> tokens: (B, 3, 64)

MultiheadAttention(embed_dim=64, num_heads=4, dropout=0.1)
  Q = K = V = tokens
  --> attended: (B, 3, 64)
  --> attn_weights: (B, 4, 3, 3)

Residual: attended = LayerNorm(attended + tokens)

Flatten: (B, 3, 64) --> (B, 192)
Linear(192->128) + ReLU + Dropout(0.1)
Output: fused (B, 128)
```

Tổng params: ~74,240

**Ý nghĩa của self-attention:** Mỗi modalité có thể "hỏi" thông tin từ
2 modalité còn lại. Attention weight thể hiện mối quan hệ nào quan trọng nhất
(ví dụ: discharge có cần thêm thông tin từ physics không?).

### 6.6 SoH Prediction Head

```
Input: fused (B, 128)
Linear(128->256) + ReLU + Dropout(0.2)
Linear(256->128) + ReLU + Dropout(0.2)
Linear(128->1)
Output: soh_pred (B, 1)  -- range [0, 100]
```

Tổng params: ~66,177

### 6.7 Tổng số tham số: 1,495,489 (~1.5M)

---

## 7. Hàm loss

```
Loss = MSE(soh_pred, soh_target)
     + 0.1 × physics_penalty
     + 0.05 × monotonicity_loss
```

**Physics penalty:**
```
physics_penalty = mean(relu(-pred) + relu(pred - 100))
```
Phạt dự đoán ngoài khoảng vật lý hợp lệ [0, 100].

**Monotonicity loss (within-battery only):**
```
Với mỗi battery_id trong batch:
  Với mỗi cặp chu kỳ (i, j) của CÙNG battery đó mà age[i] < age[j]:
    Phạt nếu pred[i] < pred[j]  (pin trẻ dự đoán SoH thấp hơn pin già)

loss = sum(relu(pred[i] - pred[j]) × mask) / n_pairs
```

**Lý do chỉ trong cùng battery:**
Battery khác nhau có tốc độ lão hóa khác nhau. Battery A cycle 50 có thể có
SoH cao hơn battery B cycle 10 nếu A có quy trình sử dụng nhẹ hơn.
So sánh cross-battery là sai về logic vật lý.

---

## 8. Metric đánh giá

### 8.1 Các metric chính

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| MAE | mean(\|pred - true\|) | Sai số tuyệt đối trung bình (%, dễ hiểu) |
| RMSE | sqrt(mean((pred-true)²)) | Phạt nặng sai số lớn, nhạy với outlier |
| R² | 1 - SS_res/SS_tot | Hệ số xác định: 1=hoàn hảo, 0=bằng mean, <0=tệ hơn mean |
| MAPE | mean(\|pred-true\|/\|true\|)×100 | Sai số phần trăm tương đối |

### 8.2 Cách đánh giá

**Single-split (60/20/20):**
```
Test MAE:   7.746%
Test RMSE: 11.156%
Test R^2:   0.499
Test MAPE: 14.42%
```

**5-fold Battery-Group Cross-Validation:**
- Chia 34 battery thành 5 nhóm, stratified by mean capacity
- Mỗi fold: train trên 4 nhóm, test trên 1 nhóm
- Báo cáo mean ± std

**Phân loại battery theo trajectory:**

| Nhóm | Tiêu chí | Số battery | MAE (CV) |
|---|---|---|---|
| Sufficient | >= 50 cycles | 3 (B0005/6/7) | 4.04 ± 0.66% |
| Intermediate | 20–49 cycles | 12 | ~8–10% |
| Insufficient | < 20 cycles | 19 | 10.87 ± 1.41% |

**Lý do tách nhóm:** Battery ít chu kỳ không có đủ trajectory để mô hình học
xu hướng lão hóa → MAE cao hơn không phải do mô hình kém mà do data hạn chế.

---

## 9. Ablation Study — Đóng góp của từng modalité

### 9.1 Phương pháp

**Zero-out ablation:** Thay vì train lại từng mô hình đơn lẻ (tốn thời gian),
ta zero-out output của từng encoder riêng lẻ lúc inference và đo sự tăng MAE:

```python
# Ablate discharge
d = torch.zeros_like(batch["discharge"])
mae_no_discharge = evaluate(model, d, eis, physics)

# Ablate EIS
e = torch.zeros_like(batch["eis"])
mae_no_eis = evaluate(model, discharge, e, physics)

# Ablate physics
p = torch.zeros_like(batch["physics"])
mae_no_physics = evaluate(model, discharge, eis, p)
```

**Tính đóng góp:**
```
delta_discharge = max(mae_no_discharge - base_mae, 0)
delta_eis       = max(mae_no_eis       - base_mae, 0)
delta_physics   = max(mae_no_physics   - base_mae, 0)
total = delta_discharge + delta_eis + delta_physics

contribution_discharge = delta_discharge / total
```

### 9.2 Kết quả ablation (sau 100 epochs)

| Metric | Giá trị |
|---|---|
| base_mae (full model) | 7.746% |
| mae_no_discharge | 17.532% (+9.786) |
| mae_no_eis | 7.864% (+0.118) |
| mae_no_physics | 10.764% (+3.018) |

| Modalité | Đóng góp |
|---|---|
| **Discharge** | **75.7%** |
| **Physics** | **23.4%** |
| EIS | 0.9% |

### 9.3 Phân tích kết quả

**Discharge 75.7%:**
Hoàn toàn hợp lý. Discharge curve là nguồn thông tin giàu nhất về SoH:
voltage sagging, capacity fade đều hiển thị rõ qua đường cong discharge.
LSTM với 1.1M params học được các pattern phức tạp này.

**Physics 23.4%:**
Empirical aging prior cung cấp một "anchor" thống kê — mô hình biết
rằng pin có cycle_age cao thì nên có SoH thấp hơn. Voltage_droop và
impedance_rise bổ sung thêm context về trạng thái hiện tại.

**EIS 0.9%:**
Kết quả này được xác nhận bằng sklearn baseline:
- EIS-only MAE: 9.56%
- Discharge-only MAE: 7.21%

EIS có tương quan với SoH (r = −0.37) nhưng yếu hơn discharge trong dataset này.
Lý do: impedance chỉ thay đổi ~0.03 Ω qua vòng đời pin, trong khi voltage
thay đổi hàng trăm mV. CrossAttention tự học trọng số thấp cho EIS.

**Đây là finding học thuật hợp lệ**, không phải lỗi mô hình: NASA PCoE
discharge curve đã chứa đủ thông tin, EIS không bổ sung nhiều thêm.

### 9.4 So sánh 4 cấu hình trong CV

| Config | MAE (all) | MAE (sufficient) | R² | Ghi chú |
|---|---|---|---|---|
| (a) baseline | 7.947 ± 2.226% | 3.775 ± 0.802% | 0.174 | Bug cutoff + cross-bat mono |
| (b) +cutoff_fix | 7.437 ± 1.462% | 3.961 ± 0.441% | 0.322 | Dùng cutoff đúng theo battery |
| **(c) +mono_fix** | **7.191 ± 0.668%** | **4.559 ± 1.080%** | **0.377** | **BEST: within-battery mono** |
| (d) +cnn1d | 8.243 ± 1.240% | 5.284 ± 1.093% | 0.259 | CNN1D kém hơn MLP |

**Kết luận:**
- cutoff_fix: giảm MAE 0.51%, giảm std 0.76%
- mono_fix: giảm std thêm 0.79% (giảm 54% so với baseline) — chủ yếu là ổn định
- CNN1D: kém hơn MLP 1.05% — physics vector 4-dim không có cấu trúc sequential

---

## 10. Kết quả cuối cùng

### 10.1 Single split (60/20/20)

```
Test MAE:   7.746%
Test RMSE: 11.156%
Test R²:    0.499
Test MAPE: 14.42%

Per-battery:
  B0005: MAE=5.14%  R²=0.545  (sufficient, 142 cycles)
  B0048: MAE=5.12%  R²=0.286
  B0053: MAE=2.45%  R²=-3.37  (outlier: SoH range 4.5%, cutoff=2.0V)
  B0026: MAE=10.49% R²=-1.96  (9 cycles — quá ít)
  B0039: MAE=18.84% R²=-0.04  (13 cycles)
  B0045: MAE=18.71% R²=-9.08  (short trajectory)
  B0051: MAE=25.06% R²=-0.29  (6 cycles)
```

**Ghi chú B0053:** MAE = 2.45% nhưng R² = −3.37 vì SoH range chỉ 4.5%
(64.5–69%). SS_tot rất nhỏ nên bất kỳ sai số nhỏ nào cũng làm R² âm.

### 10.2 5-fold CV (biến số tin cậy hơn)

```
Tất cả battery:
  MAE  = 7.191 ± 0.668%
  RMSE = 10.112 ± 0.518%
  R²   = 0.377 ± 0.116

Battery đủ dữ liệu (>= 50 cycles, B0005/6/7):
  MAE  = 4.04 ± 0.66%
  --> Mô hình thực sự hoạt động tốt khi có đủ dữ liệu
```

---

## 11. Cách chạy

```powershell
# Chạy full pipeline (parse -> train -> evaluate)
.\bafuse_env\Scripts\python.exe run_pipeline.py --epochs 100 --patience 20 --batch_size 32

# 5-fold CV với LSTM size comparison
.\bafuse_env\Scripts\python.exe scripts\cross_validate.py --folds 5 --epochs 40

# So sánh 4 configs (cutoff/mono/cnn1d)
.\bafuse_env\Scripts\python.exe scripts\cv_compare.py

# Kiểm tra chất lượng dữ liệu
.\bafuse_env\Scripts\python.exe scripts\validate_pipeline.py

# Phân tích EIS signal
.\bafuse_env\Scripts\python.exe scripts\investigate_eis.py
```
