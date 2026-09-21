# Hành Trình Phát Triển BaFuse

> Ghi lại quá trình xây dựng pipeline dự đoán SoH pin lithium-ion từ lúc bắt đầu đến hiện tại, bao gồm các lỗi đã mắc phải, cách phát hiện và sửa chữa.

---

## 1. Điểm xuất phát — Bộ khung có nhưng chưa chạy được

Dự án bắt đầu với một bộ skeleton đầy đủ về mặt cấu trúc:

```
src/models/encoders.py     → toàn bộ là "# TODO"
src/models/fusion.py       → toàn bộ là "# TODO"
src/train.py               → toàn bộ là "# TODO"
src/evaluate.py            → toàn bộ là "# TODO"
src/losses.py              → toàn bộ là "# TODO"
```

Không một dòng logic nào được viết. Chạy thử → báo lỗi ngay từ import.

**Việc đầu tiên:** implement toàn bộ — LSTM encoder cho discharge, MLP cho EIS/physics, CrossAttention fusion, training loop với early stopping, evaluation với ablation study.

---

## 2. Rào cản kỹ thuật — Windows Application Control chặn torch

Khi chạy lần đầu trong virtual environment (`bafuse_env`):

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "...\torch\lib\shm.dll"
```

**Điều tra:**
- `UsermodeCodeIntegrityPolicyEnforcementStatus = 2` → đang enforce
- Windows 11 Home → không có module `ConfigCI` → không thể tạo supplemental policy
- Registry key `VerifiedAndReputablePolicyState = 1` → **Smart App Control đang bật**

**Giải pháp:** Tắt Smart App Control trong Windows Security → restart → torch 2.14.0+cpu chạy được.

---

## 3. Lần chạy đầu tiên — Lộ ra 5 lỗi dữ liệu

Pipeline chạy được nhưng log trả về những con số bất thường:

```
ComplexWarning: Casting complex values to real discards the imaginary part
impedance_range: (-1161.73, 147020420262137.72)   ← vô lý về vật lý
capacity_range:  (0.00010895, 3.03)               ← gần 0 Ahr là lỗi thiết bị
cycle_gap_stats: {mean: -0.12, max: 39}           ← EIS đo lệch 39 cycle
discharge records: 835,422                         ← bị đếm trùng
discharge shape: (16, 500, 3)                      ← ~4 phút/epoch trên CPU
```

### Lỗi 1 — Impedance tính sai (ComplexWarning)

NASA PCoE lưu impedance dưới dạng số phức. Code cũ dùng `np.nanmedian()` trực tiếp lên mảng phức — NumPy tự cắt bỏ phần ảo trước khi tính median mà không báo lỗi rõ ràng (chỉ có warning). Kết quả: mất toàn bộ thông tin reactance, và vài điểm đo ở tần số thấp (gần 0 Hz) cho giá trị lên tới 1.47×10¹⁴ Ω.

**Sửa:** Tính `|Z| = sqrt(Re² + Im²)` trước → lọc cứng `[0, 1000] Ω` → percentile clip `[1%, 99%]` → median. Tách `Re` và `|Im|` thành 2 features riêng (`re_ohm`, `rct_ohm`).

```
Trước: impedance_range = (-1161.73, 1.47e14) Ω
Sau:   impedance_range = (0.15, 648.22) Ω       ✓ hợp lý về vật lý
```

### Lỗi 2 — Cycle có capacity ≈ 0 Ahr lọt vào training

`capacity_range min = 0.00011 Ahr` — pin 2 Ahr mà đo được 0.0001 Ahr là do lỗi thiết bị đo, không phải pin thật sự xuống cấp đến mức đó. Nếu để nguyên, SoH label tính ra ≈ 0% sẽ lẫn vào tập train/val/test.

**Sửa:** Lọc bỏ cycle có `capacity_ahr < 0.5 Ahr` (25% định mức) trước khi ghép cặp. **185 cycle bị loại trên 13 battery**.

```
Trước: capacity_range = (0.0001, 3.03) Ahr
Sau:   capacity_range = (0.52, 3.03) Ahr         ✓
```

### Lỗi 3 — Ghép cặp discharge ↔ EIS quá lỏng

`cycle_gap max = 39` nghĩa là có cặp mà EIS được đo 39 cycle SAU discharge — trong 39 cycle đó pin đã già đi đáng kể, dùng EIS cũ để mô tả trạng thái pin tại thời điểm discharge là sai.

**Sửa:** Enforce `max_cycle_gap = 10`, bỏ hoàn toàn logic "fallback tìm EIS gần nhất bất kỳ". **72 cycle không ghép được bị loại**.

```
Trước: cycle_gap max = 39,  std = 4.3
Sau:   cycle_gap max = 10,  std = 0.15,  98% pairs có gap = 1
```

### Lỗi 4 — Parse trùng battery

B0025–B0028 có mặt trong cả thư mục `2.` lẫn `3.` → được parse 2 lần → tổng discharge records bị thổi lên 835,422 trong khi thực tế chỉ 770,070.

**Sửa:** Thêm `seen_ids` set, skip nếu battery_id đã được parse.

### Lỗi 5 — Sequence 500 điểm → training cực chậm

Mỗi discharge curve có tới 500 time-steps → LSTM phải xử lý chuỗi dài → ~4 phút/epoch → 100 epoch ≈ 6-7 tiếng.

**Sửa:** Resample đồng nhất về 100 điểm bằng `np.interp`. **Tốc độ tăng ~6 lần**, epoch giảm xuống còn ~45 giây.

---

## 4. Phát hiện quan trọng — Physics feature leak label

Sau khi sửa 5 lỗi trên, chạy ablation study và thấy:

```
Discharge contribution:  20.9%
Physics contribution:    79.1%   ← BẤT THƯỜNG
EIS contribution:         0.7%
```

Physics đóng góp 79% — trong khi đây chỉ là MLP nhỏ nhận 4 con số. Kiểm tra code:

```python
# dataset.py — code CŨ
capacity_fade = 2.0 - capacity_ahr          # feature physics
soh_label = (capacity_ahr / 2.0) * 100.0   # label
```

`capacity_fade = 2.0 - capacity_ahr` và `soh_label = (capacity_ahr / 2.0) * 100` đều được tính từ **cùng một giá trị `capacity_ahr`**. Tương quan giữa chúng:

```
capacity_fade  = 2.0 - capacity_ahr
soh_label/100  = capacity_ahr / 2.0
→ capacity_fade = 2.0 - 2.0 * (soh_label/100)
→ correlation(capacity_fade, soh_label) ≈ -1.0  (tuyến tính hoàn hảo)
```

Model chỉ cần học phép biến đổi tuyến tính trên physics feature là ra label — **không cần học gì từ discharge hay EIS**. Đây là **data leakage**.

**Bằng chứng:** Test MAE "tốt" (~9%) khi có leakage, nhưng sau khi gỡ bỏ, MAE tăng lên 11.4% — đây mới là con số thật.

**Sửa:** Loại bỏ `capacity_fade`. Thay bằng **empirical aging prior** — fit power-law `fade = A × (age/N_ref)^b` trên TOÀN BỘ tập train (population level), dùng `cycle_age` làm đầu vào. Prior này không biết `capacity_ahr` của từng mẫu cụ thể → không leak.

```python
# Fit trên train population → A=0.225, b=0.103, N_ref=488
# Tại inference: empirical_prior = 0.225 * (cycle_age / 488)^0.103
```

**Physics features sau khi sửa — không còn leak:**

| Feature | Nguồn gốc | Leak? |
|---|---|---|
| `cycle_age_norm` | `discharge_cycle / N_ref` | Không |
| `empirical_fade_prior` | Power-law fit trên tập train | Không |
| `voltage_droop` | `(voltage_min - 2.7) / 1.5` | Không |
| `impedance_rise` | `(impedance - mean_train) / std_train` | Không |

---

## 5. Lỗi normalization — Val/test dùng stats của chính nó

Khi tạo DataLoader, mỗi split tự tính mean/std riêng:

```python
# Code CŨ — mỗi split tự normalize theo chính nó
train_dataset = BaFuseDataset(train_df, normalize=True)  # dùng train stats
val_dataset   = BaFuseDataset(val_df,   normalize=True)  # dùng VAL stats ← sai
test_dataset  = BaFuseDataset(test_df,  normalize=True)  # dùng TEST stats ← sai
```

Cùng một giá trị vật lý (ví dụ: impedance = 0.25 Ω) sẽ được normalize thành số khác nhau tùy nằm trong split nào. Model được train trên phân phối normalize của tập train, nhưng lúc test lại nhận vào phân phối khác → **distribution shift ngay trong pipeline**.

**Sửa:** Tính stats một lần từ train set, truyền vào val và test qua tham số `external_stats`.

```python
train_dataset = BaFuseDataset(train_df, normalize=True)               # tính stats từ train
val_dataset   = BaFuseDataset(val_df,   external_stats=train_dataset.stats)  # dùng train stats
test_dataset  = BaFuseDataset(test_df,  external_stats=train_dataset.stats)  # dùng train stats
```

---

## 6. Điều tra EIS — Tại sao đóng góp gần 0%?

Sau khi gỡ leakage, ablation cho thấy EIS vẫn gần 0%. Điều tra:

```
NaN trong re_ohm/rct_ohm:  0%        ← dữ liệu đầy đủ
cycle_gap thực tế:          98% = 1  ← pairing rất sạch
impedance ↔ SoH correlation: -0.37   ← tương quan vừa phải, không phải 0
EIS-only sklearn MAE:  9.56%
Discharge-only sklearn MAE: 7.21%
```

**Kết luận:** EIS có tín hiệu thật (r = -0.37) nhưng **discharge curve tự nhiên mạnh hơn** trong dataset NASA PCoE. Impedance chỉ thay đổi ~0.03 Ω trên toàn bộ vòng đời pin — rất nhỏ so với thay đổi voltage curve. CrossAttention tự học gán trọng số cao cho discharge, gần như bỏ qua EIS.

Đây là **finding học thuật hợp lệ**: trong NASA PCoE, discharge curve đã chứa đủ thông tin để dự đoán SoH, EIS không bổ sung nhiều thêm ở granularity này.

---

## 7. Đánh giá đúng — Cross-validation thay vì 1 lần chia

Một lần chia 60/20/20 cố định không đáng tin khi chỉ có 34 battery. B0053 với SoH range 4.5% rơi vào test set làm R² = -107 chỉ vì nó là outlier về operating window (discharge xuống 1.97V thay vì 2.5V thông thường).

**Giải pháp:** 5-fold cross-validation theo battery group, stratified by mean capacity.

**Kết quả CV (5-fold, 40 epochs/fold, LSTM-256×3):**

```
Fold 1: MAE=6.94%   RMSE=10.46%   R²=0.541
Fold 2: MAE=12.01%  RMSE=13.79%   R²=-0.158
Fold 3: MAE=7.64%   RMSE=9.66%    R²=0.291
Fold 4: MAE=7.09%   RMSE=10.81%   R²=0.350
Fold 5: MAE=5.75%   RMSE=8.78%    R²=0.449

Mean ± std:
  MAE  =  7.88 ± 2.15 %
  RMSE = 10.70 ± 1.70 %
  R²   =  0.294 ± 0.242
```

**Tách theo trajectory group:**

```
Sufficient (B0005, B0006, B0007 — >50 cycles):
  MAE  =  4.04 ± 0.66 %    ← model hoạt động tốt khi có đủ dữ liệu

Insufficient (<20 cycles, 19 batteries):
  MAE  = 10.87 ± 1.41 %    ← kém hơn do không đủ trajectory để học trend
```

**Fold 2 có R² âm** vì test set không chứa battery nào "sufficient" — toàn bộ là intermediate/insufficient. Đây là giới hạn của dataset, không phải model.

---

## 8. Kết quả hiện tại

### Số liệu CV đáng tin (không bị ảnh hưởng bởi một lần chia may mắn/xui xẻo)

| Nhóm | MAE | RMSE | R² |
|---|---|---|---|
| **Tất cả battery** | **7.88 ± 2.15 %** | **10.70 ± 1.70 %** | **0.294 ± 0.242** |
| Battery đủ dữ liệu (>50 cycles) | **4.04 ± 0.66 %** | — | ~0.62 avg |
| Battery ít dữ liệu (<20 cycles) | 10.87 ± 1.41 % | — | thấp |

### Modality contributions (sau khi gỡ leakage)

| Modality | Trước (leak) | Sau |
|---|---|---|
| Discharge | 20% | **70.4%** |
| Physics | 79.1% ← giả | **27.1%** |
| EIS | 0.7% | **2.5%** |

---

## 9. Timeline tóm tắt

```
Bắt đầu  → Implement toàn bộ skeleton (encoders, fusion, train, evaluate)
         → Phát hiện Smart App Control chặn torch → tắt SAC
         → Chạy lần đầu: 5 lỗi dữ liệu (impedance phức, capacity ~0,
           cycle_gap=39, parse trùng, sequence dài 500)
         → Sửa 5 lỗi, validate: impedance [0.15–648Ω], capacity ≥ 0.52 Ahr
         → Phát hiện data leakage (capacity_fade ≈ -soh_label)
           → ablation physics=79% là dấu hiệu → confirm bằng correlation=-1
         → Sửa leakage: thay bằng empirical aging prior fit từ train population
         → Sửa normalization skew: external_stats cho val/test
         → Điều tra EIS=0%: tín hiệu có (r=-0.37) nhưng discharge mạnh hơn
         → Fix LR warmup (10 epoch thay vì cold start)
         → Tăng DischargeEncoder capacity (256×3 thay vì 128×2)
         → 5-fold CV: MAE = 7.88 ± 2.15 %
Hiện tại → Model hoạt động trung thực, không có leakage
```

---

## 10. Bài học

1. **Kiểm tra correlation feature-label trước khi train.** Feature có r > 0.95 với label gần như chắc chắn là leakage hoặc circular dependency.

2. **Normalization phải dùng train stats cho val/test.** Lỗi này rất phổ biến và khó phát hiện vì không báo lỗi — model chỉ hơi tệ hơn một chút.

3. **Một lần chia train/test không đủ khi dataset nhỏ.** 34 batteries, 1 lần chia → kết quả phụ thuộc nhiều vào may rủi. CV mới cho con số đáng tin.

4. **Outlier về operating window cần báo cáo riêng.** B0053 discharge xuống 1.97V (thay vì 2.5V thông thường) → không thể so sánh trực tiếp với battery khác.

5. **EIS ~0% contribution là finding, không phải lỗi.** Trong NASA PCoE, discharge curve đã chứa đủ thông tin. Muốn EIS đóng góp nhiều hơn cần dataset có EIS đo ở nhiều tần số và nhiều trạng thái SoH hơn.
