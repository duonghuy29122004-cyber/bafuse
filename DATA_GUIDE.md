# BaFuse Data Pipeline Guide

## Overview

The BaFuse data processing pipeline transforms raw NASA PCoE `.mat` files into PyTorch-ready datasets for training multimodal SoH estimation models.

## File Structure

### Input Data
```
5. BatteryDataSet/
├── 1. BatteryAgingARC-FY08Q4/
│   ├── B0005.mat        ← Battery #5 (charge/discharge/impedance cycles)
│   ├── B0006.mat        ← Battery #6
│   ├── B0007.mat        ← Battery #7
│   ├── B0018.mat        ← Battery #18
│   └── README.txt       ← Data structure description
├── 2. BatteryAgingARC_25_26_27_28_P1/
├── 3. BatteryAgingARC_25-44/
├── ... (additional campaigns)
```

### Output Data (after processing)
```
data/processed/
├── discharge_raw.pkl      ← All discharge measurements (raw time-series)
├── eis_raw.pkl            ← All EIS measurements
├── paired.pkl             ← Paired discharge-EIS samples
├── train.pkl              ← Training split (no battery leakage)
├── val.pkl                ← Validation split
└── test.pkl               ← Test split
```

## Data Processing Pipeline

### Step 1: Parse .mat Files
**File:** `src/data/parse_mat.py`

Extracts discharge curves and impedance (EIS) measurements from MATLAB .mat files.

**Key Functions:**
- `parse_discharge_curve()` - Extract voltage/current/temperature time-series
- `parse_eis_spectrum()` - Extract impedance, electrolyte resistance, charge-transfer resistance
- `parse_all_mat_files()` - Process all files in dataset directory

**Output Structure:**

Discharge DataFrame:
```
cycle_idx | time_s | voltage_v | current_a | temperature_c | capacity_ahr | battery_id
    0     |   0.1  |   4.18    |   2.0     |     25.0      |    2.00      |   B0005
    0     |   0.2  |   4.17    |   2.0     |     25.1      |    2.00      |   B0005
   ...
```

EIS DataFrame:
```
cycle_idx | impedance_ohm | re_ohm | rct_ohm | battery_id
    2     |     0.045     | 0.015  | 0.030   |   B0005
    5     |     0.050     | 0.016  | 0.034   |   B0005
   ...
```

### Step 2: Pair Discharge with EIS
**File:** `src/data/pairing.py`

Matches each discharge cycle with its corresponding EIS measurement.

**Pairing Logic:**
- NASA PCoE typical pattern: discharge (cycle N) → charge (N+1) → impedance (N+2)
- For each discharge, find nearest EIS (usually N+1 or N+2)
- Aggregate discharge time-series into features: voltage_mean/min/max/std, current_mean/std, temp_mean/std
- Keep cycle_gap for quality assessment

**Output:**
```
battery_id | discharge_cycle | eis_cycle | capacity_ahr | voltage_mean | ... | impedance_ohm | cycle_gap
   B0005   |       0         |     2     |    2.00      |    3.85      | ... |    0.045      |    2
   B0005   |       3         |     5     |    1.98      |    3.82      | ... |    0.050      |    2
```

### Step 3: Battery-Level Split
**File:** `src/data/split.py`

Ensures no data leakage by assigning entire batteries to train/val/test.

**Split Strategy:**
1. Compute mean capacity per battery (SoH proxy)
2. Stratify batteries by capacity ranges (high/medium/low degradation)
3. Randomly assign batteries: 60% train, 20% val, 20% test
4. Filter paired data by assigned batteries

**Why Battery-Level:**
- Each battery has hundreds of paired samples
- Mixing battery data across splits would leak information
- Model must generalize to unseen batteries

**Statistics:**
- ~4 batteries → typically 2-3 per split
- Each battery: 100-150 charge-discharge cycles
- Train: ~200-300 samples, Val/Test: ~100-150 samples each

### Step 4: Create PyTorch Datasets
**File:** `src/data/dataset.py`

Converts paired data into PyTorch Dataset/DataLoader for model training.

**Sample Features:**

1. **Discharge (3-5 features per cycle):**
   - Voltage mean/min/max (reflects capacity fade)
   - Current mean (discharge current)
   - Temperature mean

2. **EIS (3 features per measurement):**
   - Impedance (Ω) - total impedance increase
   - Re (Ω) - electrolyte resistance
   - Rct (Ω) - charge transfer resistance

3. **Physics-Informed (4 features):**
   - Cycle age / max_cycles (0-1 normalized)
   - Capacity fade / nominal_capacity (0-1)
   - Voltage droop indicator (V - 2.7V / 1.5V)
   - Impedance rise (normalized by std)

4. **Label:**
   - SoH (%) = (current_capacity_Ahr / 2.0) × 100
   - Range: ~100% (new) → ~70% (EOL at 30% fade)

**Normalization:**
- Computed from training set statistics
- Applied to all splits
- Prevents information leakage

## Quick Start

### Run Complete Pipeline

```bash
cd d:\Capstone project\bafuse
python process_data.py --data_dir "5. BatteryDataSet" --output_dir "data/processed"
```

**Output:**
```
BaFuse Data Processing Pipeline
========================================================
[STEP 1] Parsing .mat files...
✓ Parsed 15000 discharge records
✓ Parsed 600 EIS records

[STEP 2] Pairing discharge with EIS...
✓ Created 400 discharge-EIS pairs
✓ Validation stats: ...

[STEP 3] Battery-level train/val/test split...
✓ Train: 250 samples from 2 batteries
✓ Val:   75 samples from 1 battery
✓ Test:  75 samples from 1 battery

[STEP 4] Creating PyTorch DataLoaders...
✓ Train loader: 8 batches (batch_size=32)
✓ Val loader:   3 batches
✓ Test loader:  3 batches

[STEP 5] Inspecting sample...
✓ Sample batch:
  - discharge shape: (32, 5)
  - eis shape: (32, 3)
  - physics shape: (32, 4)
  - soh_label shape: (32,)
```

### Use in Training Code

```python
import torch
from src.data.dataset import create_dataloaders

# Load split data
train_df = pd.read_pickle('data/processed/train.pkl')
val_df = pd.read_pickle('data/processed/val.pkl')
test_df = pd.read_pickle('data/processed/test.pkl')

# Create DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    batch_size=32,
    num_workers=4
)

# Training loop
for batch in train_loader:
    discharge = batch['discharge']      # (32, 5)
    eis = batch['eis']                  # (32, 3)
    physics = batch['physics']          # (32, 4)
    soh_labels = batch['soh_label']     # (32,)
    
    # Forward pass through model
    model_output = model(discharge, eis, physics)
    loss = criterion(model_output['soh_pred'], soh_labels)
```

## Data Statistics

### Discharge Features
- **Voltage (V):** 2.7 - 4.2V (discharge to charge range)
  - Mean: ~3.8V
  - Degradation: min voltage drops → harder to reach full charge
  
- **Current (A):** ~2.0A (constant discharge rate)
  - Very stable, less informative alone
  
- **Temperature (°C):** 20-30°C
  - Ambient + self-heating during discharge
  
- **Capacity (Ahr):** 2.0 → 1.4 (nominal → EOL)
  - Direct measure of degradation
  - Our prediction target (as SoH %)

### EIS Features
- **Impedance (Ω):** 0.04 - 0.15Ω
  - Reflects resistance growth
  - More sensitive to internal changes than capacity alone
  
- **Re (Ω):** ~0.01 - 0.05Ω
  - Electrolyte resistance (relatively stable)
  
- **Rct (Ω):** ~0.03 - 0.15Ω
  - Charge transfer resistance (increases with degradation)

### Physics Features
- **Cycle Age:** 0 - 150 cycles
  - Normalized by typical lifetime
  
- **Capacity Fade:** 0 - 0.3 Ahr
  - Continuous variable
  - Highly correlated with SoH (our label)
  
- **Voltage Droop:** Negative values
  - Lower minimum voltage → worse condition
  
- **Impedance Rise:** Positive values
  - Higher impedance → worse condition

## Troubleshooting

### Issue: "No .mat files found"
- Ensure `5. BatteryDataSet` directory exists
- Check subdirectory structure
- Verify .mat files are not corrupted

### Issue: "Empty paired dataframe"
- May indicate timing mismatch between discharge and EIS
- Increase `max_cycle_gap` in `pair_discharge_eis()`
- Check data validity with `validate_pairs()`

### Issue: "Unequal split sizes"
- Expected with small battery count (~4 batteries)
- Some batteries have more cycles than others
- Stratification ensures SoH representation in each split

### Issue: NaN values in features
- Some batteries may lack complete EIS data
- Dataset cleaning handles this (skips incomplete records)
- Check `validation_stats` for missing data count

## Next Steps

1. **Review Data Quality:** Explore in `notebooks/` with EDA
2. **Train Model:** Use `process_data.py` → `src/train.py`
3. **Evaluate:** Run `src/evaluate.py` on test set
4. **Ablation Study:** Compare modalities with `src/ablation.py`
5. **Dashboard:** Visualize results with `app/dashboard.py`

## References

- **NASA PCoE Dataset:** https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-center/publications/
- **Data Description:** `5. BatteryDataSet/1. BatteryAgingARC-FY08Q4/README.txt`
- **Physics Model:** Electrochemical Impedance Spectroscopy (EIS) for SoH estimation
