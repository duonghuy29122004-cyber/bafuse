# BaFuse: Battery Fusion Estimation

**Multimodal Fusion of Discharge-Curve, Impedance, and Physics-Informed Simulation Signals for EV Battery Health Estimation**

## Overview

BaFuse combines three complementary battery health signals to robustly estimate State-of-Health (SoH):

1. **Discharge Curves** - Reflect present capacity behavior and voltage dynamics
2. **Electrochemical Impedance Spectroscopy (EIS)** - Reveal internal resistance changes
3. **Physics-Informed Simulation** - Provide degradation trajectory priors based on electrochemical models

### Why Multimodal?

- **Discharge curves alone** miss internal resistance degradation
- **EIS alone** lacks capacity fade context
- **Physics simulation alone** is only a prior, not validated on actual data
- **Fusion** combines all three for more robust SoH estimation

## Project Structure

```
bafuse/
├── 5. BatteryDataSet/              # NASA PCoE battery dataset
│   ├── 1. BatteryAgingARC-FY08Q4/  # Test campaign 1 (.mat files)
│   │   ├── B0005.mat
│   │   ├── B0006.mat
│   │   ├── B0007.mat
│   │   ├── B0018.mat
│   │   └── README.txt
│   ├── 2. BatteryAgingARC_25_26_27_28_P1/
│   ├── 3. BatteryAgingARC_25-44/
│   ├── ... (more campaigns)
├── data/
│   ├── raw/                        # Symlink or copies of .mat files (optional)
│   └── processed/                  # Parsed discharge.pkl, impedance.pkl, paired.pkl
├── src/
│   ├── data/
│   │   ├── parse_mat.py            # Load .mat → DataFrame
│   │   ├── pairing.py              # Match discharge with EIS
│   │   ├── split.py                # Battery-level train/val/test
│   │   └── dataset.py              # PyTorch Dataset/DataLoader
│   ├── models/
│   │   ├── encoders.py             # Modality-specific encoders
│   │   ├── fusion.py               # Cross-attention fusion
│   │   └── bafuse.py               # Full unified model
│   ├── losses.py                   # SoH + physics-informed losses
│   ├── train.py                    # Training loop
│   ├── evaluate.py                 # Test set evaluation
│   └── ablation.py                 # Ablation study
├── notebooks/                      # EDA and prototyping
├── app/
│   └── dashboard.py                # Streamlit visualization
├── configs/
│   └── config.yaml                 # Hyperparameters
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone and install dependencies
pip install -r requirements.txt
```

## Data Preparation

### 1. NASA PCoE Dataset Structure

The dataset is located in `5. BatteryDataSet/` with subdirectories for different aging test campaigns:

```
5. BatteryDataSet/
├── 1. BatteryAgingARC-FY08Q4/
│   ├── B0005.mat (Battery #5: charge/discharge/impedance cycles)
│   ├── B0006.mat (Battery #6)
│   ├── B0007.mat (Battery #7)
│   ├── B0018.mat (Battery #18)
│   └── README.txt (Data structure description)
├── 2. BatteryAgingARC_25_26_27_28_P1/
├── ... (additional test campaigns)
```

Each `.mat` file contains:
- **Discharge cycles**: voltage, current, temperature, capacity over time
- **Charge cycles**: similar measurements during charging
- **Impedance (EIS) cycles**: electrochemical impedance spectra at various frequencies

### 2. Parse and Process Data

Run the complete pipeline to parse, pair, and split data:

```python
from src.data.parse_mat import parse_all_mat_files
from src.data.pairing import pair_discharge_eis, validate_pairs
from src.data.split import split_by_battery, get_split_statistics
from src.data.dataset import create_dataloaders
import pandas as pd

# Step 1: Parse .mat files
discharge_df, eis_df = parse_all_mat_files('5. BatteryDataSet')

# Step 2: Pair discharge curves with EIS measurements
paired_df = pair_discharge_eis(discharge_df, eis_df)
validation_stats = validate_pairs(paired_df)
print(f"Created {len(paired_df)} discharge-EIS pairs")

# Step 3: Battery-level train/val/test split (no leakage)
train_df, val_df, test_df = split_by_battery(
    paired_df,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    stratify_by_soh=True
)

split_stats = get_split_statistics(train_df, val_df, test_df)

# Step 4: Create PyTorch DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    discharge_data_df=discharge_df,  # optional, for time-series
    batch_size=32
)

# Save processed data
train_df.to_pickle('data/processed/train.pkl')
val_df.to_pickle('data/processed/val.pkl')
test_df.to_pickle('data/processed/test.pkl')
```

### 3. Data Features

Each paired sample includes:

**Discharge Features:**
- Voltage (mean, min, max, std) - reflects capacity fade
- Current (mean, std)
- Temperature (mean, std)
- Number of time samples in discharge

**EIS Features:**
- Impedance (Ω) - reflects resistance growth
- Re (electrolyte resistance)
- Rct (charge transfer resistance)

**Physics-Informed Features:**
- Cycle age / max cycles
- Capacity fade (2Ahr nominal → current capacity)
- Voltage droop (lower minimum voltage with degradation)
- Impedance rise (increasing resistance)

**Label:**
- SoH (State-of-Health) = (current_capacity / 2.0Ahr) × 100%

## Training

Configure hyperparameters in `configs/config.yaml`, then:

```bash
cd src
python train.py --config ../configs/config.yaml
```

## Evaluation

```bash
cd src
python evaluate.py --checkpoint checkpoints/best_model.pth --test_data ../data/processed/test.pkl
```

## Ablation Study

Compare single modalities vs. fusion:

```bash
cd src
python ablation.py --config ../configs/config.yaml
```

## Dashboard

Visualize results interactively:

```bash
streamlit run app/dashboard.py
```

## Model Architecture

### Encoders
- **DischargeEncoder**: LSTM on time-series voltage/current/capacity
- **EISEncoder**: CNN on impedance spectrum
- **PhysicsEncoder**: MLP on degradation features

### Fusion
- **CrossAttentionFusion**: Each modality attends to others
- **WeightedFusion**: Learnable scalar weights per modality
- **ConcatFusion**: Simple concatenation baseline

### Head
- Fused representations → MLP → SoH prediction (0-100%)

## Key Features

✅ **Battery-level splits** - Prevent train/test leakage  
✅ **Physics-informed loss** - Enforce smooth, bounded degradation  
✅ **Cross-attention fusion** - Interpretable modality contributions  
✅ **Ablation studies** - Quantify synergy between modalities  
✅ **Uncertainty quantification** - MC Dropout for confidence intervals  
✅ **Interactive dashboard** - Explore predictions and modality contributions  

## Results

Expected improvements over unimodal baselines:
- ~10-15% lower MAE than discharge-only
- ~5-10% lower RMSE than EIS-only
- Physics prior enables better generalization to unseen battery chemistry

## References

- NASA PCoE Dataset: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-center/publications/
- Cross-attention for multimodal fusion (inspired by Vision Transformers)
- Physics-informed machine learning for battery health

## Authors

Capstone Team

## License

MIT
