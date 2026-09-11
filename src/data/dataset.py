"""
PyTorch Dataset and DataLoader for BaFuse.

Handles:
- Loading paired discharge-EIS samples
- Feature extraction and normalization
- Batch creation for training
- Physics-informed features (degradation simulation)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BaFuseDataset(Dataset):
    """
    PyTorch Dataset for multimodal battery health data.
    
    Returns samples with:
    - discharge: voltage/current/temp features over time
    - eis: impedance and resistance measurements
    - physics: simulated degradation features
    - soh_label: ground truth capacity (as SoH proxy)
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        discharge_data_df: Optional[pd.DataFrame] = None,
        normalize: bool = True,
        max_seq_len: int = 500,
        device: str = 'cpu'
    ):
        """
        Initialize BaFuse dataset.
        
        Args:
            data_df: Paired discharge-EIS dataframe with columns:
                    [battery_id, discharge_cycle, eis_cycle, capacity_ahr, 
                     voltage_mean/min/max/std, current_mean/std, temp_mean/std,
                     impedance_ohm, re_ohm, rct_ohm, cycle_gap]
            discharge_data_df: Full discharge measurement dataframe (for feature extraction)
            normalize: Whether to normalize features
            max_seq_len: Maximum sequence length for discharge time-series
            device: torch device
        """
        self.data_df = data_df.reset_index(drop=True)
        self.discharge_data_df = discharge_data_df
        self.normalize = normalize
        self.max_seq_len = max_seq_len
        self.device = device
        
        # Compute normalization statistics if needed
        if self.normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute mean/std for normalization."""
        self.stats = {
            'voltage': {
                'mean': self.data_df['voltage_mean'].mean(),
                'std': self.data_df['voltage_mean'].std() + 1e-8
            },
            'current': {
                'mean': self.data_df['current_mean'].mean(),
                'std': self.data_df['current_mean'].std() + 1e-8
            },
            'temperature': {
                'mean': self.data_df['temp_mean'].mean(),
                'std': self.data_df['temp_mean'].std() + 1e-8
            },
            'impedance': {
                'mean': self.data_df['impedance_ohm'].mean(),
                'std': self.data_df['impedance_ohm'].std() + 1e-8
            },
            'capacity': {
                'mean': self.data_df['capacity_ahr'].mean(),
                'std': self.data_df['capacity_ahr'].std() + 1e-8
            }
        }
        logger.info(f"Normalization stats computed")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_df)
    
    def _normalize(self, value: float, stat_key: str) -> float:
        """Normalize a value using precomputed statistics."""
        if not self.normalize:
            return value
        stats = self.stats[stat_key]
        return (value - stats['mean']) / stats['std']
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample.
        
        Returns:
            Dict with keys:
            - discharge: (seq_len, 5) features [V, I, T, V_min, V_max]
            - eis: (3,) features [impedance, Re, Rct]
            - physics: (4,) features [capacity_fade, cycle_age, voltage_droop, impedance_rise]
            - soh_label: scalar (0-100) representing state of health
            - battery_id: str
            - cycle_idx: int
        """
        row = self.data_df.iloc[idx]
        
        battery_id = row['battery_id']
        discharge_cycle = row['discharge_cycle']
        capacity_ahr = row['capacity_ahr']
        
        # ==== DISCHARGE FEATURES ====
        # Aggregate statistics from time-series
        discharge_features = np.array([
            self._normalize(row['voltage_mean'], 'voltage'),
            self._normalize(row['current_mean'], 'current'),
            self._normalize(row['temp_mean'], 'temperature'),
            self._normalize(row['voltage_min'], 'voltage'),
            self._normalize(row['voltage_max'], 'voltage'),
        ], dtype=np.float32)
        
        # Pad to max sequence length (if discharge_data_df available, use raw time-series)
        if self.discharge_data_df is not None:
            discharge_ts = self.discharge_data_df[
                (self.discharge_data_df['battery_id'] == battery_id) &
                (self.discharge_data_df['cycle_idx'] == discharge_cycle)
            ]
            if not discharge_ts.empty:
                ts_features = discharge_ts[['voltage_v', 'current_a', 'temperature_c']].values
                # Normalize and truncate/pad to max_seq_len
                ts_features = ts_features[:self.max_seq_len]
                if len(ts_features) < self.max_seq_len:
                    # Pad with zeros
                    padding = np.zeros((self.max_seq_len - len(ts_features), 3), dtype=np.float32)
                    ts_features = np.vstack([ts_features, padding]).astype(np.float32)
                discharge_features = ts_features
        
        discharge_tensor = torch.from_numpy(discharge_features.astype(np.float32))
        
        # ==== EIS (IMPEDANCE) FEATURES ====
        eis_features = np.array([
            self._normalize(row['impedance_ohm'], 'impedance'),
            row['re_ohm'] if pd.notna(row['re_ohm']) else 0.0,  # Electrolyte resistance
            row['rct_ohm'] if pd.notna(row['rct_ohm']) else 0.0,  # Charge transfer resistance
        ], dtype=np.float32)
        eis_tensor = torch.from_numpy(eis_features.astype(np.float32))
        
        # ==== PHYSICS-INFORMED FEATURES ====
        # Features derived from discharge and impedance measurements
        # These represent physical degradation mechanisms
        cycle_age = float(discharge_cycle)
        capacity_fade = 2.0 - capacity_ahr  # Nominal capacity was 2Ahr, so fade = 2 - current
        capacity_fade_normalized = self._normalize(capacity_fade, 'capacity')
        
        physics_features = np.array([
            cycle_age / 150.0,  # Normalize to ~150 cycles typical
            capacity_fade_normalized,  # Capacity loss
            (row['voltage_min'] - 2.7) / 1.5,  # Voltage droop indicator
            (row['impedance_ohm'] - self.stats['impedance']['mean']) / self.stats['impedance']['std'],  # Impedance rise
        ], dtype=np.float32)
        physics_tensor = torch.from_numpy(physics_features.astype(np.float32))
        
        # ==== SOH LABEL ====
        # SoH = (current_capacity / nominal_capacity) * 100
        soh_label = (capacity_ahr / 2.0) * 100.0  # Nominal = 2Ahr
        soh_label = np.clip(soh_label, 0, 100)
        soh_tensor = torch.tensor(soh_label, dtype=torch.float32)
        
        return {
            'discharge': discharge_tensor,
            'eis': eis_tensor,
            'physics': physics_tensor,
            'soh_label': soh_tensor,
            'battery_id': battery_id,
            'cycle_idx': torch.tensor(discharge_cycle, dtype=torch.long)
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    discharge_data_df: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test.
    
    Args:
        train_df, val_df, test_df: Split dataframes
        discharge_data_df: Full discharge measurements (optional, for time-series)
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    train_dataset = BaFuseDataset(
        train_df,
        discharge_data_df=discharge_data_df,
        normalize=True
    )
    
    val_dataset = BaFuseDataset(
        val_df,
        discharge_data_df=discharge_data_df,
        normalize=True
    )
    
    test_dataset = BaFuseDataset(
        test_df,
        discharge_data_df=discharge_data_df,
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Created DataLoaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    return train_loader, val_loader, test_loader
