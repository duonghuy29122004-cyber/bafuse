"""
Battery-level train/validation/test split.

Ensures:
- No data leakage: each battery belongs to exactly one split
- Stratified by SoH range (capacity-based)
- Reproducible with random seed
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def split_by_battery(
    paired_df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
    stratify_by_soh: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data at battery level to prevent leakage.
    
    Algorithm:
    1. Get unique batteries
    2. Compute SoH proxy (capacity) per battery (mean across cycles)
    3. Stratify batteries into SoH ranges
    4. Randomly assign batteries to train/val/test
    5. Filter paired data by assigned batteries
    
    Args:
        paired_df: Paired discharge-EIS dataset
        train_ratio: Fraction for training (default 0.6)
        val_ratio: Fraction for validation (default 0.2)
        test_ratio: Fraction for testing (default 0.2)
        random_state: Random seed for reproducibility
        stratify_by_soh: Stratify by SoH (capacity) range
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Get unique batteries
    batteries = paired_df['battery_id'].unique()
    logger.info(f"Splitting {len(batteries)} batteries")
    
    # Compute SoH proxy (mean capacity per battery)
    battery_soh = paired_df.groupby('battery_id')['capacity_ahr'].mean().reset_index()
    battery_soh.columns = ['battery_id', 'mean_capacity']
    
    if stratify_by_soh:
        # Compute capacity deciles for stratification
        # Guard: sklearn requires each stratum to have ≥ 2 members.
        # With very few batteries this can fail → fall back to no stratification.
        try:
            battery_soh['capacity_bin'] = pd.qcut(
                battery_soh['mean_capacity'],
                q=3,  # 3 bins: low, medium, high capacity loss
                labels=['high_loss', 'medium_loss', 'low_loss'],
                duplicates='drop'
            )
            strata = battery_soh['capacity_bin'].values
            # Verify every stratum has ≥ 2 members (sklearn requirement)
            bin_counts = pd.Series(strata).value_counts()
            if bin_counts.min() < 2:
                raise ValueError(
                    f"Stratum too small: {bin_counts.to_dict()} — disabling stratification"
                )
            logger.info(f"Stratified by capacity ranges: {bin_counts.to_dict()}")
        except Exception as e:
            logger.warning(f"Stratification disabled ({e}); using random split instead.")
            strata = None
    else:
        strata = None
    
    # Split at battery level (not sample level)
    # First: train vs (val+test)
    train_batteries, temp_batteries = train_test_split(
        battery_soh['battery_id'].values,
        train_size=train_ratio,
        random_state=random_state,
        stratify=strata if stratify_by_soh else None
    )

    # P3-#9 FIX: also stratify the val/test split.
    val_ratio_of_temp = val_ratio / (val_ratio + test_ratio)
    if stratify_by_soh and len(temp_batteries) >= 4:
        temp_soh  = battery_soh[battery_soh['battery_id'].isin(temp_batteries)].copy()
        try:
            temp_soh['capacity_bin'] = pd.qcut(
                temp_soh['mean_capacity'],
                q=min(3, len(temp_soh)),
                labels=False,
                duplicates='drop',
            )
            temp_strata = temp_soh.set_index('battery_id').loc[temp_batteries, 'capacity_bin'].values
            # Require ≥ 2 per stratum
            if pd.Series(temp_strata).value_counts().min() < 2:
                raise ValueError("temp stratum too small")
        except Exception:
            temp_strata = None
    else:
        temp_strata = None

    val_batteries, test_batteries = train_test_split(
        temp_batteries,
        train_size=val_ratio_of_temp,
        random_state=random_state,
        stratify=temp_strata,
    )
    
    # Filter data by battery assignment
    train_df = paired_df[paired_df['battery_id'].isin(train_batteries)].reset_index(drop=True)
    val_df = paired_df[paired_df['battery_id'].isin(val_batteries)].reset_index(drop=True)
    test_df = paired_df[paired_df['battery_id'].isin(test_batteries)].reset_index(drop=True)
    
    logger.info(f"Train: {len(train_df)} samples from {len(train_batteries)} batteries")
    logger.info(f"Val:   {len(val_df)} samples from {len(val_batteries)} batteries")
    logger.info(f"Test:  {len(test_df)} samples from {len(test_batteries)} batteries")
    
    return train_df, val_df, test_df


def get_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> dict:
    """
    Return statistics about the split.
    
    Args:
        train_df, val_df, test_df: Split dataframes
    
    Returns:
        Dict with statistics:
        - num_samples, num_batteries per split
        - capacity/impedance ranges
        - cycle coverage
    """
    
    def compute_split_stats(df, split_name):
        return {
            f"{split_name}_num_samples": len(df),
            f"{split_name}_num_batteries": df['battery_id'].nunique(),
            f"{split_name}_capacity_range": (df['capacity_ahr'].min(), df['capacity_ahr'].max()),
            f"{split_name}_impedance_range": (df['impedance_ohm'].min(), df['impedance_ohm'].max()),
            f"{split_name}_voltage_range": (df['voltage_mean'].min(), df['voltage_mean'].max()),
            f"{split_name}_cycles_per_battery": df.groupby('battery_id').size().mean(),
        }
    
    stats = {}
    stats.update(compute_split_stats(train_df, "train"))
    stats.update(compute_split_stats(val_df, "val"))
    stats.update(compute_split_stats(test_df, "test"))
    
    # Overall statistics
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    stats['total_samples'] = len(all_df)
    stats['total_batteries'] = all_df['battery_id'].nunique()
    stats['train_ratio'] = len(train_df) / len(all_df)
    stats['val_ratio'] = len(val_df) / len(all_df)
    stats['test_ratio'] = len(test_df) / len(all_df)
    
    logger.info(f"Split statistics: {stats}")
    return stats
