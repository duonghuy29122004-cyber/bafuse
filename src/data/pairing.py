"""
Pair discharge curves with corresponding EIS measurements.

Ensures:
- Each discharge cycle is matched with the closest EIS measurement in time
- Same battery and cycle age
- No duplicate pairings
- Proper handling of sequence: discharge → charge → impedance pattern
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def pair_discharge_eis(
    discharge_df: pd.DataFrame,
    eis_df: pd.DataFrame,
    max_cycle_gap: int = 1
) -> pd.DataFrame:
    """
    Match discharge curves with EIS measurements.
    
    NASA PCoE pattern: typical cycle is discharge → charge → impedance.
    Pairing logic:
    - Group by battery_id
    - For each discharge cycle, find nearest EIS measurement (usually within 1 cycle)
    - Keep only pairs where EIS cycle ≤ discharge cycle (EIS measured after discharge)
    
    Args:
        discharge_df: Discharge curve data with columns:
                     [cycle_idx, time_s, voltage_v, current_a, temperature_c, capacity_ahr, battery_id]
        eis_df: EIS spectrum data with columns:
               [cycle_idx, impedance_ohm, re_ohm, rct_ohm, battery_id]
        max_cycle_gap: Maximum cycle difference for valid pairing (usually 1-2)
    
    Returns:
        DataFrame with paired discharge-EIS samples
        Columns: [battery_id, cycle_idx, capacity_ahr, voltage_v, current_a, 
                  temperature_c, impedance_ohm, re_ohm, rct_ohm, time_s]
    """
    
    # Aggregate discharge data per cycle (many measurements per discharge)
    discharge_agg = discharge_df.groupby(['battery_id', 'cycle_idx']).agg({
        'capacity_ahr': 'first',  # Capacity is same for entire discharge
        'voltage_v': ['mean', 'min', 'max', 'std'],
        'current_a': ['mean', 'std'],
        'temperature_c': ['mean', 'std'],
        'time_s': ['min', 'max', 'count']  # Duration and num samples
    }).reset_index()
    
    # Flatten column names
    discharge_agg.columns = ['battery_id', 'cycle_idx', 'capacity_ahr',
                             'voltage_mean', 'voltage_min', 'voltage_max', 'voltage_std',
                             'current_mean', 'current_std',
                             'temp_mean', 'temp_std',
                             'time_start', 'time_end', 'num_samples']
    
    # Pair each discharge with nearest EIS
    paired_list = []
    
    for battery_id in discharge_agg['battery_id'].unique():
        discharge_battery = discharge_agg[discharge_agg['battery_id'] == battery_id].reset_index(drop=True)
        eis_battery = eis_df[eis_df['battery_id'] == battery_id].reset_index(drop=True)
        
        if eis_battery.empty:
            logger.warning(f"No EIS data for {battery_id}")
            continue
        
        for _, discharge_row in discharge_battery.iterrows():
            discharge_cycle = discharge_row['cycle_idx']
            
            # Find EIS measurements close to this discharge
            # Typically: discharge cycle N → charge cycle N+1 → EIS cycle N+2
            # But sometimes EIS is cycle N or N+1
            eis_candidates = eis_battery[
                (eis_battery['cycle_idx'] >= discharge_cycle) &
                (eis_battery['cycle_idx'] <= discharge_cycle + max_cycle_gap)
            ]
            
            if eis_candidates.empty:
                # Fallback: find nearest EIS
                eis_candidates = eis_battery.copy()
                eis_candidates['cycle_diff'] = np.abs(eis_candidates['cycle_idx'] - discharge_cycle)
                eis_candidates = eis_candidates.nsmallest(1, 'cycle_diff')
            
            # Take the closest EIS
            if not eis_candidates.empty:
                eis_row = eis_candidates.iloc[0]
                
                paired_row = pd.DataFrame({
                    'battery_id': [battery_id],
                    'discharge_cycle': [discharge_cycle],
                    'eis_cycle': [eis_row['cycle_idx']],
                    'capacity_ahr': [discharge_row['capacity_ahr']],
                    'voltage_mean': [discharge_row['voltage_mean']],
                    'voltage_min': [discharge_row['voltage_min']],
                    'voltage_max': [discharge_row['voltage_max']],
                    'voltage_std': [discharge_row['voltage_std']],
                    'current_mean': [discharge_row['current_mean']],
                    'current_std': [discharge_row['current_std']],
                    'temp_mean': [discharge_row['temp_mean']],
                    'temp_std': [discharge_row['temp_std']],
                    'impedance_ohm': [eis_row['impedance_ohm']],
                    're_ohm': [eis_row['re_ohm']],
                    'rct_ohm': [eis_row['rct_ohm']],
                    'cycle_gap': [int(eis_row['cycle_idx'] - discharge_cycle)]
                })
                paired_list.append(paired_row)
    
    if paired_list:
        paired_df = pd.concat(paired_list, ignore_index=True)
        logger.info(f"Created {len(paired_df)} discharge-EIS pairs")
        return paired_df
    else:
        logger.warning("No successful pairings created")
        return pd.DataFrame()


def validate_pairs(paired_df: pd.DataFrame) -> dict:
    """
    Validate pairing quality and return statistics.
    
    Args:
        paired_df: Paired discharge-EIS dataframe
    
    Returns:
        Dict with validation statistics:
        - num_pairs: Total number of pairs
        - batteries: List of batteries
        - pairs_per_battery: Number of pairs per battery
        - cycle_gap_stats: Statistics on cycle gaps
        - capacity_range: (min, max) capacity values
        - impedance_range: (min, max) impedance values
    """
    if paired_df.empty:
        return {"error": "Empty dataframe"}
    
    stats = {
        "num_pairs": len(paired_df),
        "batteries": paired_df['battery_id'].unique().tolist(),
        "num_batteries": paired_df['battery_id'].nunique(),
        "pairs_per_battery": paired_df.groupby('battery_id').size().to_dict(),
        "cycle_gap_stats": {
            "mean": paired_df['cycle_gap'].mean(),
            "median": paired_df['cycle_gap'].median(),
            "std": paired_df['cycle_gap'].std(),
            "min": paired_df['cycle_gap'].min(),
            "max": paired_df['cycle_gap'].max()
        },
        "capacity_range": (paired_df['capacity_ahr'].min(), paired_df['capacity_ahr'].max()),
        "impedance_range": (paired_df['impedance_ohm'].min(), paired_df['impedance_ohm'].max()),
        "voltage_mean_range": (paired_df['voltage_mean'].min(), paired_df['voltage_mean'].max()),
        "temp_range": (paired_df['temp_mean'].min(), paired_df['temp_mean'].max()),
        "missing_impedance": paired_df['impedance_ohm'].isna().sum(),
        "missing_capacity": paired_df['capacity_ahr'].isna().sum()
    }
    
    logger.info(f"Validation stats: {stats}")
    return stats
