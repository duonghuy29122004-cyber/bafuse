"""
Parse NASA PCoE .mat files into DataFrame format.

Handles:
- Loading .mat files using scipy.io.loadmat
- Extracting discharge curve and EIS measurements from cycle structure
- Converting to standardized DataFrame structure

Dataset structure:
- Each .mat file contains a 'cycle' array
- Each cycle has type: 'charge', 'discharge', or 'impedance'
- Discharge cycles include: voltage, current, temperature, capacity, time
- Impedance cycles include: real/imaginary impedance vs frequency
"""

import scipy.io as sio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_mat_file(filepath: str) -> dict:
    """
    Load a MATLAB .mat file.
    
    Args:
        filepath: Path to .mat file
    
    Returns:
        Dictionary containing .mat file contents
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid .mat file
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        mat_data = sio.loadmat(filepath, squeeze_me=True)
        logger.debug(f"Loaded {filepath}")
        return mat_data
    except Exception as e:
        raise ValueError(f"Failed to load .mat file {filepath}: {e}")


def _extract_cycle_data(cycle: dict, field: str) -> np.ndarray:
    """
    Safely extract nested field from cycle structure.
    
    Args:
        cycle: Single cycle from 'cycle' array
        field: Field name to extract (e.g., 'Voltage_measured')
    
    Returns:
        Array or scalar value, or None if field doesn't exist
    """
    try:
        if 'data' in cycle:
            data = cycle['data']
            if isinstance(data, dict) and field in data:
                return data[field]
    except (KeyError, TypeError):
        pass
    return None


def parse_discharge_curve(mat_data: dict, battery_id: str) -> pd.DataFrame:
    """
    Extract discharge curve data from .mat structure.
    
    Args:
        mat_data: Dictionary from loadmat (contains 'cycle' array)
        battery_id: Battery identifier (e.g., 'B0005')
    
    Returns:
        DataFrame with columns: 
        [cycle_idx, time_s, voltage_v, current_a, temperature_c, capacity_ahr, battery_id]
    """
    if 'cycle' not in mat_data:
        logger.warning(f"No 'cycle' key in {battery_id}")
        return pd.DataFrame()
    
    cycles = mat_data['cycle']
    
    # Handle case where there's only one cycle (not an array)
    if not isinstance(cycles, np.ndarray):
        cycles = np.array([cycles])
    
    discharge_data = []
    
    for cycle_idx, cycle in enumerate(cycles):
        # Check if this is a discharge cycle
        cycle_type = cycle.get('type', '')
        if isinstance(cycle_type, np.ndarray):
            cycle_type = cycle_type.item() if cycle_type.size == 1 else cycle_type
        
        if cycle_type != 'discharge':
            continue
        
        # Extract discharge measurements
        voltage = _extract_cycle_data(cycle, 'Voltage_measured')
        current = _extract_cycle_data(cycle, 'Current_measured')
        temperature = _extract_cycle_data(cycle, 'Temperature_measured')
        time = _extract_cycle_data(cycle, 'Time')
        capacity = _extract_cycle_data(cycle, 'Capacity')
        
        # Ensure arrays have consistent shape
        if voltage is None or time is None:
            continue
        
        voltage = np.atleast_1d(voltage)
        current = np.atleast_1d(current) if current is not None else np.full_like(voltage, np.nan)
        temperature = np.atleast_1d(temperature) if temperature is not None else np.full_like(voltage, np.nan)
        time = np.atleast_1d(time)
        capacity_val = capacity if capacity is not None else np.nan
        
        # Ensure all arrays same length
        min_len = min(len(voltage), len(current), len(temperature), len(time))
        voltage = voltage[:min_len]
        current = current[:min_len]
        temperature = temperature[:min_len]
        time = time[:min_len]
        
        # Create dataframe for this discharge cycle
        cycle_df = pd.DataFrame({
            'cycle_idx': cycle_idx,
            'time_s': time,
            'voltage_v': voltage,
            'current_a': current,
            'temperature_c': temperature,
            'capacity_ahr': capacity_val,
            'battery_id': battery_id
        })
        discharge_data.append(cycle_df)
    
    if discharge_data:
        return pd.concat(discharge_data, ignore_index=True)
    else:
        return pd.DataFrame()


def parse_eis_spectrum(mat_data: dict, battery_id: str) -> pd.DataFrame:
    """
    Extract EIS (electrochemical impedance spectroscopy) data.
    
    NASA PCoE provides rectified (calibrated) impedance as main feature.
    Also extracts Re (electrolyte) and Rct (charge transfer) resistances.
    
    Args:
        mat_data: Dictionary from loadmat (contains 'cycle' array)
        battery_id: Battery identifier (e.g., 'B0005')
    
    Returns:
        DataFrame with columns: 
        [cycle_idx, impedance_ohm, re_ohm, rct_ohm, battery_id]
    """
    if 'cycle' not in mat_data:
        logger.warning(f"No 'cycle' key in {battery_id}")
        return pd.DataFrame()
    
    cycles = mat_data['cycle']
    
    # Handle case where there's only one cycle
    if not isinstance(cycles, np.ndarray):
        cycles = np.array([cycles])
    
    eis_data = []
    
    for cycle_idx, cycle in enumerate(cycles):
        # Check if this is an impedance cycle
        cycle_type = cycle.get('type', '')
        if isinstance(cycle_type, np.ndarray):
            cycle_type = cycle_type.item() if cycle_type.size == 1 else cycle_type
        
        if cycle_type != 'impedance':
            continue
        
        # Extract impedance measurements
        # Use Rectified_impedance (calibrated and smoothed)
        impedance = _extract_cycle_data(cycle, 'Rectified_impedance')
        re = _extract_cycle_data(cycle, 'Re')
        rct = _extract_cycle_data(cycle, 'Rct')
        
        if impedance is None:
            # Fallback to raw Battery_impedance if rectified not available
            impedance = _extract_cycle_data(cycle, 'Battery_impedance')
        
        if impedance is None:
            continue
        
        impedance = np.atleast_1d(impedance)
        
        # For scalar impedance values (typically one value per cycle)
        if impedance.size == 1:
            impedance_ohm = float(impedance)
            re_ohm = float(re) if re is not None else np.nan
            rct_ohm = float(rct) if rct is not None else np.nan
        else:
            # If impedance is a spectrum, take median
            impedance_ohm = float(np.nanmedian(impedance))
            re_ohm = float(re) if re is not None else np.nan
            rct_ohm = float(rct) if rct is not None else np.nan
        
        eis_row = pd.DataFrame({
            'cycle_idx': [cycle_idx],
            'impedance_ohm': [impedance_ohm],
            're_ohm': [re_ohm],
            'rct_ohm': [rct_ohm],
            'battery_id': [battery_id]
        })
        eis_data.append(eis_row)
    
    if eis_data:
        return pd.concat(eis_data, ignore_index=True)
    else:
        return pd.DataFrame()


def parse_all_mat_files(raw_data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all .mat files in raw data directory and subdirectories.
    
    Assumes structure like:
    raw_data_dir/
      ├── 1. BatteryAgingARC-FY08Q4/
      │   ├── B0005.mat
      │   ├── B0006.mat
      │   └── ...
      └── ...
    
    Args:
        raw_data_dir: Path to directory containing .mat files or subdirs with .mat files
    
    Returns:
        Tuple of (discharge_df, eis_df) concatenated from all batteries
    """
    raw_path = Path(raw_data_dir)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")
    
    # Find all .mat files (including in subdirectories)
    mat_files = list(raw_path.rglob('*.mat'))
    
    if not mat_files:
        logger.warning(f"No .mat files found in {raw_path}")
        return pd.DataFrame(), pd.DataFrame()
    
    logger.info(f"Found {len(mat_files)} .mat files")
    
    all_discharge = []
    all_eis = []
    
    for mat_file in sorted(mat_files):
        try:
            battery_id = mat_file.stem  # e.g., 'B0005'
            logger.info(f"Parsing {battery_id} from {mat_file}")
            
            mat_data = load_mat_file(str(mat_file))
            
            discharge_df = parse_discharge_curve(mat_data, battery_id)
            if not discharge_df.empty:
                all_discharge.append(discharge_df)
            
            eis_df = parse_eis_spectrum(mat_data, battery_id)
            if not eis_df.empty:
                all_eis.append(eis_df)
        
        except Exception as e:
            logger.error(f"Failed to parse {mat_file}: {e}")
            continue
    
    discharge_combined = pd.concat(all_discharge, ignore_index=True) if all_discharge else pd.DataFrame()
    eis_combined = pd.concat(all_eis, ignore_index=True) if all_eis else pd.DataFrame()
    
    logger.info(f"Parsed {len(discharge_combined)} discharge records and {len(eis_combined)} EIS records")
    
    return discharge_combined, eis_combined
