"""
Parse NASA PCoE .mat files into DataFrame format.

Handles:
- Loading .mat files using scipy.io.loadmat
- Extracting discharge curve and EIS measurements from cycle structure
- Converting to standardized DataFrame structure

Fixes applied:
- BUG 1: Impedance computed from |Z| magnitude, Re/Rct split from real/imag,
         physical outlier filter (0–1000 Ω) + percentile clip before median.
- BUG 4: Deduplicate by battery_id so files parsed from multiple subdirectories
         are only processed once.
"""

import warnings
import scipy.io as sio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Physical plausibility bounds for EIS (Ω)
_IMP_MIN_OHM = 0.0
_IMP_MAX_OHM = 1000.0


def load_mat_file(filepath: str) -> dict:
    """Load MATLAB .mat file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        mat_data = sio.loadmat(str(filepath), squeeze_me=False)
        logger.debug(f"Loaded {filepath}")
        return mat_data
    except Exception as e:
        raise ValueError(f"Failed to load .mat file {filepath}: {e}")


def _extract_time_series(value, flatten=True):
    """Extract time-series array from MATLAB value."""
    if value is None:
        return np.array([])
    if isinstance(value, np.void):
        return np.array([])
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.array([])
        return value.flatten() if flatten else value
    return np.atleast_1d(value)


def _safe_impedance_scalar(arr: np.ndarray) -> Tuple[float, float, float]:
    """
    Derive (|Z|_median, Re_median, Rct_proxy) from a raw impedance array
    that may contain complex numbers.

    Steps:
    1. Work with complex-aware arrays — no implicit cast to real.
    2. Compute per-element magnitude |Z| = sqrt(Re² + Im²).
    3. Apply hard physical bounds (0–1000 Ω) on magnitude.
    4. Apply soft percentile clip (1st–99th) to remove remaining outliers.
    5. Return median magnitude, median Re part, and median |Im| part as Rct proxy.

    Returns:
        (imp_median, re_median, rct_proxy)  — all float, NaN if no valid data.
    """
    if arr is None or len(arr) == 0:
        return np.nan, np.nan, np.nan

    arr = np.asarray(arr, dtype=complex).flatten()

    # --- magnitude, real, imaginary ---
    mag  = np.abs(arr)               # |Z|
    real = arr.real                  # Re(Z) — electrolyte resistance proxy
    imag = np.abs(arr.imag)          # |Im(Z)| — Rct proxy (positive convention)

    # --- hard physical filter on magnitude ---
    valid = (mag >= _IMP_MIN_OHM) & (mag <= _IMP_MAX_OHM)
    if valid.sum() == 0:
        return np.nan, np.nan, np.nan

    mag  = mag[valid]
    real = real[valid]
    imag = imag[valid]

    # --- also require Re >= 0 (electrolyte resistance is non-negative) ---
    # Note: some NASA PCoE frequencies yield slightly negative Re due to inductive
    # effects at high frequency — we clip to 0 rather than dropping those points.
    real = np.clip(real, 0.0, None)

    # --- soft percentile clip on magnitude (remove extreme tails) ---
    p1, p99 = np.nanpercentile(mag, [1, 99])
    in_range = (mag >= p1) & (mag <= p99)
    if in_range.sum() > 0:
        mag  = mag[in_range]
        real = real[in_range]
        imag = imag[in_range]

    imp_med = float(np.nanmedian(mag))
    re_med  = float(np.nanmedian(real))
    rct_med = float(np.nanmedian(imag))

    return imp_med, re_med, rct_med


def parse_discharge_curve(mat_data: dict, battery_id: str) -> pd.DataFrame:
    """Extract discharge cycle time-series data."""
    if battery_id not in mat_data:
        logger.debug(f"Battery {battery_id} not in mat_data")
        return pd.DataFrame()

    battery_struct = mat_data[battery_id]
    if not isinstance(battery_struct, np.ndarray) or battery_struct.size == 0:
        return pd.DataFrame()
    if battery_struct.dtype.names is None or 'cycle' not in battery_struct.dtype.names:
        return pd.DataFrame()

    cycles = battery_struct['cycle'][0, 0]
    if not isinstance(cycles, np.ndarray):
        cycles = np.array([cycles])

    discharge_records = []

    for cycle_idx in range(cycles.size):
        try:
            cycle = cycles.flat[cycle_idx]
            if not hasattr(cycle, 'dtype') or cycle.dtype.names is None:
                continue

            try:
                cycle_type = cycle['type']
                if isinstance(cycle_type, np.ndarray):
                    cycle_type = cycle_type.flat[0]
                cycle_type = str(cycle_type).strip()
            except (IndexError, KeyError):
                continue

            if cycle_type != 'discharge':
                continue

            try:
                data = cycle['data']
            except (IndexError, KeyError):
                continue

            if isinstance(data, np.ndarray) and data.size > 0:
                data = data.flat[0]

            if not isinstance(data, np.void) and not isinstance(data, np.ndarray):
                continue

            try:
                voltage     = _extract_time_series(data['Voltage_measured'])
                current     = _extract_time_series(data['Current_measured'])
                temperature = _extract_time_series(data['Temperature_measured'])
                time_data   = _extract_time_series(data['Time'])
            except (IndexError, ValueError, TypeError, KeyError):
                logger.debug(f"{battery_id} cycle {cycle_idx}: Could not extract fields")
                continue

            # Try to extract capacity (Ah)
            capacity_ts = np.array([])
            for cap_field in ['Capacity', 'capacity']:
                try:
                    cap_raw = data[cap_field]
                    capacity_ts = _extract_time_series(cap_raw)
                    if len(capacity_ts) > 0:
                        break
                except (IndexError, ValueError, TypeError, KeyError):
                    continue

            min_len = min(len(voltage), len(current), len(temperature), len(time_data))
            if min_len == 0:
                continue

            voltage     = voltage[:min_len]
            current     = current[:min_len]
            temperature = temperature[:min_len]
            time_data   = time_data[:min_len]

            # Cycle capacity
            if len(capacity_ts) >= min_len:
                capacity_ts   = capacity_ts[:min_len]
                cycle_capacity = float(np.nanmax(np.abs(capacity_ts)))
            else:
                dt = np.diff(time_data, prepend=time_data[0])
                charge = np.abs(current) * dt / 3600.0
                cycle_capacity = float(np.nansum(charge))

            for i in range(min_len):
                discharge_records.append({
                    'battery_id':    battery_id,
                    'cycle_idx':     cycle_idx,
                    'voltage_v':     float(voltage[i]),
                    'current_a':     float(current[i]),
                    'temperature_c': float(temperature[i]),
                    'time_s':        float(time_data[i]),
                    'capacity_ahr':  cycle_capacity,
                })

        except Exception as e:
            logger.debug(f"{battery_id} cycle {cycle_idx}: {e}")
            continue

    df = pd.DataFrame(discharge_records)
    num_cycles = df['cycle_idx'].nunique() if not df.empty else 0
    logger.info(f"{battery_id}: Parsed {len(df)} discharge records from {num_cycles} cycles")
    return df


def parse_eis_spectrum(mat_data: dict, battery_id: str) -> pd.DataFrame:
    """
    Extract impedance measurement data.

    BUG 1 fix: use _safe_impedance_scalar() — magnitude + outlier filter + Re/Rct split.
    """
    if battery_id not in mat_data:
        return pd.DataFrame()

    battery_struct = mat_data[battery_id]
    if not isinstance(battery_struct, np.ndarray) or battery_struct.size == 0:
        return pd.DataFrame()
    if battery_struct.dtype.names is None or 'cycle' not in battery_struct.dtype.names:
        return pd.DataFrame()

    cycles = battery_struct['cycle'][0, 0]
    if not isinstance(cycles, np.ndarray):
        cycles = np.array([cycles])

    eis_records = []

    for cycle_idx in range(cycles.size):
        try:
            cycle = cycles.flat[cycle_idx]
            if not hasattr(cycle, 'dtype') or cycle.dtype.names is None:
                continue

            try:
                cycle_type = cycle['type']
                if isinstance(cycle_type, np.ndarray):
                    cycle_type = cycle_type.flat[0]
                cycle_type = str(cycle_type).strip()
            except (IndexError, KeyError):
                continue

            if cycle_type != 'impedance':
                continue

            try:
                data = cycle['data']
            except (IndexError, KeyError):
                continue

            if isinstance(data, np.ndarray) and data.size > 0:
                data = data.flat[0]

            if not isinstance(data, np.void) and not isinstance(data, np.ndarray):
                continue

            imp_med = re_med = rct_med = np.nan

            # BUG 1 FIX: iterate fields, compute magnitude BEFORE any scalar reduction
            for field_name in ['Impedance', 'Rectified_impedance', 'Battery_impedance']:
                try:
                    val = data[field_name]
                    if isinstance(val, np.ndarray) and val.size > 0:
                        # suppress the ComplexWarning — we handle complex explicitly
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            imp_med, re_med, rct_med = _safe_impedance_scalar(val.flatten())
                    elif val is not None:
                        scalar = complex(val)
                        imp_med = abs(scalar)
                        re_med  = scalar.real
                        rct_med = abs(scalar.imag)
                    if not np.isnan(imp_med) and _IMP_MIN_OHM < imp_med <= _IMP_MAX_OHM:
                        break
                    # reset if out of bounds
                    imp_med = re_med = rct_med = np.nan
                except (IndexError, ValueError, TypeError, KeyError):
                    continue

            if np.isnan(imp_med):
                continue

            eis_records.append({
                'battery_id':    battery_id,
                'cycle_idx':     cycle_idx,
                'impedance_ohm': imp_med,
                're_ohm':        re_med,
                'rct_ohm':       rct_med,
            })

        except Exception as e:
            logger.debug(f"{battery_id} cycle {cycle_idx}: {e}")
            continue

    df = pd.DataFrame(eis_records)
    logger.info(f"{battery_id}: Parsed {len(df)} EIS records")
    return df


def parse_all_mat_files(raw_data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all .mat files in directory recursively.

    BUG 4 fix: skip battery_ids already parsed (deduplication by stem).
    """
    raw_path = Path(raw_data_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    mat_files = list(raw_path.rglob('*.mat'))
    if not mat_files:
        logger.warning(f"No .mat files found in {raw_path}")
        return pd.DataFrame(), pd.DataFrame()

    logger.info(f"Found {len(mat_files)} .mat files")

    all_discharge = []
    all_eis       = []
    seen_ids: set = set()          # BUG 4: track parsed battery_ids

    for mat_file in sorted(mat_files):
        try:
            battery_id = mat_file.stem

            # BUG 4 FIX: skip duplicate battery_id
            if battery_id in seen_ids:
                logger.info(f"Skipping duplicate {battery_id} from {mat_file}")
                continue
            seen_ids.add(battery_id)

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
    eis_combined       = pd.concat(all_eis,       ignore_index=True) if all_eis       else pd.DataFrame()

    logger.info(
        f"Total (deduplicated): {len(discharge_combined):,} discharge "
        f"+ {len(eis_combined):,} EIS records from {len(seen_ids)} batteries"
    )
    return discharge_combined, eis_combined
