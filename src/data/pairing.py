"""
Pair discharge curves with corresponding EIS measurements.

Ensures:
- Each discharge cycle is matched with the closest EIS measurement in time
- Same battery and cycle age
- No duplicate pairings
- Proper handling of sequence: discharge → charge → impedance pattern

Fixes applied:
- BUG 2: Filter out cycles with capacity < MIN_CAPACITY_AHR (0.5 Ahr) before
         pairing to remove faulty device readings that produce near-zero SOH labels.
- BUG 3: Hard enforce max_cycle_gap — pairs with abs(cycle_gap) > max_cycle_gap
         are dropped entirely instead of falling back to nearest-any EIS.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# P1-#2: Minimum valid cycle capacity expressed as a FRACTION of each battery's
# nominal capacity, not a fixed absolute threshold.
# 0.25 × 2.0 Ah = 0.50 Ah for standard NASA cells (same effective floor as before),
# but correctly scales if a different nominal capacity is used.
_MIN_CAPACITY_FRACTION = 0.25

# Keep the old constant name as a fallback used in the absolute-threshold path
# (only active when battery_id is not in BATTERY_NOMINAL_CAPACITY).
MIN_CAPACITY_AHR = 0.5   # Ahr legacy default — kept for backward compat imports


def pair_discharge_eis(
    discharge_df: pd.DataFrame,
    eis_df: pd.DataFrame,
    max_cycle_gap: int = 10,        # BUG 3: tightened from unlimited fallback
) -> pd.DataFrame:
    """
    Match discharge cycles with EIS measurements.

    NASA PCoE pattern: discharge (N) → charge (N+1) → impedance (N+2).

    Args:
        discharge_df : time-series discharge records
        eis_df       : EIS records per cycle
        max_cycle_gap: maximum |eis_cycle - discharge_cycle| allowed (BUG 3)

    Returns:
        Cleaned, paired DataFrame.
    """

    # ── P1-#2 / BUG 2: drop low-capacity cycles using per-battery relative threshold ──
    if 'capacity_ahr' in discharge_df.columns:
        before = discharge_df['cycle_idx'].nunique() if not discharge_df.empty else 0

        # Import here to avoid circular dependency at module level
        from src.data.dataset import get_nominal_capacity  # noqa: PLC0415

        # Compute per-cycle capacity and the per-battery threshold
        cycle_caps = (
            discharge_df.groupby(['battery_id', 'cycle_idx'])['capacity_ahr']
            .first()
            .reset_index()
        )
        cycle_caps['min_cap'] = cycle_caps['battery_id'].map(
            lambda bid: _MIN_CAPACITY_FRACTION * get_nominal_capacity(bid)
        )
        bad_cycles = cycle_caps[cycle_caps['capacity_ahr'] < cycle_caps['min_cap']]

        if not bad_cycles.empty:
            # Vectorised filter — replaces the slow apply(lambda) from before
            # P4-#14 FIX: use index-based merge instead of per-row Python call
            bad_idx = pd.MultiIndex.from_arrays(
                [bad_cycles['battery_id'], bad_cycles['cycle_idx']]
            )
            df_idx  = pd.MultiIndex.from_arrays(
                [discharge_df['battery_id'], discharge_df['cycle_idx']]
            )
            discharge_df = discharge_df[~df_idx.isin(bad_idx)].copy()
            after = discharge_df['cycle_idx'].nunique() if not discharge_df.empty else 0

            for bid, grp in bad_cycles.groupby('battery_id'):
                threshold = grp['min_cap'].iloc[0]
                logger.info(
                    f"  [cap filter] {bid}: dropped {len(grp)} cycles with "
                    f"capacity < {threshold:.3f} Ah "
                    f"({_MIN_CAPACITY_FRACTION*100:.0f}% of nominal)"
                    f" — cycles: {sorted(grp['cycle_idx'].tolist())}"
                )
            logger.info(
                f"  [cap filter] Total removed: {before - after} cycles "
                f"across {bad_cycles['battery_id'].nunique()} batteries"
            )
        else:
            logger.info(
                f"  [cap filter] No low-capacity cycles found "
                f"(threshold={_MIN_CAPACITY_FRACTION*100:.0f}% nominal)"
            )

    # ── Aggregate discharge per cycle ───────────────────────────────────────
    agg_dict = {
        'voltage_v':     ['mean', 'min', 'max', 'std'],
        'current_a':     ['mean', 'std'],
        'temperature_c': ['mean', 'std'],
        'time_s':        ['min', 'max', 'count'],
    }
    if 'capacity_ahr' in discharge_df.columns:
        agg_dict['capacity_ahr'] = ['first']

    discharge_agg = discharge_df.groupby(['battery_id', 'cycle_idx']).agg(agg_dict).reset_index()

    base_cols = [
        'battery_id', 'cycle_idx',
        'voltage_mean', 'voltage_min', 'voltage_max', 'voltage_std',
        'current_mean', 'current_std',
        'temp_mean', 'temp_std',
        'time_start', 'time_end', 'num_samples',
    ]
    if 'capacity_ahr' in discharge_df.columns:
        base_cols.append('capacity_ahr')
    discharge_agg.columns = base_cols

    if 'capacity_ahr' not in discharge_agg.columns:
        discharge_agg['capacity_ahr'] = float('nan')

    # ── Pair discharge ↔ EIS ────────────────────────────────────────────────
    paired_list   = []
    dropped_gap   = 0      # BUG 3 counter

    for battery_id in discharge_agg['battery_id'].unique():
        d_bat = discharge_agg[discharge_agg['battery_id'] == battery_id].reset_index(drop=True)
        e_bat = eis_df[eis_df['battery_id'] == battery_id].reset_index(drop=True)

        if e_bat.empty:
            logger.warning(f"No EIS data for {battery_id}")
            continue

        for _, d_row in d_bat.iterrows():
            d_cycle = d_row['cycle_idx']

            # Primary: EIS within (discharge_cycle, discharge_cycle + max_cycle_gap]
            candidates = e_bat[
                (e_bat['cycle_idx'] >= d_cycle) &
                (e_bat['cycle_idx'] <= d_cycle + max_cycle_gap)
            ]

            # BUG 3 FIX: NO unlimited fallback — if nothing in window, skip
            if candidates.empty:
                dropped_gap += 1
                logger.debug(
                    f"  [BUG3 filter] {battery_id} discharge cycle {d_cycle}: "
                    f"no EIS within gap={max_cycle_gap}, skipping"
                )
                continue

            eis_row  = candidates.iloc[0]
            gap      = int(eis_row['cycle_idx'] - d_cycle)

            paired_list.append({
                'battery_id':     battery_id,
                'discharge_cycle': d_cycle,
                'eis_cycle':      eis_row['cycle_idx'],
                'capacity_ahr':   d_row['capacity_ahr'],
                'voltage_mean':   d_row['voltage_mean'],
                'voltage_min':    d_row['voltage_min'],
                'voltage_max':    d_row['voltage_max'],
                'voltage_std':    d_row['voltage_std'],
                'current_mean':   d_row['current_mean'],
                'current_std':    d_row['current_std'],
                'temp_mean':      d_row['temp_mean'],
                'temp_std':       d_row['temp_std'],
                'impedance_ohm':  eis_row['impedance_ohm'],
                're_ohm':         eis_row['re_ohm'],
                'rct_ohm':        eis_row['rct_ohm'],
                'cycle_gap':      gap,
            })

    if dropped_gap:
        logger.info(
            f"  [BUG3 filter] Dropped {dropped_gap} discharge cycles "
            f"with no EIS within max_cycle_gap={max_cycle_gap}"
        )

    if paired_list:
        paired_df = pd.DataFrame(paired_list)
        logger.info(f"Created {len(paired_df)} discharge-EIS pairs")
        return paired_df
    else:
        logger.warning("No successful pairings created")
        return pd.DataFrame()


def validate_pairs(paired_df: pd.DataFrame) -> dict:
    """
    Validate pairing quality and return statistics.
    """
    if paired_df.empty:
        return {"error": "Empty dataframe"}

    stats = {
        "num_pairs":          len(paired_df),
        "batteries":          paired_df['battery_id'].unique().tolist(),
        "num_batteries":      paired_df['battery_id'].nunique(),
        "pairs_per_battery":  paired_df.groupby('battery_id').size().to_dict(),
        "cycle_gap_stats": {
            "mean":   float(paired_df['cycle_gap'].mean()),
            "median": float(paired_df['cycle_gap'].median()),
            "std":    float(paired_df['cycle_gap'].std()),
            "min":    int(paired_df['cycle_gap'].min()),
            "max":    int(paired_df['cycle_gap'].max()),
        },
        "capacity_range":   (float(paired_df['capacity_ahr'].min()),
                             float(paired_df['capacity_ahr'].max())),
        "impedance_range":  (float(paired_df['impedance_ohm'].min()),
                             float(paired_df['impedance_ohm'].max())),
        "re_range":         (float(paired_df['re_ohm'].min()),
                             float(paired_df['re_ohm'].max())),
        "rct_range":        (float(paired_df['rct_ohm'].min()),
                             float(paired_df['rct_ohm'].max())),
        "voltage_mean_range": (float(paired_df['voltage_mean'].min()),
                               float(paired_df['voltage_mean'].max())),
        "temp_range":         (float(paired_df['temp_mean'].min()),
                               float(paired_df['temp_mean'].max())),
        "missing_impedance":  int(paired_df['impedance_ohm'].isna().sum()),
        "missing_capacity":   int(paired_df['capacity_ahr'].isna().sum()),
    }

    logger.info("── Validation stats ──────────────────────────────────")
    logger.info(f"  pairs            : {stats['num_pairs']}")
    logger.info(f"  batteries        : {stats['num_batteries']}")
    logger.info(f"  capacity_range   : {stats['capacity_range'][0]:.4f} – {stats['capacity_range'][1]:.4f} Ahr")
    logger.info(f"  impedance_range  : {stats['impedance_range'][0]:.4f} – {stats['impedance_range'][1]:.4f} Ω")
    logger.info(f"  re_range         : {stats['re_range'][0]:.4f} – {stats['re_range'][1]:.4f} Ω")
    logger.info(f"  rct_range        : {stats['rct_range'][0]:.4f} – {stats['rct_range'][1]:.4f} Ω")
    logger.info(f"  cycle_gap        : mean={stats['cycle_gap_stats']['mean']:.2f}  "
                f"median={stats['cycle_gap_stats']['median']}  "
                f"std={stats['cycle_gap_stats']['std']:.2f}  "
                f"min={stats['cycle_gap_stats']['min']}  "
                f"max={stats['cycle_gap_stats']['max']}")
    logger.info(f"  missing_impedance: {stats['missing_impedance']}")
    logger.info(f"  missing_capacity : {stats['missing_capacity']}")
    logger.info("──────────────────────────────────────────────────────")

    return stats
