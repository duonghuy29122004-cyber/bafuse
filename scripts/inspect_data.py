#!/usr/bin/env python
"""
Inspect NASA PCoE .mat file structure and test data parsing.

Usage:
    python scripts/inspect_data.py                        # inspect first .mat file
    python scripts/inspect_data.py --file B0005.mat       # specific file
    python scripts/inspect_data.py --full                 # full parse test
"""

import argparse
import sys
import logging
from pathlib import Path

import scipy.io as sio
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# .mat structure inspection
# ─────────────────────────────────────────────

def inspect_mat(mat_file: Path):
    """Print full structure of a .mat file."""
    print(f"\nInspecting: {mat_file}")
    print("=" * 70)

    mat_data = sio.loadmat(str(mat_file), squeeze_me=False)
    battery_id = mat_file.stem

    # Top-level keys
    print("Top-level keys:")
    for key in sorted(mat_data.keys()):
        if key.startswith("__"):
            continue
        val = mat_data[key]
        shape = getattr(val, "shape", "N/A")
        print(f"  {key}: {type(val).__name__}  shape={shape}")
        if hasattr(val, "dtype") and val.dtype.names:
            print(f"    fields: {val.dtype.names}")

    if battery_id not in mat_data:
        print(f"  [!] Key '{battery_id}' not found in file")
        return

    battery_struct = mat_data[battery_id]
    cycles = battery_struct["cycle"][0, 0]
    print(f"\nTotal cycles: {cycles.size}")

    # Cycle type summary
    type_counts: dict = {}
    for i in range(cycles.size):
        c = cycles.flat[i]
        ct = c["type"]
        if isinstance(ct, np.ndarray):
            ct = ct.flat[0]
        ct = str(ct).strip()
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print("Cycle type counts:", type_counts)

    # First cycle of each type
    seen: set = set()
    for i in range(cycles.size):
        c = cycles.flat[i]
        ct = c["type"]
        if isinstance(ct, np.ndarray):
            ct = ct.flat[0]
        ct = str(ct).strip()
        if ct in seen:
            continue
        seen.add(ct)

        print(f"\n--- First '{ct}' cycle (index {i}) ---")
        data = c["data"]
        if isinstance(data, np.ndarray) and data.size > 0:
            data = data.flat[0]

        if hasattr(data, "dtype") and data.dtype.names:
            print(f"  fields : {data.dtype.names}")
            for field in data.dtype.names:
                raw = data[field]
                arr = raw.flatten() if isinstance(raw, np.ndarray) else np.atleast_1d(raw)
                print(f"  {field:30s}: len={len(arr)}  sample={arr[:3]}")
        elif isinstance(data, np.ndarray):
            print(f"  shape={data.shape}  dtype={data.dtype}  sample={data.flat[:3]}")

        if len(seen) >= 3:
            break


# ─────────────────────────────────────────────
# Parse test
# ─────────────────────────────────────────────

def test_parse(mat_file: Path):
    """Run parse_discharge_curve and parse_eis_spectrum and show results."""
    from src.data.parse_mat import load_mat_file, parse_discharge_curve, parse_eis_spectrum

    print(f"\nParse test: {mat_file}")
    print("=" * 70)

    mat_data = load_mat_file(str(mat_file))
    battery_id = mat_file.stem

    discharge_df = parse_discharge_curve(mat_data, battery_id)
    eis_df = parse_eis_spectrum(mat_data, battery_id)

    print(f"Discharge records : {len(discharge_df)}")
    if not discharge_df.empty:
        print(f"  columns  : {list(discharge_df.columns)}")
        print(f"  cycles   : {discharge_df['cycle_idx'].nunique()}")
        print(f"  cap range: {discharge_df['capacity_ahr'].min():.4f} – {discharge_df['capacity_ahr'].max():.4f} Ahr")
        print(f"  sample   :\n{discharge_df.head(3).to_string(index=False)}")

    print(f"\nEIS records       : {len(eis_df)}")
    if not eis_df.empty:
        print(f"  columns  : {list(eis_df.columns)}")
        print(f"  sample   :\n{eis_df.head(3).to_string(index=False)}")


# ─────────────────────────────────────────────
# Full pipeline test
# ─────────────────────────────────────────────

def test_full_pipeline(data_dir: str):
    """Parse all .mat files, pair, and show stats."""
    from src.data.parse_mat import parse_all_mat_files
    from src.data.pairing import pair_discharge_eis, validate_pairs

    print(f"\nFull pipeline test on: {data_dir}")
    print("=" * 70)

    discharge_df, eis_df = parse_all_mat_files(data_dir)
    print(f"Total discharge records : {len(discharge_df):,}")
    print(f"Total EIS records       : {len(eis_df):,}")

    paired_df = pair_discharge_eis(discharge_df, eis_df)
    stats = validate_pairs(paired_df)
    print(f"\nPairing stats:")
    for k, v in stats.items():
        if not isinstance(v, (list, dict)):
            print(f"  {k}: {v}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Inspect NASA PCoE .mat data")
    p.add_argument("--file",     default=None, help="Specific .mat file path")
    p.add_argument("--data_dir", default="5. BatteryDataSet")
    p.add_argument("--full",     action="store_true", help="Run full pipeline test")
    p.add_argument("--parse",    action="store_true", help="Run parse test only")
    args = p.parse_args()

    if args.full:
        test_full_pipeline(args.data_dir)
        return

    # Find target .mat file
    if args.file:
        mat_file = Path(args.file)
    else:
        mat_files = list(Path(args.data_dir).rglob("*.mat"))
        if not mat_files:
            print(f"No .mat files found in {args.data_dir}")
            sys.exit(1)
        mat_file = mat_files[0]
        print(f"No --file specified, using: {mat_file}")

    inspect_mat(mat_file)

    if args.parse:
        test_parse(mat_file)


if __name__ == "__main__":
    main()

