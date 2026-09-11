"""
Complete data processing pipeline for BaFuse.

Usage:
    python process_data.py --data_dir "5. BatteryDataSet" --output_dir "data/processed"

This script:
1. Parses all .mat files
2. Pairs discharge with EIS
3. Performs battery-level split
4. Creates PyTorch DataLoaders
5. Saves processed data
"""

import argparse
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.parse_mat import parse_all_mat_files
from src.data.pairing import pair_discharge_eis, validate_pairs
from src.data.split import split_by_battery, get_split_statistics
from src.data.dataset import create_dataloaders


def main(args):
    """Run complete data pipeline."""
    
    logger.info("="*80)
    logger.info("BaFuse Data Processing Pipeline")
    logger.info("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # ========== STEP 1: PARSE .mat FILES ==========
    logger.info("\n[STEP 1] Parsing .mat files...")
    try:
        discharge_df, eis_df = parse_all_mat_files(args.data_dir)
        logger.info(f"✓ Parsed {len(discharge_df)} discharge records")
        logger.info(f"✓ Parsed {len(eis_df)} EIS records")
    except Exception as e:
        logger.error(f"✗ Failed to parse .mat files: {e}")
        return 1
    
    # Save parsed data
    discharge_df.to_pickle(output_dir / "discharge_raw.pkl")
    eis_df.to_pickle(output_dir / "eis_raw.pkl")
    logger.info(f"Saved raw data to {output_dir}")
    
    # ========== STEP 2: PAIR DISCHARGE WITH EIS ==========
    logger.info("\n[STEP 2] Pairing discharge with EIS...")
    try:
        paired_df = pair_discharge_eis(discharge_df, eis_df, max_cycle_gap=2)
        logger.info(f"✓ Created {len(paired_df)} discharge-EIS pairs")
        
        # Validate pairs
        val_stats = validate_pairs(paired_df)
        logger.info(f"✓ Validation stats:")
        for key, val in val_stats.items():
            if not isinstance(val, (list, dict)):
                logger.info(f"  - {key}: {val}")
    except Exception as e:
        logger.error(f"✗ Failed to pair data: {e}")
        return 1
    
    # Save paired data
    paired_df.to_pickle(output_dir / "paired.pkl")
    logger.info(f"Saved paired data to {output_dir / 'paired.pkl'}")
    
    # ========== STEP 3: BATTERY-LEVEL SPLIT ==========
    logger.info("\n[STEP 3] Battery-level train/val/test split...")
    try:
        train_df, val_df, test_df = split_by_battery(
            paired_df,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
            stratify_by_soh=True
        )
        logger.info(f"✓ Train: {len(train_df)} samples")
        logger.info(f"✓ Val:   {len(val_df)} samples")
        logger.info(f"✓ Test:  {len(test_df)} samples")
        
        # Get split statistics
        split_stats = get_split_statistics(train_df, val_df, test_df)
        logger.info(f"✓ Split statistics:")
        for key, val in split_stats.items():
            if not isinstance(val, dict):
                logger.info(f"  - {key}: {val}")
    except Exception as e:
        logger.error(f"✗ Failed to split data: {e}")
        return 1
    
    # Save split data
    train_df.to_pickle(output_dir / "train.pkl")
    val_df.to_pickle(output_dir / "val.pkl")
    test_df.to_pickle(output_dir / "test.pkl")
    logger.info(f"Saved split data to {output_dir}")
    
    # ========== STEP 4: CREATE DATALOADERS ==========
    logger.info("\n[STEP 4] Creating PyTorch DataLoaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df,
            val_df,
            test_df,
            discharge_data_df=discharge_df,  # Include raw time-series
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        logger.info(f"✓ Train loader: {len(train_loader)} batches")
        logger.info(f"✓ Val loader:   {len(val_loader)} batches")
        logger.info(f"✓ Test loader:  {len(test_loader)} batches")
    except Exception as e:
        logger.error(f"✗ Failed to create DataLoaders: {e}")
        return 1
    
    # ========== STEP 5: SAMPLE INSPECTION ==========
    logger.info("\n[STEP 5] Inspecting sample...")
    try:
        sample = next(iter(train_loader))
        logger.info(f"✓ Sample batch keys: {sample.keys()}")
        logger.info(f"  - discharge shape: {sample['discharge'].shape}")
        logger.info(f"  - eis shape: {sample['eis'].shape}")
        logger.info(f"  - physics shape: {sample['physics'].shape}")
        logger.info(f"  - soh_label shape: {sample['soh_label'].shape}")
        logger.info(f"  - SoH values (sample): {sample['soh_label'][:5].numpy()}")
    except Exception as e:
        logger.error(f"✗ Failed to inspect sample: {e}")
        return 1
    
    # ========== SUMMARY ==========
    logger.info("\n" + "="*80)
    logger.info("✓ Data processing complete!")
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("Files:")
    logger.info(f"  - discharge_raw.pkl (raw discharge measurements)")
    logger.info(f"  - eis_raw.pkl (raw EIS measurements)")
    logger.info(f"  - paired.pkl (paired discharge-EIS)")
    logger.info(f"  - train.pkl, val.pkl, test.pkl (split data)")
    logger.info("\nNext step: Run training with config.yaml")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BaFuse data processing pipeline"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="5. BatteryDataSet",
        help="Path to directory containing battery .mat files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers"
    )
    
    args = parser.parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
