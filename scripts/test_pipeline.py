#!/usr/bin/env python
"""Quick test of BaFuse data pipeline and training"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all imports work"""
    logger.info("="*80)
    logger.info("STEP 1: Testing imports...")
    logger.info("="*80)
    
    try:
        from src.data.parse_mat import parse_all_mat_files
        logger.info("✓ parse_mat imported")
        
        from src.data.pairing import pair_discharge_eis, validate_pairs
        logger.info("✓ pairing imported")
        
        from src.data.split import split_by_battery, get_split_statistics
        logger.info("✓ split imported")
        
        from src.data.dataset import create_dataloaders
        logger.info("✓ dataset imported")
        
        import torch
        logger.info(f"✓ torch {torch.__version__} imported")
        
        import pandas as pd
        logger.info(f"✓ pandas {pd.__version__} imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_parsing():
    """Test data parsing"""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Testing data parsing...")
    logger.info("="*80)
    
    try:
        from src.data.parse_mat import parse_all_mat_files
        
        data_dir = "5. BatteryDataSet"
        if not os.path.exists(data_dir):
            logger.error(f"✗ Data directory not found: {data_dir}")
            return False
        
        logger.info(f"Parsing from: {data_dir}")
        discharge_df, eis_df = parse_all_mat_files(data_dir)
        
        logger.info(f"✓ Parsed {len(discharge_df)} discharge records")
        logger.info(f"✓ Parsed {len(eis_df)} EIS records")
        
        if len(discharge_df) == 0 or len(eis_df) == 0:
            logger.error("✗ No data parsed!")
            return False
        
        logger.info(f"  Discharge shape: {discharge_df.shape}")
        logger.info(f"  EIS shape: {eis_df.shape}")
        
        return discharge_df, eis_df
    except Exception as e:
        logger.error(f"✗ Data parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pairing(discharge_df, eis_df):
    """Test discharge-EIS pairing"""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Testing discharge-EIS pairing...")
    logger.info("="*80)
    
    try:
        from src.data.pairing import pair_discharge_eis, validate_pairs
        
        paired_df = pair_discharge_eis(discharge_df, eis_df)
        logger.info(f"✓ Created {len(paired_df)} pairs")
        
        if len(paired_df) == 0:
            logger.error("✗ No pairs created!")
            return False
        
        stats = validate_pairs(paired_df)
        logger.info(f"  Batteries: {stats.get('num_batteries', 'N/A')}")
        logger.info(f"  Capacity range: {stats.get('capacity_range', 'N/A')}")
        
        return paired_df
    except Exception as e:
        logger.error(f"✗ Pairing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_split(paired_df):
    """Test train/val/test split"""
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Testing train/val/test split...")
    logger.info("="*80)
    
    try:
        from src.data.split import split_by_battery, get_split_statistics
        
        train_df, val_df, test_df = split_by_battery(
            paired_df,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        logger.info(f"✓ Train: {len(train_df)} samples")
        logger.info(f"✓ Val: {len(val_df)} samples")
        logger.info(f"✓ Test: {len(test_df)} samples")
        
        stats = get_split_statistics(train_df, val_df, test_df)
        logger.info(f"  Total samples: {stats.get('total_samples', 'N/A')}")
        logger.info(f"  Total batteries: {stats.get('total_batteries', 'N/A')}")
        
        return train_df, val_df, test_df
    except Exception as e:
        logger.error(f"✗ Split failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(train_df, val_df, test_df, discharge_df):
    """Test PyTorch DataLoader creation"""
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Testing PyTorch DataLoader...")
    logger.info("="*80)
    
    try:
        from src.data.dataset import create_dataloaders
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df,
            val_df,
            test_df,
            discharge_data_df=discharge_df,
            batch_size=8,
            num_workers=0
        )
        
        logger.info(f"✓ Train loader: {len(train_loader)} batches")
        logger.info(f"✓ Val loader: {len(val_loader)} batches")
        logger.info(f"✓ Test loader: {len(test_loader)} batches")
        
        # Inspect first batch
        batch = next(iter(train_loader))
        logger.info(f"\n  Batch sample:")
        logger.info(f"    discharge: {batch['discharge'].shape}")
        logger.info(f"    eis: {batch['eis'].shape}")
        logger.info(f"    physics: {batch['physics'].shape}")
        logger.info(f"    soh_label: {batch['soh_label'].shape}")
        logger.info(f"    SoH values: {batch['soh_label'][:4].tolist()}")
        
        return train_loader, val_loader, test_loader
    except Exception as e:
        logger.error(f"✗ DataLoader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model creation"""
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Testing BaFuse model...")
    logger.info("="*80)
    
    try:
        import torch
        from src.models.bafuse import BaFuse
        
        model = BaFuse(
            discharge_input_size=5,
            eis_num_frequencies=3,
            physics_num_features=4,
            latent_dim=32,
            fusion_dim=64
        )
        
        logger.info(f"✓ BaFuse model created")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        discharge = torch.randn(4, 5)
        eis = torch.randn(4, 3)
        physics = torch.randn(4, 4)
        
        output = model(discharge, eis, physics)
        logger.info(f"✓ Forward pass works")
        logger.info(f"  Output keys: {output.keys()}")
        logger.info(f"  SoH prediction shape: {output['soh_pred'].shape}")
        
        return model
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(train_loader, model):
    """Test one training step"""
    logger.info("\n" + "="*80)
    logger.info("STEP 7: Testing training step...")
    logger.info("="*80)
    
    try:
        import torch
        from src.losses import SoHPredictionLoss
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  Device: {device}")
        
        model = model.to(device)
        criterion = SoHPredictionLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Get first batch
        batch = next(iter(train_loader))
        discharge = batch['discharge'].to(device)
        eis = batch['eis'].to(device)
        physics = batch['physics'].to(device)
        soh_label = batch['soh_label'].to(device)
        
        # Forward pass
        output = model(discharge, eis, physics)
        pred = output['soh_pred'].squeeze()
        
        # Compute loss
        loss = criterion(pred, soh_label)
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Loss: {loss.item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info(f"✓ Backward pass successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "BaFuse Training Pipeline Test" + " "*29 + "║")
    logger.info("╚" + "="*78 + "╝")
    
    # Test 1: Imports
    if not test_imports():
        logger.error("\n✗ Import test failed! Check dependencies.")
        return 1
    
    # Test 2: Data parsing
    data_result = test_data_parsing()
    if not data_result:
        logger.error("\n✗ Data parsing failed!")
        return 1
    discharge_df, eis_df = data_result
    
    # Test 3: Pairing
    paired_df = test_pairing(discharge_df, eis_df)
    if paired_df is False:
        logger.error("\n✗ Pairing failed!")
        return 1
    
    # Test 4: Split
    split_result = test_split(paired_df)
    if not split_result:
        logger.error("\n✗ Split failed!")
        return 1
    train_df, val_df, test_df = split_result
    
    # Test 5: DataLoader
    loader_result = test_dataloader(train_df, val_df, test_df, discharge_df)
    if not loader_result:
        logger.error("\n✗ DataLoader creation failed!")
        return 1
    train_loader, val_loader, test_loader = loader_result
    
    # Test 6: Model
    model = test_model()
    if not model:
        logger.error("\n✗ Model creation failed!")
        return 1
    
    # Test 7: Training
    if not test_training_step(train_loader, model):
        logger.error("\n✗ Training step failed!")
        return 1
    
    # Success!
    logger.info("\n" + "="*80)
    logger.info("✓ ALL TESTS PASSED!")
    logger.info("="*80)
    logger.info("\nYour BaFuse pipeline is ready to train!")
    logger.info("\nNext steps:")
    logger.info("  1. Review config.yaml for training hyperparameters")
    logger.info("  2. Run: python src/train.py --config configs/config.yaml")
    logger.info("  3. Evaluate: python src/evaluate.py")
    logger.info("  4. Ablation: python src/ablation.py")
    logger.info("\n")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
