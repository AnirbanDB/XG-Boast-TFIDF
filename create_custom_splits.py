#!/usr/bin/env python3
"""
Create custom dataset splits as requested:
- Training set: 15435 samples (CSV form)
- Validation set: 1800 samples (CSV form)  
- Test set: 200 samples (CSV form)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

def create_custom_splits():
    """Create the exact dataset splits requested by the user."""
    
    print("TARGET: Creating custom dataset splits as requested...")
    print("Target sizes:")
    print("- Training set: 15,435 samples")
    print("- Validation set: 1,800 samples") 
    print("- Test set: 200 samples")
    print("- Total: 17,435 samples")
    
    # Load the full dataset
    data_path = "../firco_alerts_final_5000_7.csv"
    print(f"\n📁 Loading dataset from: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"SUCCESS: Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check if we have enough data
    required_total = 15435 + 1800 + 200  # 17,435
    if len(df) < required_total:
        print(f"WARNING:  Warning: Dataset has {len(df)} rows but we need {required_total} rows")
        print(f"   We'll use all available data and adjust proportions")
        
        # Use proportional splits if we don't have enough data
        train_ratio = 15435 / required_total  # ~0.886
        val_ratio = 1800 / required_total     # ~0.103
        test_ratio = 200 / required_total     # ~0.011
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_ratio,
            random_state=42,
            stratify=df.get('decision.last_action', None) if 'decision.last_action' in df.columns else None
        )
        
        # Second split: val vs test from temp_df
        val_size = val_ratio / (val_ratio + test_ratio)  # Proportion of val in the remaining data
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            random_state=42,
            stratify=temp_df.get('decision.last_action', None) if 'decision.last_action' in temp_df.columns else None
        )
        
    else:
        # We have enough data - create exact splits
        print("SUCCESS: Sufficient data available for exact splits")
        
        # First, create test set (200 samples)
        remaining_df, test_df = train_test_split(
            df,
            test_size=200,
            random_state=42,
            stratify=df.get('decision.last_action', None) if 'decision.last_action' in df.columns else None
        )
        
        # Then split remaining into train (15435) and validation (1800)
        train_df, val_df = train_test_split(
            remaining_df,
            train_size=15435,
            test_size=1800,
            random_state=42,
            stratify=remaining_df.get('decision.last_action', None) if 'decision.last_action' in remaining_df.columns else None
        )
    
    # Verify splits
    print(f"\nSTATS: Final split sizes:")
    print(f"   Training: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    print(f"   Total: {len(train_df) + len(val_df) + len(test_df)} samples")
    
    # Create splits directory
    splits_dir = "splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    # Save the splits with clear names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    train_path = os.path.join(splits_dir, f"train_set_{len(train_df)}_samples_{timestamp}.csv")
    val_path = os.path.join(splits_dir, f"validation_set_{len(val_df)}_samples_{timestamp}.csv")
    test_path = os.path.join(splits_dir, f"test_set_{len(test_df)}_samples_{timestamp}.csv")
    
    # Save the datasets
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSAVED: Splits saved successfully:")
    print(f"   📂 Training set: {train_path}")
    print(f"   📂 Validation set: {val_path}")
    print(f"   📂 Test set: {test_path}")
    
    # Also create convenient symlinks with simple names
    simple_train_path = os.path.join(splits_dir, "training_set.csv")
    simple_val_path = os.path.join(splits_dir, "validation_set.csv")
    simple_test_path = os.path.join(splits_dir, "test_set.csv")
    
    # Remove existing symlinks if they exist
    for path in [simple_train_path, simple_val_path, simple_test_path]:
        if os.path.exists(path):
            os.remove(path)
    
    # Copy files with simple names
    train_df.to_csv(simple_train_path, index=False)
    val_df.to_csv(simple_val_path, index=False)
    test_df.to_csv(simple_test_path, index=False)
    
    print(f"\n🔗 Simple names created for easy access:")
    print(f"   📂 Training: splits/training_set.csv ({len(train_df)} samples)")
    print(f"   📂 Validation: splits/validation_set.csv ({len(val_df)} samples)")
    print(f"   📂 Test: splits/test_set.csv ({len(test_df)} samples)")
    
    # Show class distribution if possible
    if 'decision.last_action' in df.columns:
        print(f"\nDISTRIBUTION: Class distribution in decision.last_action:")
        print("Training set:")
        print(train_df['decision.last_action'].value_counts())
        print("\nValidation set:")
        print(val_df['decision.last_action'].value_counts())
        print("\nTest set:")
        print(test_df['decision.last_action'].value_counts())
    
    print(f"\nCOMPLETE: Custom dataset splits created successfully!")
    print(f"SUCCESS: Ready for Swagger UI testing:")
    print(f"   - Use training_set.csv for /train endpoint")
    print(f"   - Use validation_set.csv for /validate endpoint") 
    print(f"   - Use test_set.csv for /predict endpoint")
    
    return {
        'train_path': simple_train_path,
        'val_path': simple_val_path,
        'test_path': simple_test_path,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df)
    }

if __name__ == "__main__":
    result = create_custom_splits()
    print(f"\nDONE: All done! Use these files in your Swagger UI tests.") 