#!/usr/bin/env python3
"""
Test script to verify batch prediction functionality
"""

import os
import sys
import glob
from pathlib import Path

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from main_xgb_F import predict_batch_csv
from api_utils import get_latest_model_path, load_model

def test_batch_prediction():
    """Test batch prediction with the test CSV file"""
    
    # Paths
    test_csv_path = os.path.join(current_dir, "splits", "test_set.csv")
    
    # Check if test file exists
    if not os.path.exists(test_csv_path):
        print(f"❌ Test CSV file not found: {test_csv_path}")
        return False
    
    print(f"✅ Found test CSV file: {test_csv_path}")
    
    # Get latest model
    model_path = get_latest_model_path()
    if not model_path:
        print(f"❌ No trained model found in {os.path.join(current_dir, 'saved_models')}")
        print("Available files:")
        for f in os.listdir(os.path.join(current_dir, "saved_models")):
            print(f"  - {f}")
        return False
    
    print(f"✅ Found model: {model_path}")
    
    # Test model loading
    print("🔄 Testing model loading...")
    model = load_model(model_path)
    if not model:
        print(f"❌ Failed to load model from {model_path}")
        return False
    print("✅ Model loaded successfully")
    
    # Count lines in CSV file
    with open(test_csv_path, 'r') as f:
        line_count = sum(1 for line in f)
    data_rows = line_count - 1  # Subtract header
    print(f"✅ CSV file has {data_rows} data rows (+ 1 header)")
    
    # Run batch prediction
    print("\n🔄 Running batch prediction...")
    try:
        results = predict_batch_csv(test_csv_path, model_path)
        
        print(f"\n📊 BATCH PREDICTION RESULTS:")
        print(f"   - Total predictions generated: {len(results)}")
        print(f"   - Expected predictions: {data_rows}")
        print(f"   - Match: {'✅' if len(results) == data_rows else '❌'}")
        
        if len(results) > 0:
            # Check first prediction structure
            first_result = results[0]
            print(f"\n📋 First prediction structure:")
            print(f"   - Row index: {first_result['row_index']}")
            print(f"   - Targets predicted: {list(first_result['predictions'].keys())}")
            
            # Check if we have predictions for all target columns
            target_count = len(first_result['predictions'])
            print(f"   - Number of targets: {target_count}")
            
            # Show sample predictions for first row
            print(f"\n🔍 Sample predictions for row 0:")
            for target, pred_data in first_result['predictions'].items():
                predicted_class = pred_data['predicted_class']
                prob_keys = list(pred_data['probabilities'].keys())
                print(f"   - {target}: {predicted_class} (classes: {prob_keys})")
        
        # Check last prediction
        if len(results) > 1:
            last_result = results[-1]
            print(f"\n📋 Last prediction (row {last_result['row_index']}):")
            for target, pred_data in last_result['predictions'].items():
                predicted_class = pred_data['predicted_class']
                print(f"   - {target}: {predicted_class}")
        
        # Verify row indices are sequential
        row_indices = [r['row_index'] for r in results]
        expected_indices = list(range(len(results)))
        indices_match = row_indices == expected_indices
        print(f"\n🔍 Row indices verification:")
        print(f"   - Sequential indices: {'✅' if indices_match else '❌'}")
        print(f"   - First 5 indices: {row_indices[:5]}")
        print(f"   - Last 5 indices: {row_indices[-5:]}")
        
        return len(results) == data_rows and indices_match
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 BATCH PREDICTION TEST")
    print("=" * 50)
    
    success = test_batch_prediction()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ BATCH PREDICTION TEST PASSED")
        print("All 200 rows should be processed correctly in the API.")
    else:
        print("❌ BATCH PREDICTION TEST FAILED")
        print("There's an issue with the batch prediction logic.")
    
    sys.exit(0 if success else 1)
