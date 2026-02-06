#!/usr/bin/env python3
"""
Debug script to trace what's happening in the prediction endpoint.
"""

import os
import sys
import pandas as pd
from dataset_utils import load_and_preprocess_data
from api_utils import load_model, get_latest_model_path
from config import LABEL_COLUMNS

def debug_prediction_process():
    """Debug the prediction process step by step."""
    
    # Test file path
    test_file = "/Users/anirbandeb/Downloads/compliance_predictor-main/Firco/xgb/splits/test_set.csv"
    
    print("=== DEBUG: Prediction Process ===")
    
    # Step 1: Check raw file
    print(f"\n1. Raw file check:")
    raw_df = pd.read_csv(test_file)
    print(f"Raw CSV shape: {raw_df.shape}")
    print(f"Raw CSV columns: {len(raw_df.columns)}")
    
    # Step 2: Test data preprocessing
    print(f"\n2. Data preprocessing:")
    try:
        df = load_and_preprocess_data(test_file, is_training=False)
        print(f"Preprocessed shape: {df.shape}")
        print(f"Preprocessed columns: {len(df.columns)}")
        
        # Step 3: Prepare features
        print(f"\n3. Feature preparation:")
        X = df.drop(columns=LABEL_COLUMNS, errors='ignore')
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature columns count: {len(X.columns)}")
        
        # Step 4: Load model
        print(f"\n4. Model loading:")
        model_path = get_latest_model_path()
        print(f"Model path: {model_path}")
        
        if model_path and os.path.exists(model_path):
            model = load_model(model_path)
            if model:
                print(f"Model loaded successfully")
                
                # Step 5: Make predictions
                print(f"\n5. Making predictions:")
                batch_predictions = model.predict(X)
                print(f"Prediction keys: {list(batch_predictions.keys())}")
                
                for target, preds in batch_predictions.items():
                    print(f"Target '{target}': {len(preds)} predictions")
                    print(f"First 3 predictions: {preds[:3] if len(preds) > 3 else preds}")
                
                # Step 6: Generate formatted results
                print(f"\n6. Result formatting:")
                predictions_list = []
                
                for i in range(len(df)):
                    result = {}
                    for target, preds in batch_predictions.items():
                        if target in model.label_encoders:
                            encoder = model.label_encoders[target]
                            predicted_class_index = preds[i]
                            predicted_class = encoder.inverse_transform([predicted_class_index])[0] if predicted_class_index < len(encoder.classes_) else "Unknown"
                            
                            result[target] = {
                                'predicted_class': predicted_class,
                                'confidence': 'N/A'  # Simplified for debug
                            }
                    
                    predictions_list.append({
                        "input_id": i,
                        "input_text": df.iloc[i].get('hit.matching_text', 'N/A'),
                        "predictions": result
                    })
                
                print(f"Total formatted predictions: {len(predictions_list)}")
                print(f"Sample prediction (first row): {predictions_list[0] if predictions_list else 'None'}")
                
                return len(predictions_list)
            else:
                print("Failed to load model")
                return 0
        else:
            print("Model file not found")
            return 0
            
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    prediction_count = debug_prediction_process()
    print(f"\n=== FINAL RESULT: {prediction_count} predictions generated ===")
