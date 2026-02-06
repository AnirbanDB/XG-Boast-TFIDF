#!/usr/bin/env python3
"""
Debug script to investigate the TF-IDF vectorizer issue
"""

import os
import sys
import joblib
import pandas as pd

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from api_utils import get_latest_model_path, load_model
from dataset_utils import load_and_preprocess_data
from config import LABEL_COLUMNS

def debug_vectorizer_issue():
    """Debug the TF-IDF vectorizer issue in detail."""
    
    # Get model and test data paths
    model_path = get_latest_model_path()
    test_csv_path = os.path.join(current_dir, "splits", "test_set.csv")
    
    print("🔍 DEBUGGING TF-IDF VECTORIZER ISSUE")
    print("=" * 60)
    
    # Load model
    print("1. Loading model...")
    model = load_model(model_path)
    if not model:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully")
    
    # Check model structure
    print("\n2. Checking model structure...")
    print(f"   - Model type: {type(model)}")
    print(f"   - Has feature_transformer: {hasattr(model, 'feature_transformer')}")
    print(f"   - Transformer type: {type(model.feature_transformer) if hasattr(model, 'feature_transformer') else 'None'}")
    print(f"   - Transformer fitted: {model.feature_transformer.is_fitted if hasattr(model, 'feature_transformer') else 'N/A'}")
    
    # Check vectorizers in detail
    print("\n3. Checking text vectorizers...")
    if hasattr(model, 'feature_transformer') and hasattr(model.feature_transformer, 'text_vectorizers'):
        vectorizers = model.feature_transformer.text_vectorizers
        print(f"   - Number of vectorizers: {len(vectorizers)}")
        
        for col_name, vectorizer in vectorizers.items():
            print(f"\n   Vectorizer for '{col_name}':")
            print(f"     - Type: {type(vectorizer)}")
            print(f"     - Has vocabulary_: {hasattr(vectorizer, 'vocabulary_')}")
            print(f"     - Vocabulary_ is None: {getattr(vectorizer, 'vocabulary_', None) is None}")
            print(f"     - Has idf_: {hasattr(vectorizer, 'idf_')}")
            print(f"     - IDF_ is None: {getattr(vectorizer, 'idf_', None) is None}")
            
            # Try to check the internal _tfidf attribute
            if hasattr(vectorizer, '_tfidf'):
                tfidf = vectorizer._tfidf
                print(f"     - Has _tfidf: True")
                print(f"     - _tfidf type: {type(tfidf)}")
                print(f"     - _tfidf has idf_: {hasattr(tfidf, 'idf_')}")
                print(f"     - _tfidf idf_ is None: {getattr(tfidf, 'idf_', None) is None}")
            else:
                print(f"     - Has _tfidf: False")
    else:
        print("   ❌ No text vectorizers found")
        return False
    
    # Try to load test data and see what happens
    print("\n4. Loading test data...")
    df = load_and_preprocess_data(test_csv_path, is_training=False)
    print(f"   - Dataset shape: {df.shape}")
    
    # Try to prepare stage 1 features like the model does
    print("\n5. Testing feature preparation...")
    try:
        X = df.drop(columns=LABEL_COLUMNS, errors='ignore')
        print(f"   - X shape: {X.shape}")
        
        # This is what happens in model.predict()
        X_stage1 = model._prepare_stage1_features_for_prediction(X)
        print(f"   - X_stage1 shape: {X_stage1.shape}")
        
        # Check if the text columns exist
        vectorizer_cols = list(model.feature_transformer.text_vectorizers.keys())
        print(f"   - Vectorizer columns: {vectorizer_cols}")
        
        for col in vectorizer_cols:
            if col in X_stage1.columns:
                print(f"   - Column '{col}' exists in X_stage1")
                sample_data = X_stage1[col].iloc[:5].tolist()
                print(f"     Sample data: {sample_data}")
            else:
                print(f"   - Column '{col}' MISSING from X_stage1")
        
    except Exception as e:
        print(f"   ❌ Error in feature preparation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try to transform step by step
    print("\n6. Testing transform step by step...")
    try:
        # Test the first vectorizer manually
        first_col = list(model.feature_transformer.text_vectorizers.keys())[0]
        first_vectorizer = model.feature_transformer.text_vectorizers[first_col]
        
        print(f"   - Testing vectorizer for '{first_col}'")
        
        if first_col in X_stage1.columns:
            text_data = X_stage1[first_col].fillna("").astype(str)
            print(f"   - Text data shape: {text_data.shape}")
            print(f"   - Sample text: {text_data.iloc[0]}")
            
            # Try the transform
            print("   - Attempting transform...")
            result = first_vectorizer.transform(text_data)
            print(f"   ✅ Transform successful! Result shape: {result.shape}")
            
        else:
            print(f"   ❌ Column '{first_col}' not found in data")
            
    except Exception as e:
        print(f"   ❌ Transform failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's check if we can fix the vectorizer
        print("\n7. Attempting to diagnose vectorizer state...")
        
        if hasattr(first_vectorizer, '_tfidf'):
            tfidf = first_vectorizer._tfidf
            if hasattr(tfidf, 'idf_') and tfidf.idf_ is None:
                print("   - TF-IDF idf_ is None - this is the problem!")
                print("   - This means the vectorizer was not properly saved/loaded")
            elif not hasattr(tfidf, 'idf_'):
                print("   - TF-IDF doesn't have idf_ attribute")
        
        return False
    
    print("\n✅ DEBUG COMPLETED SUCCESSFULLY")
    return True

if __name__ == "__main__":
    debug_vectorizer_issue()
