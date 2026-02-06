# Example Usage of Firco Base Model Architecture
# Demonstrates how to use the new professional architecture with factory pattern

import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models import ModelFactory, FircoHierarchicalXGBoost


def demonstrate_base_model_architecture():
    """
    Demonstrates the new base model architecture and factory pattern.
    """
    print("🚀 Firco Base Model Architecture Demo")
    print("=" * 50)
    
    # Method 1: Create model using factory pattern
    print("\n1. Creating model using Factory Pattern:")
    model_factory = ModelFactory.create_model('firco_hierarchical_xgb')
    print(f"   ✅ Created: {model_factory.__class__.__name__}")
    
    # Method 2: Create model directly
    print("\n2. Creating model directly:")
    direct_config = {
        'model_id': 'Direct_Model',
        'model_type': 'hierarchical_xgboost', 
        'version': '1.1'
    }
    direct_model = FircoHierarchicalXGBoost(direct_config)
    print(f"   ✅ Created: {direct_model.__class__.__name__}")
    
    # Get model information
    print("\n3. Model Information:")
    info = direct_model.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Demonstrate abstract method enforcement
    print("\n4. Abstract Base Class Benefits:")
    print("   ✅ Standardized interface across all models")
    print("   ✅ Enforced implementation of core methods")
    print("   ✅ Consistent logging and error handling")
    print("   ✅ Factory pattern for easy model instantiation")
    print("   ✅ Professional software engineering practices")
    
    # Show available model types
    print("\n5. Available Model Types in Factory:")
    available_models = ModelFactory.get_available_models()
    for model_type in available_models:
        print(f"   • {model_type}")
    
    # Example data structure (for demonstration)
    print("\n6. Expected Data Structure:")
    sample_data = {
        'Customer_Name': ['John Doe', 'Jane Smith'],
        'Transaction_Description': ['Payment for services', 'Wire transfer'],
        'Transaction_Amount': [1000.0, 5000.0],
        'Customer_Risk_Score': [0.3, 0.7],
        'Transaction_Type': ['Payment', 'Wire'],
        'Customer_Country': ['USA', 'UK'],
        'Alert_Type': ['High Risk', 'Medium Risk'],
        'Additional_Info': ['Standard transaction', 'Large amount']
    }
    
    sample_df = pd.DataFrame(sample_data)
    print("   Sample DataFrame structure:")
    print(sample_df.head())
    
    print("\n✨ Base Model Architecture Implementation Complete!")
    print("   Now your codebase follows professional software engineering practices")
    print("   similar to the prediqai-Deep architecture you admired.")


if __name__ == "__main__":
    demonstrate_base_model_architecture()
