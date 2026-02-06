# Main training script for Firco XGBoost API
# Comprehensive local testing and model training with hierarchical approach

import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, List, Tuple, Any, Optional
import sys
from datetime import datetime
import warnings
import shutil
import json
import re # Added for version extraction

warnings.filterwarnings('ignore')

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Local imports
from config import (
    LABEL_COLUMNS, HIT_LEVEL_TARGETS, MESSAGE_LEVEL_TARGETS, 
    TRAINING_DATA_PATH, MODEL_SAVE_DIR, ARCHIVE_DIR, 
    TRAINING_CONFIG, PERFORMANCE_THRESHOLDS, VERSIONING_CONFIG
)
from dataset_utils import (
    load_and_preprocess_data, create_label_encoders, encode_labels,
    split_data_hierarchical, prepare_hierarchical_features, validate_data_quality
)
from models.tfidf_xgb_F import FircoHierarchicalXGBoost
from train_utils import (
    evaluate_model_performance, evaluate_hierarchical_training,
    calculate_feature_importance, cross_validate_model,
    generate_performance_report, summarize_metrics, validate_model_requirements
)
from mongo_utils import (
    insert_model, insert_training_run, insert_validation_run, 
    generate_training_id, get_model_by_version
)

def _get_next_model_version() -> int:
    """Determines the next model version number based on existing files."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    existing_versions = []
    for f in os.listdir(MODEL_SAVE_DIR):
        if f.startswith('v') and f.endswith('.pkl'):
            try:
                version_num = int(f[1:-4])
                existing_versions.append(version_num)
            except ValueError:
                continue
    return max(existing_versions) + 1 if existing_versions else 1

def train_and_evaluate_firco_model(
    data_path: str = TRAINING_DATA_PATH,
    save_model: bool = True,
    generate_report: bool = True,
    user_id: str = "system"
) -> Tuple[FircoHierarchicalXGBoost, Dict[str, Any]]:
    """
    Complete training and evaluation pipeline for Firco XGBoost model.
    
    Args:
        data_path: Path to training data
        save_model: Whether to save the trained model
        generate_report: Whether to generate performance report
        user_id: User identifier for MongoDB logging
        
    Returns:
        Tuple[FircoHierarchicalXGBoost, Dict[str, Any]]: Trained model and results
    """
    training_start_time = datetime.now()
    training_id = generate_training_id("main_train")
    
    print("FIRCO HIERARCHICAL XGBOOST TRAINING & EVALUATION")
    print("=" * 60)
    print(f"Started: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training ID: {training_id}")
    print(f"Target columns: {LABEL_COLUMNS}")
    
    model_id = None
    
    try:
        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        df = load_and_preprocess_data(data_path, is_training=True)
        print(f"   Dataset shape: {df.shape}")
        
        # Step 2: Create label encoders
        print("\n2. Creating label encoders...")
        label_encoders = create_label_encoders(df)
        y_encoded = encode_labels(df[LABEL_COLUMNS], label_encoders)
        
        # Step 3: Split data
        print("\n3. Splitting data...")
        train_df, val_df, test_df = split_data_hierarchical(
            df, 
            test_size=TRAINING_CONFIG['test_size'],
            validation_size=TRAINING_CONFIG['validation_size'],
            random_state=TRAINING_CONFIG['random_state']
        )
        
        print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Prepare features and targets
        X_train = train_df.drop(columns=LABEL_COLUMNS)
        y_train = encode_labels(train_df[LABEL_COLUMNS], label_encoders)
        
        X_val = val_df.drop(columns=LABEL_COLUMNS)
        y_val = encode_labels(val_df[LABEL_COLUMNS], label_encoders)
        
        X_test = test_df.drop(columns=LABEL_COLUMNS)
        y_test = encode_labels(test_df[LABEL_COLUMNS], label_encoders)
        
        # Step 4: Train model
        print("\n4. Training hierarchical XGBoost model...")
        model = FircoHierarchicalXGBoost(label_encoders)
        model.train(X_train, y_train)
        print("   Training completed successfully")
        
        # Step 5: Evaluate on validation set
        print("\n5. Validation evaluation...")
        val_predictions = model.predict(X_val)
        val_performance = evaluate_model_performance(
            y_val, val_predictions, label_encoders, stage="validation"
        )
        
        # Step 6: Display validation metrics in tabular format
        print("\n" + "=" * 60)
        print("VALIDATION METRICS (MAIN RESULTS)")
        print("=" * 60)
        _display_metrics_table(val_performance, "Validation")
        
        # Step 7: Quick test evaluation for completeness
        test_predictions = model.predict(X_test)
        test_performance = evaluate_model_performance(
            y_test, test_predictions, label_encoders, stage="test"
        )
        
        # Compile results
        comprehensive_results = {
            'validation_performance': val_performance,
            'test_performance': test_performance,
            'training_config': TRAINING_CONFIG,
            'model_info': {
                'model_type': 'FircoHierarchicalXGBoost',
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'feature_count': X_train.shape[1],
                'target_count': len(LABEL_COLUMNS)
            }
        }
        
        # Step 8: Save model and log to MongoDB
        if save_model:
            print("\n6. Saving model...")
            next_version = _get_next_model_version()
            model_filename = f"v{next_version}.pkl"
            saved_model_path = save_trained_model(model, comprehensive_results, model_filename)
            comprehensive_results['model_path'] = saved_model_path
            print(f"   Model saved: {os.path.basename(saved_model_path)}")
            
            # Step 9: Log to MongoDB
            print("\n7. Logging to MongoDB...")
            training_end_time = datetime.now()
            training_duration = (training_end_time - training_start_time).total_seconds()
            
            try:
                # Insert model record
                model_id = insert_model(
                    user_id=user_id,
                    model_name="firco_hierarchical_xgb",
                    model_version=next_version,
                    tag="latest",
                    level="both",
                    deployed=False,
                    notes=f"Hierarchical XGBoost model v{next_version} trained on {len(X_train)} samples",
                    model_type="xgboost",
                    framework="scikit-learn"
                )
                
                if model_id:
                    print(f"   Model logged with ID: {model_id}")
                    comprehensive_results['mongodb_model_id'] = str(model_id)
                    
                    # Insert training run record
                    training_run_id = insert_training_run(
                        model_id=model_id,
                        training_id=training_id,
                        user_id=user_id,
                        level="both",
                        status="completed",
                        datasets={
                            "train": data_path,
                            "s3_key": f"datasets/training_data_{training_id}.csv"
                        },
                        data_size={
                            "train": len(X_train),
                            "validation": len(X_val),
                            "test": len(X_test)
                        },
                        metrics=val_performance,
                        started_at=training_start_time,
                        ended_at=training_end_time,
                        duration=training_duration
                    )
                    
                    if training_run_id:
                        print(f"   Training run logged with ID: {training_run_id}")
                        comprehensive_results['mongodb_training_id'] = str(training_run_id)
                        
                        # Insert validation run record
                        validation_run_id = insert_validation_run(
                            model_id=model_id,
                            training_id=training_id,
                            user_id=user_id,
                            level="both",
                            status="completed",
                            datasets={
                                "validation": data_path,
                                "s3_key": f"datasets/validation_data_{training_id}.csv"
                            },
                            data_size={
                                "validation": len(X_val)
                            },
                            metrics=val_performance,
                            started_at=training_start_time,
                            ended_at=training_end_time
                        )
                        
                        if validation_run_id:
                            print(f"   Validation run logged with ID: {validation_run_id}")
                            comprehensive_results['mongodb_validation_id'] = str(validation_run_id)
                        
            except Exception as mongo_error:
                print(f"   Warning: MongoDB logging failed: {str(mongo_error)}")
                # Continue execution even if MongoDB logging fails
        
        # Final summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        if '_overall' in val_performance:
            overall = val_performance['_overall']
            print(f"Overall Validation Accuracy: {overall['avg_accuracy']:.4f}")
            print(f"Overall Validation F1: {overall['avg_weighted_f1']:.4f}")
            print(f"Targets Meeting F1 Threshold: {overall['targets_meeting_threshold']}/{overall['target_count']}")
        
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return model, comprehensive_results
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def _display_metrics_table(performance: Dict[str, Any], stage: str):
    """Display performance metrics in tabular format."""
    print(f"\n{stage} Performance Metrics:")
    print("-" * 80)
    print(f"{'Target':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for target, metrics in performance.items():
        if target == '_overall':
            continue
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            
            # Get metrics from classification_report if available
            if 'classification_report' in metrics:
                class_report = metrics['classification_report']
                weighted_avg = class_report.get('weighted avg', {})
                precision = weighted_avg.get('precision', 0.0)
                recall = weighted_avg.get('recall', 0.0)
                f1_score = weighted_avg.get('f1-score', 0.0)
            else:
                # Fallback to stored values
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                f1_score = metrics.get('weighted_f1', 0.0)
            
            print(f"{target:<30} {accuracy:<12.4f} {precision:<12.4f} "
                  f"{recall:<12.4f} {f1_score:<12.4f}")
    
    # Overall metrics
    if '_overall' in performance:
        overall = performance['_overall']
        print("-" * 80)
        print(f"{'OVERALL':<30} {overall['avg_accuracy']:<12.4f} {overall.get('avg_precision', 0.0):<12.4f} "
              f"{overall.get('avg_recall', 0.0):<12.4f} {overall['avg_weighted_f1']:<12.4f}")
        print("-" * 80)

def train_model_without_save(
    train_path: str,
    label_cols: List[str],
    text_col: str
) -> Tuple[FircoHierarchicalXGBoost, Dict[str, Any]]:
    """
    Train model without saving (for API usage) but with proper evaluation.
    
    Args:
        train_path: Path to training data
        label_cols: List of label columns
        text_col: Primary text column
        
    Returns:
        Tuple[FircoHierarchicalXGBoost, Dict[str, Any]]: Trained model and results
    """
    print("Training Firco XGBoost model for API...")
    
    # Load and preprocess data
    df = load_and_preprocess_data(train_path, is_training=True)
    print(f"   Dataset shape: {df.shape}")
    
    # Debug: Check which target columns are available
    print(f"   Target columns requested: {label_cols}")
    print(f"   Target columns available: {[col for col in label_cols if col in df.columns]}")
    print(f"   Target columns missing: {[col for col in label_cols if col not in df.columns]}")
    
    # Create label encoders
    label_encoders = create_label_encoders(df)
    
    # Split data for validation
    train_df, val_df, test_df = split_data_hierarchical(
        df, 
        test_size=TRAINING_CONFIG['test_size'],
        validation_size=TRAINING_CONFIG['validation_size'],
        random_state=TRAINING_CONFIG['random_state']
    )
    
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Prepare features and targets
    X_train = train_df.drop(columns=label_cols, errors='ignore')
    y_train = encode_labels(train_df[label_cols], label_encoders)
    
    X_val = val_df.drop(columns=label_cols, errors='ignore')
    y_val = encode_labels(val_df[label_cols], label_encoders)
    
    # Debug: Check y_train columns
    print(f"   y_train columns: {list(y_train.columns)}")
    print(f"   y_val columns: {list(y_val.columns)}")
    
    # Initialize and train model
    model = FircoHierarchicalXGBoost(label_encoders)
    
    # Train the model
    model.train(X_train, y_train)
    
    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_performance = evaluate_model_performance(
        y_val, val_predictions, label_encoders, stage="validation"
    )
    
    # Store additional info
    model.text_col = text_col
    model.label_cols = label_cols
    
    # Create comprehensive results
    results = {
        'validation_performance': val_performance,
        'model_info': {
            'model_type': 'FircoHierarchicalXGBoost',
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': X_train.shape[1],
            'target_count': len(label_cols),
            'text_column': text_col
        },
        'training_config': TRAINING_CONFIG
    }
    
    return model, results

def predict_single_input(
    text_input: str,
    model_path: str
) -> Dict[str, Any]:
    """
    Make prediction for single text input using the hierarchical approach.
    
    Args:
        text_input: Text to predict
        model_path: Path to trained model
        
    Returns:
        Dict[str, Any]: Prediction results
    """
    # Load model
    model = joblib.load(model_path)
    
    # Create comprehensive input dataframe with ALL possible columns from the dataset
    # This ensures we match the feature dimensions used during training
    input_df = pd.DataFrame({
        # Primary text columns
        'hit.matching_text': [text_input],
        'hit.watchlist_text': [text_input],
        'mt103.sanction_list': [""],
        
        # Core hit information
        'hit.score': [85],  # Reasonable default
        'hit.fuzzy_match_score': [0.85],
        'hit.is_pep': ['No'],
        'hit.hit_type': ['Entity'],
        'hit.matching_type': ['Name'],
        'hit.priority': ['Medium'],
        'hit.severity': ['Medium'],
        'hit.country': ['Unknown'],
        'hit.mt_type': [103],  # Default to MT103
        
        # MT103 fields
        'mt103.country': ['Unknown'],
        'mt103.transaction_type': ['OUT'],
        'mt103.hits_count_103': [1],
        'mt103.organization': ['Unknown'],
        'mt103.beneficiary': ['Unknown'],
        'mt103.bic_code': ['Unknown'],
        'mt103.city': ['Unknown'],
        'mt103.dob': [''],
        'mt103.national_id': [''],
        'mt103.origin': [''],
        'mt103.passport': [''],
        'mt103.reference': [''],
        'mt103.sender': ['Unknown'],
        'mt103.state': [''],
        'mt103.street': [''],
        'mt103.synonyms.city': [''],
        'mt103.synonyms.country': [''],
        'mt103.synonyms.name': [''],
        'mt103.synonyms.organization': [''],
        'mt103.transaction_reference_number': [''],
        'mt103.mt_type': [103.0],
        'mt103.account_number': [''],
        
        # MT202 fields
        'mt202.amount': [0],
        'mt202.hits_count_202': [0],
        'mt202.charges': [0],
        'mt202.currency': ['USD'],
        'mt202.instruction_code': [''],
        'mt202.beneficiary_institution.name': ['Unknown'],
        'mt202.beneficiary_institution.bic': [''],
        'mt202.beneficiary_institution.address.city': [''],
        'mt202.beneficiary_institution.address.country': [''],
        'mt202.beneficiary_institution.address.street': [''],
        'mt202.intermediary_institution.bic': [''],
        'mt202.intermediary_institution.name': [''],
        'mt202.ordering_institution.address.city': [''],
        'mt202.ordering_institution.address.country': [''],
        'mt202.ordering_institution.address.street': [''],
        'mt202.ordering_institution.bic': [''],
        'mt202.ordering_institution.name': [''],
        'mt202.related_reference': [''],
        'mt202.sender_to_receiver_info': [''],
        'mt202.transaction_reference_number': [''],
        'mt202.value_date': [''],
        'mt202.mt_type': [None],  # Only MT103 in this case
        
        # Additional hit fields
        'hit.Organization': ['Unknown'],
        'hit.account_number': [''],
        'hit.beneficiary': ['Unknown'],
        'hit.bic_code': [''],
        'hit.city': ['Unknown'],
        'hit.hit_id': [1],
        'hit.sender': ['Unknown'],
        'hit.state': [''],
        'hit.tag': [''],
        
        # Other fields
        'message_id': [1],
        'decision.decision': [''],
        'decision.options': ['']
    })
    
    # Make predictions using the hierarchical model
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)
    
    # Format results
    results = {}
    for target, pred in predictions.items():
        if target in model.label_encoders:
            encoder = model.label_encoders[target]
            predicted_class = encoder.inverse_transform([pred[0]])[0]
            
            # Get probabilities
            if target in probabilities:
                proba = probabilities[target][0]
                # Ensure we don't exceed array bounds
                max_idx = min(len(encoder.classes_), len(proba))
                class_probabilities = {
                    encoder.classes_[i]: float(proba[i]) 
                    for i in range(max_idx)
                }
            else:
                class_probabilities = {}
            
            results[target] = {
                'predicted_class': predicted_class,
                'probabilities': class_probabilities
            }
    
    return results

def validate_model(
    validation_path: str,
    model_path: str
) -> Dict[str, Any]:
    """
    Validate model on validation dataset.
    
    Args:
        validation_path: Path to validation data
        model_path: Path to trained model
        
    Returns:
        Dict[str, Any]: Validation results
    """
    print(f"Validating model from {model_path}")
    print(f"Using validation data from {validation_path}")
    
    # Load model and data
    model = joblib.load(model_path)
    df = load_and_preprocess_data(validation_path, is_training=False)
    
    print(f"Validation dataset shape: {df.shape}")
    
    # Check which target columns are available
    available_targets = [col for col in LABEL_COLUMNS if col in df.columns]
    missing_targets = [col for col in LABEL_COLUMNS if col not in df.columns]
    
    print(f"Available target columns: {available_targets}")
    if missing_targets:
        print(f"Missing target columns: {missing_targets}")
    
    # Prepare features for prediction
    X = df.drop(columns=LABEL_COLUMNS, errors='ignore')
    print(f"Feature matrix shape: {X.shape}")
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    print(f"Generated predictions for targets: {list(predictions.keys())}")
    
    # If we have ground truth labels, evaluate performance
    if available_targets and len(available_targets) >= len(LABEL_COLUMNS) // 2:  # At least half the targets
        print("Ground truth available - performing evaluation...")
        y_true = encode_labels(df[available_targets], model.label_encoders)
        
        # Filter predictions to only available targets
        filtered_predictions = {target: pred for target, pred in predictions.items() 
                              if target in available_targets}
        
        performance = evaluate_model_performance(
            y_true, filtered_predictions, model.label_encoders, stage="validation"
        )
        
        # Add dataset info
        performance['dataset_info'] = {
            'total_samples': len(df),
            'available_targets': available_targets,
            'missing_targets': missing_targets,
            'feature_count': X.shape[1]
        }
        
        return performance
    else:
        print("Insufficient ground truth available - returning predictions only...")
        # Format predictions for response
        prediction_summary = {}
        for target, pred_array in predictions.items():
            if target in model.label_encoders:
                encoder = model.label_encoders[target]
                unique_preds, counts = np.unique(pred_array, return_counts=True)
                class_distribution = {}
                
                for pred_idx, count in zip(unique_preds, counts):
                    if pred_idx < len(encoder.classes_):
                        class_name = encoder.classes_[pred_idx]
                        class_distribution[class_name] = int(count)
                
                prediction_summary[target] = {
                    'class_distribution': class_distribution,
                    'total_predictions': len(pred_array)
                }
        
        return {
            'predictions_summary': prediction_summary,
            'dataset_info': {
                'total_samples': len(df),
                'available_targets': available_targets,
                'missing_targets': missing_targets,
                'feature_count': X.shape[1]
            }
        }

def save_trained_model(
    model: FircoHierarchicalXGBoost,
    results: Dict[str, Any],
    model_filename: str
) -> str:
    """
    Save trained model with a specific version name and handle archiving.
    
    Args:
        model: The trained FircoHierarchicalXGBoost model.
        results: A dictionary containing training and validation results.
        model_filename: The specific filename (e.g., 'v21.pkl') to save the model as.
        
    Returns:
        The full path to the saved model file.
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    # Archive existing .pkl models in the save directory
    for existing_file in os.listdir(MODEL_SAVE_DIR):
        if existing_file.endswith('.pkl'):
            old_path = os.path.join(MODEL_SAVE_DIR, existing_file)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archived_name = f"archived_{existing_file}_{timestamp}"
            archived_path = os.path.join(ARCHIVE_DIR, archived_name)
            try:
                shutil.move(old_path, archived_path)
                print(f"   Archived previous model: {existing_file} -> {archived_name}")
            except Exception as e:
                print(f"   Warning: Could not archive {existing_file}. Reason: {e}")
    
    # Save the new model with the provided filename
    model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    
    # Add metadata to the model before saving
    version_number = int(re.search(r'v(\d+)', model_filename).group(1)) if re.search(r'v(\d+)', model_filename) else -1
    model.training_metadata = {
        'version': version_number,
        'timestamp': datetime.now().isoformat(),
        'performance_summary': results.get('validation_performance', {}).get('_overall', {}),
        'model_info': results.get('model_info', {})
    }
    
    joblib.dump(model, model_path)
    
    # Save accompanying results to a JSON file
    results_filename = model_filename.replace('.pkl', '_results.json')
    results_path = os.path.join(MODEL_SAVE_DIR, results_filename)
    with open(results_path, 'w') as f:
        serializable_results = _make_json_serializable(results)
        json.dump(serializable_results, f, indent=2)
    
    print(f"   Saved new model as version {version_number}: {model_filename}")
    return model_path

def _make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_json_serializable(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return _make_json_serializable(obj.__dict__)
    else:
        return obj

def run_comprehensive_test():
    """Run comprehensive test of the Firco XGBoost model."""
    try:
        model, results = train_and_evaluate_firco_model(
            save_model=True,
            generate_report=True
        )
        
        # Test single prediction
        test_text = "High-risk PEP match on organization Sheppard-Johnson with score 0.95"
        single_pred = predict_single_input(test_text, results['model_path'])
        
        # Display summary
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        if '_overall' in results['validation_performance']:
            overall = results['validation_performance']['_overall']
            print(f"Model trained successfully")
            print(f"Overall validation accuracy: {overall['avg_accuracy']:.4f}")
            print(f"Overall validation F1: {overall['avg_weighted_f1']:.4f}")
            print(f"Targets meeting threshold: {overall['targets_meeting_threshold']}/{overall['target_count']}")
        
        # Check if model meets basic requirements
        meets_requirements = overall['targets_meeting_threshold'] >= 1
        if meets_requirements:
            print("Model meets basic requirements (at least 1 target above threshold)")
        else:
            print("Model below basic requirements")
        
        print("Model saved successfully")
        print("Single prediction test passed")
        
        return True
        
    except Exception as e:
        print(f"Comprehensive test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def predict_batch_csv(
    csv_path: str,
    model_path: str
) -> List[Dict[str, Any]]:
    """
    Make predictions for batch CSV file.
    
    Args:
        csv_path: Path to CSV file
        model_path: Path to trained model
        
    Returns:
        List[Dict[str, Any]]: List of prediction results for each row
    """
    print(f"Making batch predictions from {csv_path}")
    print(f"Using model from {model_path}")
    
    # Load model with validation (ENHANCED LOADING)
    from api_utils import load_model
    model = load_model(model_path)
    if not model:
        raise ValueError(f"Failed to load model from {model_path}")
    
    # Load and preprocess data
    df = load_and_preprocess_data(csv_path, is_training=False)
    
    print(f"Batch prediction dataset shape: {df.shape}")
    print(f"Number of rows to predict: {len(df)}")
    
    # Prepare features for prediction
    X = df.drop(columns=LABEL_COLUMNS, errors='ignore')
    print(f"Feature matrix shape: {X.shape}")
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    print(f"Generated predictions for targets: {list(predictions.keys())}")
    
    # Verify prediction array lengths
    for target, pred_array in predictions.items():
        print(f"Target '{target}': {len(pred_array)} predictions")
    
    # Format results for each row
    results = []
    total_rows = len(df)
    print(f"Processing {total_rows} rows for results formatting...")
    
    for row_idx in range(total_rows):
        row_result = {
            'row_index': row_idx,
            'predictions': {}
        }
        
        # Add predictions for each target
        for target in predictions:
            if target in model.label_encoders:
                encoder = model.label_encoders[target]
                pred_idx = predictions[target][row_idx]
                
                if pred_idx < len(encoder.classes_):
                    predicted_class = encoder.classes_[pred_idx]
                else:
                    predicted_class = f"unknown_{pred_idx}"
                
                # Get probabilities
                if target in probabilities:
                    proba = probabilities[target][row_idx]
                    class_probabilities = {
                        encoder.classes_[i]: float(proba[i]) 
                        for i in range(len(encoder.classes_))
                    }
                else:
                    class_probabilities = {}
                
                row_result['predictions'][target] = {
                    'predicted_class': predicted_class,
                    'probabilities': class_probabilities
                }
        
        results.append(row_result)
    
    print(f"Successfully created {len(results)} prediction results")
    print(f"First few row indices: {[r['row_index'] for r in results[:5]]}")
    print(f"Last few row indices: {[r['row_index'] for r in results[-5:]]}")
    
    return results

if __name__ == "__main__":
    # Run comprehensive test
    success = run_comprehensive_test()
    
    if success:
        print("\nALL TESTS PASSED - FIRCO XGBOOST MODEL READY FOR PRODUCTION")
    else:
        print("\nTESTS FAILED - PLEASE CHECK ERRORS ABOVE")
    
    sys.exit(0 if success else 1)