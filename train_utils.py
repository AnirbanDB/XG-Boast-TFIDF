# Training utilities for Firco XGBoost API
# Comprehensive evaluation and training support with hierarchical training

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Local imports
from config import (
    LABEL_COLUMNS, HIT_LEVEL_TARGETS, MESSAGE_LEVEL_TARGETS,
    PERFORMANCE_THRESHOLDS, HIERARCHICAL_TRAINING_CONFIG
)

def _convert_numpy_to_python(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_to_python(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return _convert_numpy_to_python(obj.__dict__)
    else:
        return obj

def evaluate_model_performance(
    y_true: pd.DataFrame,
    y_pred: Dict[str, np.ndarray],
    label_encoders: Dict[str, LabelEncoder],
    stage: str = "validation"
) -> Dict[str, Any]:
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        y_true: True labels (encoded)
        y_pred: Predicted labels by target
        label_encoders: Label encoders for decoding
        stage: Evaluation stage name
        
    Returns:
        Dict[str, Any]: Performance metrics for each target
    """
    performance_results = {}
    
    # Individual target performance
    accuracies = []
    weighted_f1_scores = []
    targets_meeting_threshold = 0
    
    for target_col in LABEL_COLUMNS:
        if target_col in y_true.columns and target_col in y_pred:
            try:
                y_true_target = y_true[target_col].values
                y_pred_target = y_pred[target_col]
                
                # Basic metrics
                accuracy = accuracy_score(y_true_target, y_pred_target)
                
                # Classification report
                class_report = classification_report(
                    y_true_target, y_pred_target,
                    output_dict=True,
                    zero_division=0
                )
                
                # Extract weighted F1 score
                weighted_f1 = class_report.get('weighted avg', {}).get('f1-score', 0.0)
                
                # Store results
                performance_results[target_col] = {
                    'accuracy': float(accuracy),
                    'weighted_f1': float(weighted_f1),
                    'classification_report': class_report
                }
                
                # Add to overall calculations
                accuracies.append(accuracy)
                weighted_f1_scores.append(weighted_f1)
                
                # Check if target meets F1 threshold
                f1_threshold = PERFORMANCE_THRESHOLDS.get('min_f1_score', 0.75)
                if weighted_f1 >= f1_threshold:
                    targets_meeting_threshold += 1
                
            except Exception as e:
                print(f"  Error evaluating {target_col}: {str(e)}")
                performance_results[target_col] = {
                    'accuracy': 0.0,
                    'weighted_f1': 0.0,
                    'error': str(e)
                }
        else:
            print(f"  Warning: Target {target_col} not found in predictions or true labels")
            performance_results[target_col] = {
                'accuracy': 0.0,
                'weighted_f1': 0.0,
                'error': 'Target not found'
            }
    
    # Overall metrics
    if accuracies and weighted_f1_scores:
        overall_metrics = {
            'avg_accuracy': float(np.mean(accuracies)),
            'avg_weighted_f1': float(np.mean(weighted_f1_scores)),
            'targets_meeting_threshold': int(targets_meeting_threshold),
            'target_count': len(LABEL_COLUMNS),
            'stage': stage
        }
    else:
        overall_metrics = {
            'avg_accuracy': 0.0,
            'avg_weighted_f1': 0.0,
            'targets_meeting_threshold': 0,
            'target_count': len(LABEL_COLUMNS),
            'stage': stage,
            'error': 'No valid predictions found'
        }
    
    performance_results['_overall'] = overall_metrics
    
    # Convert all numpy types to Python native types for JSON serialization
    performance_results = _convert_numpy_to_python(performance_results)
    
    return performance_results

def _calculate_comprehensive_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_name: str
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a single target.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_name: Name of the target column
        
    Returns:
        Dict[str, Any]: Comprehensive metrics dictionary
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Class-wise metrics
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    class_metrics = {}
    
    for class_name in unique_classes:
        class_metrics[class_name] = {
            'precision': precision_score(y_true, y_pred, labels=[class_name], average='micro', zero_division=0),
            'recall': recall_score(y_true, y_pred, labels=[class_name], average='micro', zero_division=0),
            'f1': f1_score(y_true, y_pred, labels=[class_name], average='micro', zero_division=0),
            'support': np.sum(y_true == class_name)
        }
    
    # Compile comprehensive metrics
    metrics = {
        'accuracy': float(accuracy),
        'classification_report': _convert_numpy_to_python(report),
        'confusion_matrix': cm.tolist(),
        'class_metrics': _convert_numpy_to_python(class_metrics),
        'macro_avg': _convert_numpy_to_python(report.get('macro avg', {})),
        'weighted_avg': _convert_numpy_to_python(report.get('weighted avg', {})),
        'target_name': str(target_name),
        'unique_classes': [str(cls) for cls in unique_classes],
        'class_distribution': {str(cls): int(np.sum(y_true == cls)) for cls in unique_classes}
    }
    
    return metrics

def _calculate_overall_metrics(performance_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate overall metrics across all targets.
    
    Args:
        performance_results: Dictionary of performance results for each target
        
    Returns:
        Dict[str, Any]: Overall metrics summary
    """
    target_results = {k: v for k, v in performance_results.items() if not k.startswith('_')}
    
    if not target_results:
        return {'avg_accuracy': 0.0, 'avg_weighted_f1': 0.0, 'avg_macro_f1': 0.0}
    
    accuracies = [metrics['accuracy'] for metrics in target_results.values()]
    weighted_f1s = [metrics['weighted_avg'].get('f1-score', 0) for metrics in target_results.values()]
    macro_f1s = [metrics['macro_avg'].get('f1-score', 0) for metrics in target_results.values()]
    
    overall_metrics = {
        'avg_accuracy': float(np.mean(accuracies)),
        'avg_weighted_f1': float(np.mean(weighted_f1s)),
        'avg_macro_f1': float(np.mean(macro_f1s)),
        'std_accuracy': float(np.std(accuracies)),
        'std_weighted_f1': float(np.std(weighted_f1s)),
        'std_macro_f1': float(np.std(macro_f1s)),
        'target_count': int(len(target_results)),
        'targets_meeting_threshold': int(sum(1 for f1 in weighted_f1s if f1 >= PERFORMANCE_THRESHOLDS['target_f1_score']))
    }
    
    return overall_metrics

def evaluate_hierarchical_training(
    hit_level_results: Dict[str, Dict[str, Any]],
    message_level_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate the effectiveness of hierarchical training approach.
    
    Args:
        hit_level_results: Performance results for hit-level targets
        message_level_results: Performance results for message-level targets
        
    Returns:
        Dict[str, Any]: Hierarchical training evaluation results
    """
    print("\n=== Hierarchical Training Evaluation ===")
    
    # Extract hit-level metrics
    hit_metrics = {}
    for target in HIT_LEVEL_TARGETS:
        if target in hit_level_results:
            hit_metrics[target] = {
                'accuracy': hit_level_results[target]['accuracy'],
                'weighted_f1': hit_level_results[target]['weighted_avg'].get('f1-score', 0),
                'macro_f1': hit_level_results[target]['macro_avg'].get('f1-score', 0)
            }
    
    # Extract message-level metrics
    message_metrics = {}
    for target in MESSAGE_LEVEL_TARGETS:
        if target in message_level_results:
            message_metrics[target] = {
                'accuracy': message_level_results[target]['accuracy'],
                'weighted_f1': message_level_results[target]['weighted_avg'].get('f1-score', 0),
                'macro_f1': message_level_results[target]['macro_avg'].get('f1-score', 0)
            }
    
    # Calculate hierarchical metrics
    hierarchical_evaluation = {
        'hit_level_performance': hit_metrics,
        'message_level_performance': message_metrics,
        'hit_level_avg_f1': np.mean([m['weighted_f1'] for m in hit_metrics.values()]) if hit_metrics else 0,
        'message_level_avg_f1': np.mean([m['weighted_f1'] for m in message_metrics.values()]) if message_metrics else 0,
        'overall_avg_f1': np.mean([m['weighted_f1'] for m in {**hit_metrics, **message_metrics}.values()]),
        'hierarchy_effectiveness': 'effective' if len(hit_metrics) > 0 and len(message_metrics) > 0 else 'limited'
    }
    
    print(f"Hit-level average F1: {hierarchical_evaluation['hit_level_avg_f1']:.4f}")
    print(f"Message-level average F1: {hierarchical_evaluation['message_level_avg_f1']:.4f}")
    print(f"Overall average F1: {hierarchical_evaluation['overall_avg_f1']:.4f}")
    
    return hierarchical_evaluation

def calculate_feature_importance(model, top_n: int = 20) -> Dict[str, Any]:
    """
    Calculate and return feature importance from XGBoost models.
    
    Args:
        model: Trained model object
        top_n: Number of top features to return
        
    Returns:
        Dict[str, Any]: Feature importance results
    """
    print("Calculating feature importance...")
    
    feature_importance_results = {}
    
    # Handle both old structure (models) and new structure (hit_level_models, message_level_models)
    models_dict = {}
    
    if hasattr(model, 'models') and isinstance(model.models, dict):
        models_dict = model.models
        print("Found 'models' attribute")
    elif hasattr(model, 'hit_level_models') and hasattr(model, 'message_level_models'):
        # Combine hit_level and message_level models
        if model.hit_level_models:
            models_dict.update(model.hit_level_models)
        if model.message_level_models:
            models_dict.update(model.message_level_models)
        print(f"Found hit_level_models and message_level_models: {list(models_dict.keys())}")
    
    if models_dict:
        for target, xgb_model in models_dict.items():
            if hasattr(xgb_model, 'feature_importances_'):
                # Get feature importance
                importances = xgb_model.feature_importances_
                
                # Try to get feature names
                feature_names = None
                if hasattr(model, 'preprocessor') and hasattr(model.preprocessor, 'get_feature_names_out'):
                    try:
                        feature_names = model.preprocessor.get_feature_names_out()
                    except:
                        pass
                
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Get top N features
                top_features = importance_df.head(top_n)
                
                feature_importance_results[target] = {
                    'top_features': top_features.to_dict('records'),
                    'total_features': len(importances),
                    'importance_sum': float(np.sum(importances)),
                    'top_n_importance_sum': float(top_features['importance'].sum())
                }
                
                print(f"  {target}: Top feature - {top_features.iloc[0]['feature']} ({top_features.iloc[0]['importance']:.4f})")
    else:
        print("No models found in expected attributes")
    
    return feature_importance_results

def cross_validate_model(model, X, y, cv_folds: int = 5) -> Dict[str, Any]:
    """
    Perform cross-validation on the model.
    
    Args:
        model: Model to cross-validate
        X: Features
        y: Labels
        cv_folds: Number of CV folds
        
    Returns:
        Dict[str, Any]: Cross-validation results
    """
    print(f"Performing {cv_folds}-fold cross-validation...")
    
    cv_results = {}
    
    if hasattr(model, 'models') and isinstance(model.models, dict):
        for target, target_model in model.models.items():
            if target in y.columns:
                y_target = y[target]
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    target_model, X, y_target, 
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring='f1_weighted'
                )
                
                cv_results[target] = {
                    'cv_scores': cv_scores.tolist(),
                    'mean_score': float(np.mean(cv_scores)),
                    'std_score': float(np.std(cv_scores)),
                    'min_score': float(np.min(cv_scores)),
                    'max_score': float(np.max(cv_scores))
                }
                
                print(f"  {target}: CV F1 = {cv_results[target]['mean_score']:.4f} ± {cv_results[target]['std_score']:.4f}")
    
    return cv_results

def generate_performance_report(
    performance_results: Dict[str, Dict[str, Any]],
    feature_importance: Optional[Dict[str, Any]] = None,
    cv_results: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive performance report.
    
    Args:
        performance_results: Model performance results
        feature_importance: Feature importance results
        cv_results: Cross-validation results
        save_path: Path to save the report
        
    Returns:
        str: Formatted performance report
    """
    print("Generating comprehensive performance report...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FIRCO XGBOOST COMPLIANCE PREDICTOR - PERFORMANCE REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overall metrics
    if '_overall' in performance_results:
        overall = performance_results['_overall']
        report_lines.append("OVERALL PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Average Accuracy: {overall['avg_accuracy']:.4f}")
        report_lines.append(f"Average Weighted F1: {overall['avg_weighted_f1']:.4f}")
        report_lines.append(f"Average Macro F1: {overall['avg_macro_f1']:.4f}")
        report_lines.append(f"Targets Meeting Threshold: {overall['targets_meeting_threshold']}/{overall['target_count']}")
        report_lines.append("")
    
    # Individual target performance
    report_lines.append("INDIVIDUAL TARGET PERFORMANCE")
    report_lines.append("-" * 40)
    
    for target, metrics in performance_results.items():
        if target.startswith('_'):
            continue
            
        report_lines.append(f"\n{target}:")
        report_lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")
        report_lines.append(f"  Weighted F1: {metrics['weighted_avg'].get('f1-score', 0):.4f}")
        report_lines.append(f"  Macro F1: {metrics['macro_avg'].get('f1-score', 0):.4f}")
        report_lines.append(f"  Classes: {len(metrics['unique_classes'])}")
        
        # Class distribution
        report_lines.append("  Class Distribution:")
        for class_name, count in metrics['class_distribution'].items():
            report_lines.append(f"    {class_name}: {count}")
    
    # Feature importance
    if feature_importance:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("FEATURE IMPORTANCE ANALYSIS")
        report_lines.append("=" * 80)
        
        for target, importance_data in feature_importance.items():
            report_lines.append(f"\n{target} - Top 10 Features:")
            report_lines.append("-" * 40)
            
            for i, feature_info in enumerate(importance_data['top_features'][:10], 1):
                report_lines.append(f"  {i:2d}. {feature_info['feature']}: {feature_info['importance']:.4f}")
    
    # Cross-validation results
    if cv_results:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("CROSS-VALIDATION RESULTS")
        report_lines.append("=" * 80)
        
        for target, cv_data in cv_results.items():
            report_lines.append(f"\n{target}:")
            report_lines.append(f"  Mean CV F1: {cv_data['mean_score']:.4f} ± {cv_data['std_score']:.4f}")
            report_lines.append(f"  Range: [{cv_data['min_score']:.4f}, {cv_data['max_score']:.4f}]")
    
    # Recommendations
    report_lines.append("\n" + "=" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("=" * 80)
    
    if '_overall' in performance_results:
        overall = performance_results['_overall']
        if overall['avg_weighted_f1'] >= PERFORMANCE_THRESHOLDS['target_f1_score']:
            report_lines.append("PASS: Model performance meets target thresholds")
        else:
            report_lines.append("FAIL: Model performance below target thresholds")
            report_lines.append("   Consider: Feature engineering, hyperparameter tuning, or more data")
        
        if overall['targets_meeting_threshold'] < overall['target_count']:
            report_lines.append("WARNING: Some targets underperforming - consider individual optimization")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Performance report saved to: {save_path}")
    
    return report_text

def save_metrics_to_json(
    performance_results: Dict[str, Dict[str, Any]],
    save_path: str
) -> None:
    """
    Save performance metrics to JSON file.
    
    Args:
        performance_results: Performance results to save
        save_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    
    for target, metrics in performance_results.items():
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_metrics[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}
            else:
                serializable_metrics[key] = value
        
        serializable_results[target] = serializable_metrics
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Metrics saved to JSON: {save_path}")

def summarize_metrics(performance_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary dataframe of key metrics.
    
    Args:
        performance_results: Performance results dictionary
        
    Returns:
        pd.DataFrame: Summary metrics dataframe
    """
    summary_data = []
    
    for target, metrics in performance_results.items():
        if target.startswith('_'):
            continue
            
        summary_data.append({
            'Target': target,
            'Accuracy': metrics['accuracy'],
            'Weighted F1': metrics['weighted_avg'].get('f1-score', 0),
            'Macro F1': metrics['macro_avg'].get('f1-score', 0),
            'Weighted Precision': metrics['weighted_avg'].get('precision', 0),
            'Weighted Recall': metrics['weighted_avg'].get('recall', 0),
            'Classes': len(metrics['unique_classes']),
            'Samples': sum(metrics['class_distribution'].values())
        })
    
    return pd.DataFrame(summary_data)

def plot_performance_metrics(
    performance_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Create visualizations of performance metrics.
    
    Args:
        performance_results: Performance results dictionary
        save_path: Path to save plots
    """
    try:
        # Create summary dataframe
        summary_df = summarize_metrics(performance_results)
        
        if summary_df.empty:
            print("No data available for plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Firco XGBoost Model Performance', fontsize=16)
        
        # Plot 1: F1 Scores
        axes[0, 0].bar(summary_df['Target'], summary_df['Weighted F1'])
        axes[0, 0].set_title('Weighted F1 Scores by Target')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=PERFORMANCE_THRESHOLDS['target_f1_score'], color='r', linestyle='--', label='Target')
        axes[0, 0].legend()
        
        # Plot 2: Accuracy
        axes[0, 1].bar(summary_df['Target'], summary_df['Accuracy'])
        axes[0, 1].set_title('Accuracy by Target')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=PERFORMANCE_THRESHOLDS['min_accuracy'], color='r', linestyle='--', label='Min Target')
        axes[0, 1].legend()
        
        # Plot 3: Precision vs Recall
        axes[1, 0].scatter(summary_df['Weighted Recall'], summary_df['Weighted Precision'])
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].set_xlabel('Weighted Recall')
        axes[1, 0].set_ylabel('Weighted Precision')
        for i, target in enumerate(summary_df['Target']):
            axes[1, 0].annotate(target, (summary_df['Weighted Recall'].iloc[i], summary_df['Weighted Precision'].iloc[i]))
        
        # Plot 4: Class Distribution
        axes[1, 1].bar(summary_df['Target'], summary_df['Classes'])
        axes[1, 1].set_title('Number of Classes by Target')
        axes[1, 1].set_ylabel('Number of Classes')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating performance plots: {e}")
        
def validate_model_requirements(model, X, y) -> Dict[str, Any]:
    """
    Validate that the model meets minimum requirements.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation_results = {
        'model_trained': False,
        'has_predictions': False,
        'meets_f1_threshold': False,
        'meets_accuracy_threshold': False,
        'errors': []
    }
    
    try:
        # Check if model is trained
        if hasattr(model, 'models') and model.models:
            validation_results['model_trained'] = True
        else:
            validation_results['errors'].append("Model not trained or empty")
        
        # Check if model can make predictions
        if validation_results['model_trained']:
            try:
                # Test with a small sample for validation, not limiting actual predictions
                test_sample = X.head(min(5, len(X)))
                predictions = model.predict(test_sample)
                validation_results['has_predictions'] = True
            except Exception as e:
                validation_results['errors'].append(f"Model prediction failed: {e}")
        
        # Validate performance thresholds
        if validation_results['has_predictions']:
            predictions = model.predict(X)
            performance = evaluate_model_performance(y, predictions, model.label_encoders)
            
            if '_overall' in performance:
                overall = performance['_overall']
                validation_results['meets_f1_threshold'] = overall['avg_weighted_f1'] >= PERFORMANCE_THRESHOLDS['target_f1_score']
                validation_results['meets_accuracy_threshold'] = overall['avg_accuracy'] >= PERFORMANCE_THRESHOLDS['min_accuracy']
    
    except Exception as e:
        validation_results['errors'].append(f"Validation error: {e}")
    
    return validation_results 