# Firco TF-IDF + XGBoost Model with Hierarchical Training
# Advanced feature engineering and robust compliance prediction

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
import re
from collections import Counter

warnings.filterwarnings('ignore')

# Local imports for configuration
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import (
    LABEL_COLUMNS, HIT_LEVEL_TARGETS, MESSAGE_LEVEL_TARGETS,
    FEATURE_ENGINEERING, TEXT_PROCESSING, XGBOOST_PARAMS,
    FEATURE_SELECTION, CLASS_BALANCE_CONFIG, HIERARCHICAL_TRAINING_CONFIG
)

class SmartFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Smart feature transformer that creates fewer but more meaningful features.
    Focuses on deriving insightful features from multiple related columns.
    """
    
    def __init__(self):
        self.text_vectorizers = {}
        self.categorical_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """Fit the transformer on training data."""
        # Identify available columns
        self.text_columns = [col for col in FEATURE_ENGINEERING['text_columns'] if col in X.columns]
        self.categorical_columns = [col for col in FEATURE_ENGINEERING['categorical_columns'] if col in X.columns]
        self.numerical_columns = [col for col in FEATURE_ENGINEERING['numerical_columns'] if col in X.columns]
        
        # Fit text vectorizers for all available text columns
        for text_col in self.text_columns:
            if text_col in X.columns:
                text_data = X[text_col].fillna("").astype(str)
                
                # Check if we have enough text data
                non_empty_texts = [text for text in text_data if text.strip()]
                if len(non_empty_texts) >= 2:
                    try:
                        # Use simpler parameters for non-primary text columns
                        if text_col == 'hit.matching_text':
                            vectorizer = TfidfVectorizer(
                                max_features=TEXT_PROCESSING['max_features'],
                                ngram_range=TEXT_PROCESSING['ngram_range'],
                                min_df=TEXT_PROCESSING['min_df'],
                                max_df=TEXT_PROCESSING['max_df'],
                                sublinear_tf=TEXT_PROCESSING['sublinear_tf'],
                                stop_words=TEXT_PROCESSING['stop_words'],
                                norm=TEXT_PROCESSING['norm']
                            )
                        else:
                            # Simpler parameters for comment/secondary text columns
                            vectorizer = TfidfVectorizer(
                                max_features=500,  # Reduced for secondary columns
                                ngram_range=(1, 2),
                                min_df=2,
                                max_df=0.9
                            )
                        
                        vectorizer.fit(text_data)
                        self.text_vectorizers[text_col] = vectorizer
                    except:
                        # Fallback with even simpler parameters
                        vectorizer = TfidfVectorizer(
                            max_features=200,
                            ngram_range=(1, 1),
                            min_df=2,
                            max_df=0.9
                        )
                        try:
                            vectorizer.fit(text_data)
                            self.text_vectorizers[text_col] = vectorizer
                        except:
                            # Skip this text column if it fails completely
                            print(f"Warning: Could not fit vectorizer for {text_col}")
                            continue
        
        # Fit categorical encoders
        for col in self.categorical_columns:
            if col in X.columns:
                encoder = LabelEncoder()
                cat_data = X[col].fillna("Unknown").astype(str)
                encoder.fit(cat_data)
                self.categorical_encoders[col] = encoder
        
        # Fit numerical scaler
        if self.numerical_columns:
            available_num_cols = [col for col in self.numerical_columns if col in X.columns]
            if available_num_cols:
                numerical_data = X[available_num_cols].fillna(0)
                self.numerical_scaler.fit(numerical_data)
                self.numerical_columns = available_num_cols
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform features using fitted transformers."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        feature_matrices = []
        
        # Transform all text features
        for text_col in self.text_vectorizers.keys():
            if text_col in X.columns:
                text_data = X[text_col].fillna("").astype(str)
                text_matrix = self.text_vectorizers[text_col].transform(text_data)
                feature_matrices.append(text_matrix)
        
        # Transform categorical features
        categorical_data = []
        for col in self.categorical_columns:
            if col in X.columns:
                cat_data = X[col].fillna("Unknown").astype(str)
                # Handle unseen categories
                encoded_data = []
                for val in cat_data:
                    if val in self.categorical_encoders[col].classes_:
                        encoded_data.append(self.categorical_encoders[col].transform([val])[0])
                    else:
                        encoded_data.append(0)  # Default to first class
                categorical_data.append(encoded_data)
        
        if categorical_data:
            categorical_matrix = csr_matrix(np.array(categorical_data).T)
            feature_matrices.append(categorical_matrix)
        
        # Transform numerical features
        if self.numerical_columns:
            numerical_data_list = []
            for col in self.numerical_columns:
                if col in X.columns:
                    numerical_data_list.append(X[col].fillna(0).values)
            
            if numerical_data_list:
                numerical_data = np.column_stack(numerical_data_list)
                numerical_scaled = self.numerical_scaler.transform(numerical_data)
                numerical_matrix = csr_matrix(numerical_scaled)
                feature_matrices.append(numerical_matrix)
        
        # Combine all features
        if feature_matrices:
            combined_matrix = hstack(feature_matrices)
            return combined_matrix
        else:
            return csr_matrix((X.shape[0], 0))


class FircoHierarchicalXGBoost:
    """
    Hierarchical XGBoost model for Firco compliance prediction.
    Uses TF-IDF features and advanced feature engineering.
    """
    
    def __init__(self, label_encoders: Optional[Dict[str, LabelEncoder]] = None):
        """
        Initialize the Firco Hierarchical XGBoost model.
        
        Args:
            label_encoders: Optional pre-fitted label encoders
        """
        self.label_encoders = label_encoders or {}
        self.hit_level_models = {}
        self.message_level_models = {}
        self.feature_transformer = SmartFeatureTransformer()
        self.feature_selector = None
        self.is_trained = False
        self.feature_names = []
        self.training_stats = {}
        
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> 'FircoHierarchicalXGBoost':
        """
        Train the hierarchical XGBoost model.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame with encoded labels
            
        Returns:
            self: The trained model
        """
        print("   Starting hierarchical training...")
        
        # Transform features
        print("   Transforming features...")
        X_transformed = self.feature_transformer.fit_transform(X)
        
        # Apply feature selection if configured
        if FEATURE_SELECTION.get('apply_selection', False) and FEATURE_SELECTION.get('k_best', 0) > 0:
            print("   Applying feature selection...")
            k_best = min(FEATURE_SELECTION['k_best'], X_transformed.shape[1])
            self.feature_selector = SelectKBest(f_classif, k=k_best)
            X_transformed = self.feature_selector.fit_transform(X_transformed, y.iloc[:, 0])
        
        print(f"   Final feature matrix shape: {X_transformed.shape}")
        
        # Train models for each target
        target_performance = {}
        
        for target_col in LABEL_COLUMNS:
            if target_col in y.columns:
                print(f"   Training model for target: {target_col}")
                
                # Get target values
                y_target = y[target_col].values
                
                # Skip if all values are the same
                if len(np.unique(y_target)) < 2:
                    print(f"   Skipping {target_col} - insufficient class diversity")
                    continue
                
                # Fix XGBoost class validation issue by remapping classes
                unique_classes = np.unique(y_target)
                class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
                y_target_remapped = np.array([class_mapping[cls] for cls in y_target])
                
                print(f"   Remapped {len(unique_classes)} classes for {target_col}")
                
                # Configure XGBoost parameters
                xgb_params = XGBOOST_PARAMS.copy()
                xgb_params['num_class'] = len(unique_classes)  # Set to actual number of classes
                
                # Handle class imbalance
                if CLASS_BALANCE_CONFIG.get('handle_imbalance', False):
                    class_weights = compute_class_weight(
                        'balanced', 
                        classes=np.unique(y_target_remapped), 
                        y=y_target_remapped
                    )
                    # XGBoost doesn't use class_weight directly, but we can use scale_pos_weight
                    if len(np.unique(y_target_remapped)) == 2:
                        neg_count = np.sum(y_target_remapped == 0)
                        pos_count = np.sum(y_target_remapped == 1)
                        if pos_count > 0:
                            xgb_params['scale_pos_weight'] = neg_count / pos_count
                
                # Train XGBoost model
                model = xgb.XGBClassifier(**xgb_params)
                model.fit(X_transformed, y_target_remapped)
                
                # Store model with class mapping
                model.class_mapping = class_mapping
                model.reverse_mapping = {v: k for k, v in class_mapping.items()}
                
                # Store model
                if target_col in HIT_LEVEL_TARGETS:
                    self.hit_level_models[target_col] = model
                elif target_col in MESSAGE_LEVEL_TARGETS:
                    self.message_level_models[target_col] = model
                
                # Quick validation
                y_pred_remapped = model.predict(X_transformed)
                # Map predictions back to original labels for validation
                y_pred = np.array([model.reverse_mapping[pred] for pred in y_pred_remapped])
                accuracy = np.mean(y_pred == y_target)
                target_performance[target_col] = accuracy
                
                print(f"   {target_col} training accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        self.training_stats = {
            'feature_count': X_transformed.shape[1],
            'sample_count': X_transformed.shape[0],
            'target_performance': target_performance
        }
        
        print("   Hierarchical training completed successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions for all targets.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dict[str, np.ndarray]: Predictions for each target
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Transform features
        X_transformed = self.feature_transformer.transform(X)
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X_transformed = self.feature_selector.transform(X_transformed)
        
        predictions = {}
        
        # Make predictions for hit-level targets
        for target_col, model in self.hit_level_models.items():
            pred_remapped = model.predict(X_transformed)
            # Map back to original labels
            if hasattr(model, 'reverse_mapping'):
                predictions[target_col] = np.array([model.reverse_mapping[pred] for pred in pred_remapped])
            else:
                predictions[target_col] = pred_remapped
        
        # Make predictions for message-level targets  
        for target_col, model in self.message_level_models.items():
            pred_remapped = model.predict(X_transformed)
            # Map back to original labels
            if hasattr(model, 'reverse_mapping'):
                predictions[target_col] = np.array([model.reverse_mapping[pred] for pred in pred_remapped])
            else:
                predictions[target_col] = pred_remapped
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get prediction probabilities for all targets.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dict[str, np.ndarray]: Prediction probabilities for each target
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Transform features
        X_transformed = self.feature_transformer.transform(X)
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X_transformed = self.feature_selector.transform(X_transformed)
        
        probabilities = {}
        
        # Get probabilities for hit-level targets
        for target_col, model in self.hit_level_models.items():
            probabilities[target_col] = model.predict_proba(X_transformed)
        
        # Get probabilities for message-level targets
        for target_col, model in self.message_level_models.items():
            probabilities[target_col] = model.predict_proba(X_transformed)
        
        return probabilities
    
    def get_feature_importance(self, target_col: str) -> Optional[Dict[str, float]]:
        """
        Get feature importance for a specific target.
        
        Args:
            target_col: Target column name
            
        Returns:
            Optional[Dict[str, float]]: Feature importance scores
        """
        model = None
        if target_col in self.hit_level_models:
            model = self.hit_level_models[target_col]
        elif target_col in self.message_level_models:
            model = self.message_level_models[target_col]
        
        if model is None:
            return None
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create feature names (simplified)
        feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        return dict(zip(feature_names, importances))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        return {
            'is_trained': self.is_trained,
            'hit_level_targets': list(self.hit_level_models.keys()),
            'message_level_targets': list(self.message_level_models.keys()),
            'training_stats': self.training_stats,
            'model_type': 'FircoHierarchicalXGBoost'
        }
