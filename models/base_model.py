# Base Model Architecture for Firco Compliance Prediction System
# Abstract base class providing consistent interface for all compliance models

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib
import os
import json
from pathlib import Path

class BaseFircoModel(ABC):
    """
    Abstract base class for all Firco compliance prediction models.
    
    This class defines the interface and common functionality that all 
    compliance models must implement. It follows the Template Method pattern
    and provides a consistent API for model operations.
    
    Key Features:
    - Standardized interface for training, prediction, and validation
    - Common preprocessing and validation methods
    - Hierarchical training support (hit-level and message-level)
    - Model versioning and metadata management
    - Consistent error handling and logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base model with configuration.
        
        Args:
            config: Dictionary containing model configuration parameters
                - model_id: Unique identifier for the model
                - model_type: Type of model (e.g., 'xgboost', 'bert', 'random_forest')
                - level: Prediction level ('hit', 'message', or 'both')
                - hierarchical_training: Enable hierarchical training
                - feature_config: Feature engineering configuration
        """
        self.config = config
        self.model_id = config.get('model_id', f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.model_type = config.get('model_type', 'unknown')
        self.level = config.get('level', 'both')  # 'hit', 'message', or 'both'
        self.version = config.get('version', '1.0.0')
        
        # Training state
        self.is_trained = False
        self.training_completed_at = None
        self.training_duration = None
        
        # Metrics and metadata
        self.training_metrics = {}
        self.validation_metrics = {}
        self.feature_columns = []
        self.target_columns = []
        self.model_artifacts = {}
        self.performance_thresholds = config.get('performance_thresholds', {})
        
        # Firco-specific attributes
        self.hierarchical_training = config.get('hierarchical_training', True)
        self.mt_types = config.get('mt_types', ['MT103', 'MT202'])
        self.hit_level_targets = config.get('hit_level_targets', [
            'hit.review_decision', 'hit.review_comments'
        ])
        self.message_level_targets = config.get('message_level_targets', [
            'decision.last_action', 'decision.reviewer_comments'
        ])
        
        # Feature engineering configuration
        self.feature_config = config.get('feature_config', {})
        self.text_columns = self.feature_config.get('text_columns', [])
        self.categorical_columns = self.feature_config.get('categorical_columns', [])
        self.numerical_columns = self.feature_config.get('numerical_columns', [])
        
        # Initialize logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.model_id}")
        
        # Label encoders for target variables
        self.label_encoders = {}
        
        self.logger.info(f"Initialized {self.__class__.__name__} with ID: {self.model_id}")
    
    @abstractmethod
    def train(self, 
              train_data: pd.DataFrame, 
              val_data: Optional[pd.DataFrame] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            train_data: Training dataset with features and targets
            val_data: Optional validation dataset
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics and metadata
            
        Example:
            {
                'training_duration': 120.5,
                'training_samples': 10000,
                'validation_samples': 2000,
                'hit_level_metrics': {...},
                'message_level_metrics': {...},
                'model_artifacts': [...]
            }
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, level: str = None) -> Dict[str, Any]:
        """
        Make predictions on the provided data.
        
        Args:
            data: Input data for prediction
            level: Prediction level ('hit', 'message', or 'both'). 
                  If None, uses model's default level.
            
        Returns:
            Dictionary containing predictions and metadata
            
        Example:
            {
                'predictions': [...],
                'probabilities': [...],
                'prediction_count': 100,
                'processing_time': 5.2,
                'level': 'both'
            }
        """
        pass
    
    @abstractmethod
    def predict_proba(self, data: pd.DataFrame, level: str = None) -> Dict[str, np.ndarray]:
        """
        Make probability predictions on the provided data.
        
        Args:
            data: Input data for prediction
            level: Prediction level ('hit', 'message', or 'both')
            
        Returns:
            Dictionary mapping target names to probability arrays
        """
        pass
    
    @abstractmethod
    def validate(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the model performance on test data.
        
        Args:
            test_data: Test dataset for validation
            
        Returns:
            Dictionary containing validation metrics
            
        Example:
            {
                'accuracy': 0.95,
                'precision': 0.94,
                'recall': 0.96,
                'f1_score': 0.95,
                'confusion_matrix': [...],
                'classification_report': {...}
            }
        """
        pass
    
    @abstractmethod
    def save_model(self, save_path: str) -> bool:
        """
        Save the trained model to the specified path.
        
        Args:
            save_path: Path where the model should be saved
            
        Returns:
            Boolean indicating success/failure
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from the specified path.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Boolean indicating success/failure
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self, level: str = None) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance scores from the trained model.
        
        Args:
            level: Model level ('hit', 'message', or 'both')
            
        Returns:
            Dictionary mapping target names to feature importance dictionaries
        """
        pass
    
    # ============================================================================
    # COMMON PREPROCESSING AND VALIDATION METHODS
    # ============================================================================
    
    def preprocess_firco_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Firco-specific data preprocessing.
        Can be overridden by subclasses for custom preprocessing.
        
        Args:
            data: Raw Firco alert data
            
        Returns:
            Preprocessed data ready for training/prediction
        """
        processed_data = data.copy()
        
        # Basic data cleaning
        processed_data = self._handle_missing_values(processed_data)
        processed_data = self._normalize_column_names(processed_data)
        processed_data = self._handle_mt_types(processed_data)
        processed_data = self._clean_text_columns(processed_data)
        
        self.logger.info(f"Preprocessed data: {processed_data.shape}")
        return processed_data
    
    def validate_firco_data(self, data: pd.DataFrame, is_training: bool = False) -> bool:
        """
        Validate Firco alert data format and requirements.
        
        Args:
            data: Data to validate
            is_training: Whether this is training data (stricter validation)
            
        Returns:
            Boolean indicating if data is valid
        """
        if data.empty:
            self.logger.error("Input data is empty")
            return False
        
        # Check for required columns
        required_cols = self.get_required_firco_columns(is_training)
        missing_cols = set(required_cols) - set(data.columns)
        
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Validate data types and ranges
        if not self._validate_data_types(data):
            return False
        
        # Training-specific validation
        if is_training:
            if not self._validate_training_targets(data):
                return False
        
        self.logger.info(f"Data validation passed for {len(data)} rows")
        return True
    
    def get_required_firco_columns(self, is_training: bool = False) -> List[str]:
        """
        Get list of required Firco columns.
        Can be overridden by subclasses.
        
        Args:
            is_training: Whether this is for training data
            
        Returns:
            List of required column names
        """
        base_columns = [
            'hit.matching_text',
            'hit.score',
            'hit.hit_type',
            'hit.matching_type',
            'hit.mt_type'
        ]
        
        if is_training:
            base_columns.extend(self.hit_level_targets + self.message_level_targets)
        
        return base_columns + self.feature_columns
    
    def extract_hierarchical_targets(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract hit-level and message-level targets for hierarchical training.
        
        Args:
            data: Input data with both levels
            
        Returns:
            Tuple of (hit_level_data, message_level_data)
        """
        # Hit-level data (each row is a hit)
        hit_data = data.copy()
        
        # Message-level data (aggregate hits by message_id)
        if 'message_id' in data.columns:
            message_data = data.groupby('message_id').agg({
                # Take first occurrence for most fields
                **{col: 'first' for col in data.columns if col not in self.hit_level_targets},
                # Aggregate hit-level targets meaningfully
                **{col: lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0] 
                   for col in self.hit_level_targets if col in data.columns}
            }).reset_index()
        else:
            # If no message_id, treat each row as a separate message
            message_data = data.copy()
        
        self.logger.info(f"Extracted hierarchical targets: {len(hit_data)} hits, {len(message_data)} messages")
        return hit_data, message_data
    
    # ============================================================================
    # MODEL INFORMATION AND METADATA METHODS
    # ============================================================================
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'version': self.version,
            'level': self.level,
            'is_trained': self.is_trained,
            'training_completed_at': self.training_completed_at,
            'training_duration': self.training_duration,
            'hierarchical_training': self.hierarchical_training,
            'mt_types': self.mt_types,
            'hit_level_targets': self.hit_level_targets,
            'message_level_targets': self.message_level_targets,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'performance_thresholds': self.performance_thresholds,
            'config': self.config,
            'artifacts': list(self.model_artifacts.keys())
        }
    
    def update_training_metrics(self, metrics: Dict[str, Any]):
        """
        Update training metrics.
        
        Args:
            metrics: Dictionary of training metrics
        """
        self.training_metrics.update(metrics)
        self.training_metrics['last_updated'] = datetime.utcnow().isoformat()
        self.logger.info(f"Updated training metrics: {list(metrics.keys())}")
    
    def update_validation_metrics(self, metrics: Dict[str, Any]):
        """
        Update validation metrics.
        
        Args:
            metrics: Dictionary of validation metrics
        """
        self.validation_metrics.update(metrics)
        self.validation_metrics['last_updated'] = datetime.utcnow().isoformat()
        self.logger.info(f"Updated validation metrics: {list(metrics.keys())}")
    
    def set_trained_status(self, status: bool = True, training_duration: float = None):
        """
        Set model training status.
        
        Args:
            status: Training completion status
            training_duration: Time taken for training in seconds
        """
        self.is_trained = status
        if status:
            self.training_completed_at = datetime.utcnow().isoformat()
            if training_duration:
                self.training_duration = training_duration
        self.logger.info(f"Training status set to: {status}")
    
    def check_performance_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check if model performance meets defined thresholds.
        
        Args:
            metrics: Performance metrics to check
            
        Returns:
            Dictionary indicating which thresholds are met
        """
        results = {}
        for metric_name, threshold in self.performance_thresholds.items():
            if metric_name in metrics:
                results[metric_name] = metrics[metric_name] >= threshold
            else:
                results[metric_name] = False
        
        return results
    
    # ============================================================================
    # PRIVATE HELPER METHODS
    # ============================================================================
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill text columns with empty string
        text_cols = [col for col in self.text_columns if col in data.columns]
        for col in text_cols:
            data[col] = data[col].fillna('')
        
        # Fill numerical columns with 0
        numerical_cols = [col for col in self.numerical_columns if col in data.columns]
        for col in numerical_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Fill categorical columns with 'Unknown'
        categorical_cols = [col for col in self.categorical_columns if col in data.columns]
        for col in categorical_cols:
            data[col] = data[col].fillna('Unknown')
        
        return data
    
    def _normalize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for consistency."""
        # Remove extra whitespace and standardize format
        data.columns = data.columns.str.strip()
        return data
    
    def _handle_mt_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle MT type specific processing."""
        if 'hit.mt_type' in data.columns:
            # Ensure MT type values are standardized
            data['hit.mt_type'] = data['hit.mt_type'].fillna('MT103')
            # Add derived features based on MT type
            data['is_mt103'] = (data['hit.mt_type'] == 'MT103').astype(int)
            data['is_mt202'] = (data['hit.mt_type'] == 'MT202').astype(int)
        
        return data
    
    def _clean_text_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns."""
        text_cols = [col for col in self.text_columns if col in data.columns]
        for col in text_cols:
            if data[col].dtype == 'object':
                # Basic text cleaning
                data[col] = data[col].astype(str)
                data[col] = data[col].str.strip()
                data[col] = data[col].replace('nan', '')
        
        return data
    
    def _validate_data_types(self, data: pd.DataFrame) -> bool:
        """Validate data types are appropriate."""
        try:
            # Check numerical columns
            numerical_cols = [col for col in self.numerical_columns if col in data.columns]
            for col in numerical_cols:
                pd.to_numeric(data[col], errors='raise')
            
            return True
        except Exception as e:
            self.logger.error(f"Data type validation failed: {str(e)}")
            return False
    
    def _validate_training_targets(self, data: pd.DataFrame) -> bool:
        """Validate training target columns."""
        required_targets = []
        
        if self.level in ['hit', 'both']:
            required_targets.extend([col for col in self.hit_level_targets if col in data.columns])
        
        if self.level in ['message', 'both']:
            required_targets.extend([col for col in self.message_level_targets if col in data.columns])
        
        if not required_targets:
            self.logger.error("No valid target columns found for training")
            return False
        
        # Check for sufficient non-null values in targets
        for target in required_targets:
            if data[target].isnull().sum() > len(data) * 0.9:  # More than 90% null
                self.logger.error(f"Target column '{target}' has too many null values")
                return False
        
        return True
    
    def save_model_metadata(self, save_path: str) -> bool:
        """
        Save model metadata to a JSON file.
        
        Args:
            save_path: Directory path to save metadata
            
        Returns:
            Success status
        """
        try:
            metadata_path = os.path.join(save_path, 'model_metadata.json')
            model_info = self.get_model_info()
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            serializable_info = convert_numpy_types(model_info)
            
            with open(metadata_path, 'w') as f:
                json.dump(serializable_info, f, indent=2, default=str)
            
            self.logger.info(f"Model metadata saved to {metadata_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model metadata: {str(e)}")
            return False
    
    def load_model_metadata(self, model_path: str) -> bool:
        """
        Load model metadata from a JSON file.
        
        Args:
            model_path: Directory path containing metadata
            
        Returns:
            Success status
        """
        try:
            metadata_path = os.path.join(model_path, 'model_metadata.json')
            
            if not os.path.exists(metadata_path):
                self.logger.warning("Model metadata file not found")
                return False
            
            with open(metadata_path, 'r') as f:
                model_info = json.load(f)
            
            # Restore metadata
            self.model_id = model_info.get('model_id', self.model_id)
            self.model_type = model_info.get('model_type', self.model_type)
            self.version = model_info.get('version', self.version)
            self.level = model_info.get('level', self.level)
            self.is_trained = model_info.get('is_trained', False)
            self.training_completed_at = model_info.get('training_completed_at')
            self.training_duration = model_info.get('training_duration')
            self.training_metrics = model_info.get('training_metrics', {})
            self.validation_metrics = model_info.get('validation_metrics', {})
            self.feature_columns = model_info.get('feature_columns', [])
            self.target_columns = model_info.get('target_columns', [])
            
            self.logger.info(f"Model metadata loaded from {metadata_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model metadata: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """String representation of the model."""
        return (f"{self.__class__.__name__}("
                f"id={self.model_id}, "
                f"type={self.model_type}, "
                f"level={self.level}, "
                f"trained={self.is_trained})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return self.__str__()


class ModelFactory:
    """
    Factory class for creating different types of Firco compliance models.
    
    This implements the Factory pattern to provide a centralized way to create
    model instances based on model type configuration.
    """
    
    _model_registry = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """
        Register a model class with the factory.
        
        Args:
            model_type: String identifier for the model type
            model_class: Class that implements BaseFircoModel
        """
        if not issubclass(model_class, BaseFircoModel):
            raise ValueError(f"Model class must inherit from BaseFircoModel")
        
        cls._model_registry[model_type] = model_class
    
    @classmethod
    def create_model(cls, model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseFircoModel:
        """
        Create a model instance based on the model type.
        
        Args:
            model_type: Type of model to create
            config: Optional model configuration dictionary
            
        Returns:
            Instance of the specified model type
            
        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in cls._model_registry:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(cls._model_registry.keys())}")
        
        model_class = cls._model_registry[model_type]
        
        # Use default config if none provided
        if config is None:
            config = {
                'model_id': model_type,
                'model_type': model_type,
                'version': '1.0',
                'hierarchical_training': True
            }
        
        return model_class(config)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List of registered model type strings
        """
        return list(cls._model_registry.keys())
