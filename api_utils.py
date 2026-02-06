# API Utility Functions for Firco XGBoost API
# Extracted from monolithic xgb_app_F.py for better maintainability

import os
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, UploadFile
import glob
import shutil

from config import MODEL_SAVE_DIR, ARCHIVE_DIR
from s3_utils import download_file_from_s3, S3_BUCKET_NAME

logger = logging.getLogger(__name__)

# ============================================================================
# FILE VALIDATION UTILITIES (Enhanced from PrediqAI-Deep pattern)
# ============================================================================

def validate_csv_file(file: UploadFile):
    """Validate that the uploaded file is a CSV file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

def validate_level(level: str):
    """Validate that the level is one of the allowed values."""
    if level not in ("hit", "message", "both"):
        raise HTTPException(status_code=400, detail="level must be 'hit', 'message', or 'both'")

def validate_required_param(param_value: Any, param_name: str):
    """Validate that a required parameter is provided."""
    if not param_value:
        raise HTTPException(status_code=400, detail=f"{param_name} is required")

# ============================================================================
# MODEL MANAGEMENT UTILITIES (Preserving your model logic)
# ============================================================================

def get_latest_model_path() -> Optional[str]:
    """
    Get path to the latest model version.
    Preserves your original model versioning logic.
    """
    try:
        model_files = []
        for f in os.listdir(MODEL_SAVE_DIR):
            if f.startswith('v') and f.endswith('.pkl'):
                try:
                    version_num = int(f[1:-4])  # Extract number from v1.pkl -> 1
                    model_files.append((version_num, f))
                except ValueError:
                    continue
        
        if model_files:
            # Sort by version number and return the highest
            model_files.sort(key=lambda x: x[0])
            latest_model = model_files[-1][1]
            return os.path.join(MODEL_SAVE_DIR, latest_model)
        return None
    except Exception as e:
        logger.error(f"Error getting latest model path: {e}")
        return None

def load_model(model_path: str):
    """
    Load a model from the given path.
    If the model is not found locally, it attempts to download it from S3.
    """
    logger.info(f"Attempting to load model from: {model_path}")
    
    # --- S3 INTEGRATION: DOWNLOAD IF NOT EXISTS ---
    if not os.path.exists(model_path):
        logger.warning(f"Model not found locally at {model_path}. Attempting to download from S3.")
        
        # Construct the S3 object name from the local path
        s3_object_name = f"models/{os.path.basename(model_path)}"
        
        # Ensure the local directory exists before downloading
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        
        download_successful = download_file_from_s3(s3_object_name, model_path)
        
        if not download_successful:
            logger.error(f"Failed to download model from s3://{S3_BUCKET_NAME}/{s3_object_name}")
            return None
        logger.info(f"Model successfully downloaded to {model_path}")
    # --- END S3 INTEGRATION ---
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Validate model state after loading (NEW VALIDATION)
        if hasattr(model, 'validate_model_state'):
            validation_result = model.validate_model_state()
            logger.info(f"Model validation result: {validation_result}")
            
            if not validation_result['is_trained']:
                logger.error(f"Loaded model is not trained properly")
                return None
                
            if not validation_result['feature_transformer_fitted']:
                logger.error(f"Model's feature transformer is not fitted")
                return None
                
            # Check for unfitted vectorizers
            unfitted_vectorizers = [col for col, fitted in validation_result['vectorizers_fitted'].items() if not fitted]
            if unfitted_vectorizers:
                logger.error(f"Model has unfitted vectorizers: {unfitted_vectorizers}")
                return None
            
            # Check for TF-IDF transformer issues
            broken_tfidf = [col for col, ready in validation_result.get('vectorizers_tfidf_ready', {}).items() if not ready]
            if broken_tfidf:
                logger.warning(f"Model has broken TF-IDF transformers: {broken_tfidf} - attempting to fix...")
                
                # Try to fix any post-loading issues
                if hasattr(model, 'fix_model_state_after_loading'):
                    if not model.fix_model_state_after_loading():
                        logger.error("Failed to fix model state after loading")
                        return None
                    else:
                        logger.info("Successfully fixed model state after loading")
                else:
                    logger.error("Model doesn't have fix_model_state_after_loading method")
                    return None
            else:
                logger.info("Model validation passed - all TF-IDF transformers are ready")
                
                # Still try to run the fix method for any other issues
                if hasattr(model, 'fix_model_state_after_loading'):
                    model.fix_model_state_after_loading()
        else:
            logger.warning("Model doesn't have validation methods - this might be an old model format")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def get_next_model_version() -> int:
    """
    Determine the next model version number.
    Preserves your original versioning logic.
    """
    try:
        model_files = []
        for f in os.listdir(MODEL_SAVE_DIR):
            if f.startswith('v') and f.endswith('.pkl'):
                try:
                    version_num = int(f[1:-4])
                    model_files.append(version_num)
                except ValueError:
                    continue
        
        return max(model_files) + 1 if model_files else 1
    except Exception:
        return 1

def count_existing_models() -> int:
    """Count existing model files in the saved models directory."""
    try:
        pkl_files = glob.glob(os.path.join(MODEL_SAVE_DIR, "*.pkl"))
        return len(pkl_files)
    except Exception as e:
        logger.error(f"Error counting models: {str(e)}")
        return 0

def archive_previous_model(current_version: int) -> bool:
    """
    Archive the previous model when a new one is trained.
    Moves the previous model from saved_models to archive directory.
    """
    try:
        if current_version <= 1:
            # No previous model to archive
            return True
            
        previous_version = current_version - 1
        previous_model_filename = f"v{previous_version}.pkl"
        source_path = os.path.join(MODEL_SAVE_DIR, previous_model_filename)
        
        if os.path.exists(source_path):
            # Create archived filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archived_filename = f"archived_v{previous_version}.pkl_{timestamp}"
            destination_path = os.path.join(ARCHIVE_DIR, archived_filename)
            
            # Move file to archive
            shutil.move(source_path, destination_path)
            logger.info(f"Archived model v{previous_version} to {archived_filename}")
            return True
        else:
            logger.warning(f"Previous model v{previous_version} not found for archiving")
            return True  # Not an error if previous model doesn't exist
            
    except Exception as e:
        logger.error(f"Failed to archive previous model: {str(e)}")
        return False

# ============================================================================
# RESPONSE FORMATTING UTILITIES
# ============================================================================

def format_response(message: str, data: dict = None, status: str = "success") -> dict:
    """Format standard API response"""
    response = {
        "message": message,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    if data:
        response["data"] = data
    return response


def format_prediction_response(predictions: list, model_version: str, processing_time: float) -> dict:
    """Format prediction response with consistent structure"""
    return {
        "predictions": predictions,
        "model_version": model_version,
        "processing_time": processing_time,
        "total_predictions": len(predictions),
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }

def format_training_response(training_id: str, model_version: str, archived_count: int) -> Dict[str, Any]:
    """Format training response in a consistent manner."""
    return {
        "message": "Training started successfully",
        "training_id": training_id,
        "model_version": model_version,
        "status": "training",
        "estimated_time": "15-20 minutes",
        "download_url": f"/v1/firco-xgb/download-model/{model_version}",
        "archived_models": archived_count
    }

def format_validation_response(metrics: Dict[str, Any], accuracy: float) -> Dict[str, Any]:
    """Format validation response consistently."""
    return {
        "message": "Validation completed successfully",
        "metrics": metrics,
        "accuracy": accuracy
    }

def format_model_info_response(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Format model information response."""
    return {
        "model_info": model_info,
        "status": "available" if model_info else "no_model"
    }

def format_health_response(status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """Format health check response consistently."""
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "details": details
    }

# ============================================================================
# FILE MANAGEMENT UTILITIES
# ============================================================================

def get_model_file_info(model_path: str) -> Dict[str, Any]:
    """
    Get model file information.
    OPTIMIZED: Fast health check without loading entire model.
    """
    try:
        stat = os.stat(model_path)
        size_mb = stat.st_size / (1024 * 1024)
        created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # PERFORMANCE FIX: Don't load entire model for health check
        # Only load model metadata if specifically needed for detailed info
        training_time = None
        performance_summary = None
        
        # Quick health check - just verify file exists and is readable
        # Model metadata loading moved to separate function for when actually needed
        
        return {
            "version": os.path.basename(model_path),
            "size_mb": round(size_mb, 2),
            "training_time": training_time,
            "performance_summary": performance_summary,
            "created_at": created_at,
            "file_accessible": True
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {
            "version": os.path.basename(model_path) if model_path else "unknown",
            "size_mb": 0.0,
            "training_time": None,
            "performance_summary": None,
            "created_at": datetime.now().isoformat(),
            "file_accessible": False
        }

def get_detailed_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get detailed model information by loading the full model.
    Use this only when you need training metadata, not for health checks.
    """
    try:
        # Get basic file info first (fast)
        basic_info = get_model_file_info(model_path)
        
        # Load model for detailed metadata (slow)
        model = joblib.load(model_path)
        training_time = None
        performance_summary = None
        
        if hasattr(model, 'training_metadata'):
            metadata = model.training_metadata
            training_time = metadata.get('timestamp')
            performance_summary = metadata.get('performance_summary')
        
        # Update with detailed info
        basic_info.update({
            "training_time": training_time,
            "performance_summary": performance_summary,
            "model_loaded": True
        })
        
        return basic_info
    except Exception as e:
        logger.error(f"Error loading detailed model info: {str(e)}")
        # Fall back to basic info
        return get_model_file_info(model_path)

def get_model_file_info(model_file_path: str) -> Dict[str, Any]:
    """Get information about a model file."""
    try:
        filename = os.path.basename(model_file_path)
        version = filename.replace('.pkl', '')
        file_size = os.path.getsize(model_file_path)
        created_time = os.path.getctime(model_file_path)
        
        return {
            "name": filename,
            "version": version,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(created_time).isoformat(),
            "status": "active",
            "path": model_file_path
        }
    except Exception as e:
        logger.error(f"Error getting model file info: {str(e)}")
        return {
            "name": os.path.basename(model_file_path),
            "version": "unknown",
            "size_mb": 0.0,
            "created_at": datetime.now().isoformat(),
            "status": "error",
            "path": model_file_path
        }


# ============================================================================
# ERROR HANDLING UTILITIES
# ============================================================================

def handle_training_error(error: Exception, training_id: str) -> Dict[str, Any]:
    """Handle training errors consistently."""
    error_msg = f"Training failed for {training_id}: {str(error)}"
    logger.error(error_msg)
    return {
        "error": "training_failed",
        "message": error_msg,
        "training_id": training_id,
        "status": "failed"
    }

def handle_prediction_error(error: Exception) -> Dict[str, Any]:
    """Handle prediction errors consistently."""
    error_msg = f"Prediction failed: {str(error)}"
    logger.error(error_msg)
    return {
        "error": "prediction_failed",
        "message": error_msg,
        "predictions": []
    }

def handle_validation_error(error: Exception) -> Dict[str, Any]:
    """Handle validation errors consistently."""
    error_msg = f"Validation failed: {str(error)}"
    logger.error(error_msg)
    return {
        "error": "validation_failed",
        "message": error_msg,
        "metrics": {},
        "accuracy": 0.0
    }
