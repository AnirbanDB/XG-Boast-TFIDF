# Firco XGBoost FastAPI Application - Modernized Architecture
# Scalable, maintainable API preserving your excellent TF-IDF + XGBoost ML logic and UI features

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, File, UploadFile, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import os
import joblib
import traceback
import io
import glob
import json
import shutil
from datetime import datetime
from pathlib import Path
import logging
import asyncio

# Path setup and local imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config import (
    MODEL_SAVE_DIR, ARCHIVE_DIR, UPLOADS_DIR, PREDICTIONS_DIR,
    API_CONFIG, LABEL_COLUMNS, HIT_LEVEL_TARGETS, MESSAGE_LEVEL_TARGETS,
    VERSIONING_CONFIG, PERFORMANCE_THRESHOLDS
)
from dataset_utils import load_and_preprocess_data, create_label_encoders, encode_labels, split_data_hierarchical
from models import FircoHierarchicalXGBoost, ModelFactory
from main_xgb_F import train_model_without_save, predict_single_input, validate_model, predict_batch_csv
from train_utils import (
    evaluate_model_performance, calculate_feature_importance,
    generate_performance_report, summarize_metrics
)

# Modern modular imports (PrediqAI-Deep architecture pattern)
from api_utils import (
    validate_csv_file, validate_level, validate_required_param,
    get_latest_model_path, load_model, get_next_model_version,
    format_training_response, format_prediction_response, format_validation_response,
    format_model_info_response, format_health_response, format_response,
    count_existing_models, archive_previous_model, get_model_file_info
)
from async_tasks import task_manager, get_training_status
from state_manager import state_manager, get_current_model_state, get_training_state, get_health_state
from schemas import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with your excellent configuration
app = FastAPI(
    title=API_CONFIG['title'],
    description=API_CONFIG['description'],
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (preserving your configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router (preserving your routing structure)
router = APIRouter(prefix="/v1/firco-xgb")

# Create necessary directories (preserving your setup)
for directory in [MODEL_SAVE_DIR, ARCHIVE_DIR, UPLOADS_DIR, PREDICTIONS_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Modern async startup/shutdown (PrediqAI-Deep pattern)
@app.on_event("startup")
async def startup_event():
    """Initialize database connections and state manager."""
    logger.info("Starting Firco XGBoost API...")
    try:
        # Initialize state manager (replaces global state)
        if await state_manager.initialize():
            logger.info("State manager initialized successfully")
        else:
            logger.error("Failed to initialize state manager")
            raise RuntimeError("State manager initialization failed")
            
        logger.info("Firco XGBoost API started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Application startup failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup database connections and resources."""
    logger.info("Shutting down Firco XGBoost API...")
    try:
        await state_manager.cleanup()
        logger.info("State manager cleanup completed")
        logger.info("Firco XGBoost API shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Pydantic models for request/response
class TrainingRequest(BaseModel):
    config: Dict[str, Any] = {
        "label_cols": LABEL_COLUMNS,
        "text_col": "hit.matching_text",
        "random_state": 42,
        "test_size": 0.2,
        "validation_size": 0.15
    }

class TrainingResponse(BaseModel):
    message: str
    training_id: str
    model_version: str
    status: str
    estimated_time: str = "15-20 minutes"
    download_url: Optional[str] = None
    archived_models: int = 0

class PredictionRequest(BaseModel):
    text: str
    model_version: Optional[str] = None

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_version: str
    processing_time: float

class ModelInfo(BaseModel):
    version: str
    size_mb: float
    training_time: Optional[str]
    performance_summary: Optional[Dict[str, Any]]
    created_at: str

# ============================================================================
# ENDPOINT DEFINITIONS (Preserving your excellent API design and UI features)
# ============================================================================

# Root endpoint (preserving your design)
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint for Firco XGBoost API."""
    return {
        "message": "Firco XGBoost Compliance Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
        "description": "TF-IDF + XGBoost hierarchical compliance prediction"
    }

# Health check endpoint (modernized with database state)
@router.get("/health", tags=["Status"])
async def health_check():
    """
    Comprehensive health check endpoint.
    Now uses database-driven state instead of global variables.
    """
    health_status = await get_health_state()
    return format_health_response(
        status=health_status["status"],
        details=health_status
    )

# Training status endpoint (modernized)
@router.get("/training-status", tags=["Status"])
async def get_training_status_endpoint(
    training_id: Optional[str] = Query(None, description="Specific training ID to check"),
    user_id: Optional[str] = Query(None, description="User ID to get latest training status")
):
    """
    Get training status from database instead of global state.
    Supports both specific training ID and latest status for user.
    """
    return await get_training_status(training_id, user_id)

# Models endpoint (modernized with database integration)
@router.get("/models", tags=["Model Management"])
async def list_models(
    user_id: Optional[str] = Query("api_user", description="User ID to filter models"),
    include_archived: bool = Query(True, description="Include archived models in response")
):
    """
    List available models with enhanced information.
    Now integrates database records with filesystem data.
    """
    try:
        # Get current model info from state manager
        current_model_info = await get_current_model_state()
        
        # Get model files from filesystem (preserving your file system logic)
        current_models = []
        model_files = glob.glob(os.path.join(MODEL_SAVE_DIR, "*.pkl"))
        for model_file in model_files:
            file_info = get_model_file_info(model_file)
            current_models.append(file_info)
        
        # Get archived models if requested (preserving your archiving logic)
        archived_models = []
        if include_archived:
            archived_files = glob.glob(os.path.join(ARCHIVE_DIR, "archived_*.pkl_*"))
            for archived_file in archived_files:
                filename = os.path.basename(archived_file)
                version_match = filename.split('_')[1] if '_' in filename else "unknown"
                archived_models.append({
                    "name": filename,
                    "version": version_match,
                    "size_mb": round(os.path.getsize(archived_file) / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(os.path.getmtime(archived_file)).isoformat(),
                    "status": "archived"
                })
        
        return {
            "current_model_status": current_model_info["status"],
            "current_models": current_models,
            "archived_models": archived_models,
            "total_models": len(current_models) + len(archived_models),
            "latest_model": current_model_info.get("model_info"),
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# Training endpoint (modernized with async task manager)
@router.post("/train", tags=["Training & Prediction"])
async def start_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file containing training data"),
    config: str = Form(
        '{"label_cols": ["hit.review_decision", "hit.review_comments", "decision.last_action", "decision.reviewer_comments"], "text_col": "hit.matching_text"}',
        description="Training configuration as JSON string"
    ),
    user_id: str = Form("api_user", description="User ID for training tracking")
):
    """
    Start model training using modern async task management.
    Preserves your excellent TF-IDF + XGBoost ML logic while improving scalability.
    """
    # Validate inputs using modular utilities
    validate_csv_file(file)
    
    # Check if training is already in progress (database-driven check)
    current_training_state = await get_training_state(user_id=user_id)
    if current_training_state.get("is_training", False):
        raise HTTPException(
            status_code=409, 
            detail="A training process is already running. Please wait for it to complete."
        )
    
    try:
        # Parse configuration
        config_dict = json.loads(config)
        file_content = await file.read()
        
        # Generate training ID and model version (preserving your versioning logic)
        training_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        next_version = get_next_model_version()
        model_filename = f"v{next_version}.pkl"
        
        # Count existing models for archiving info
        archived_count = count_existing_models()
        
        # Start async training task using modern task manager
        background_tasks.add_task(
            task_manager.start_training_task,
            file_content, 
            config_dict, 
            training_id, 
            user_id
        )
        
        # Register model in database for tracking
        await state_manager.register_new_model(
            model_version=model_filename,
            training_id=training_id,
            user_id=user_id,
            level="both",
            notes=f"API training started at {datetime.now().isoformat()}"
        )
        
        # Return immediate response using formatting utility
        return format_training_response(
            training_id=training_id,
            model_version=model_filename,
            archived_count=archived_count
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in the configuration string.")
    except Exception as e:
        logger.error(f"Failed to start training process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while starting training: {str(e)}")

# Validation endpoint (modernized with async task manager)
@router.post("/validate", tags=["Training & Prediction"])
async def validate_model_endpoint(
    file: UploadFile = File(..., description="CSV file with ground truth data for validation"),
    training_id: Optional[str] = Query(None, description="Training ID to validate (uses latest if not provided)"),
    model_version: Optional[str] = Query(None, description="Model version to validate (e.g., 'v23', uses latest if not provided)"),
    level: str = Query("both", description="Validation level: 'hit', 'message', or 'both'"),
    model_type: str = Query("firco_xgb_tfidf", description="Model type for validation"),
    user_id: str = Query("api_user", description="User identifier for logging"),
    summary_only: Optional[bool] = Query(False, description="Return only summary metrics for Swagger UI (full results via curl)")
):
    """
    Validate model performance using modern async validation.
    Supports both training_id and model_version parameters for flexibility.
    
    Parameters:
    - training_id: Use a specific training session ID
    - model_version: Use a specific model version (e.g., 'v23')
    - If neither provided, uses the latest available model
    - summary_only: Return condensed metrics for Swagger UI display
    
    **Note for Swagger UI users**: Use summary_only=true for large datasets to avoid display issues.
    For complete detailed metrics, use curl or API clients directly.
    """
    # Validate inputs
    validate_csv_file(file)
    validate_level(level)
    
    # Determine which model to use (ENHANCED LOGIC)
    if model_version:
        # User specified a model version explicitly
        if not model_version.startswith('v'):
            model_version = f"v{model_version}"
        training_id = model_version
    elif training_id:
        # User specified a training_id - use as is
        pass
    else:
        # Use the latest available model file directly
        latest_model_path = get_latest_model_path()
        if latest_model_path:
            # Extract version from path (e.g., v23.pkl -> v23)
            model_filename = os.path.basename(latest_model_path)
            training_id = model_filename.replace('.pkl', '')
        else:
            raise HTTPException(
                status_code=404, 
                detail="No trained models found. Please train a model first or specify a training_id/model_version."
            )
        
        if not training_id:
            raise HTTPException(status_code=404, detail="No valid training ID found")
    
    try:
        file_content = await file.read()
        
        # Start async validation task
        validation_results = await task_manager.start_validation_task(
            file_content=file_content,
            training_id=training_id,
            level=level,
            model_type=model_type,
            user_id=user_id
        )
        
        # Extract metrics and format response (FIXED - direct access to results)
        validation_results_data = validation_results.get("results", {})
        
        # If validation_results_data is the performance dict from validate_model
        if "overall_metrics" in validation_results_data:
            metrics = validation_results_data
            accuracy = validation_results_data.get("overall_metrics", {}).get("accuracy", 0.0)
        else:
            # Fallback - treat the entire results as metrics
            metrics = validation_results_data
            accuracy = validation_results_data.get("accuracy", 0.0)
        
        # Handle summary response for Swagger UI
        if summary_only:
            # Return condensed metrics without detailed classification reports
            summary_metrics = {}
            dataset_info = metrics.get("dataset_info", {})
            overall_metrics = metrics.get("_overall", {})
            
            # Extract key metrics for each target
            for target_name, target_metrics in metrics.items():
                if target_name.startswith(('hit.', 'decision.')) and isinstance(target_metrics, dict):
                    summary_metrics[target_name] = {
                        "accuracy": target_metrics.get("accuracy", 0.0),
                        "weighted_f1": target_metrics.get("weighted_f1", 0.0)
                    }
            
            return {
                "message": "Validation completed successfully (Summary mode)",
                "summary_metrics": summary_metrics,
                "overall": overall_metrics,
                "dataset_info": dataset_info,
                "note": "This is a summary response. Use summary_only=false or curl for detailed classification reports.",
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        
        return format_validation_response(metrics=metrics, accuracy=accuracy)
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# Prediction endpoint (modernized)
@router.post("/predict", tags=["Training & Prediction"])
async def predict(
    text: Optional[str] = Form(None, description="Text input for single prediction"),
    file: Optional[UploadFile] = File(None, description="CSV file for batch predictions"),
    model_version: Optional[str] = Query(None, description="Model version (e.g., 'v16'). Leave blank to use latest trained model."),
    user_id: Optional[str] = Query("api_user", description="User identifier for logging")
):
    """
    Make predictions using text input or uploaded CSV file.
    
    **Usage:**
    - For single prediction: provide `text` parameter
    - For batch predictions: upload a CSV file
    - Optionally specify `model_version` (e.g., 'v16') or leave blank for latest model
    """
    # Validate inputs
    if not text and not file:
        raise HTTPException(status_code=400, detail="Either text or file must be provided for prediction")
    
    if file:
        validate_csv_file(file)
    
    # Get model info from state manager
    model_info = await get_current_model_state()
    if model_info["status"] != "available":
        raise HTTPException(status_code=404, detail="No trained model available. Please train a model first.")
    
    prediction_start_time = datetime.now()
    
    try:
        model = None
        model_path = None

        # Determine which model to use
        if model_version:
            # Add .pkl extension if not present
            if not model_version.endswith('.pkl'):
                model_version_with_ext = f"{model_version}.pkl"
            else:
                model_version_with_ext = model_version
            
            model_path = os.path.join(MODEL_SAVE_DIR, model_version_with_ext)
            if not os.path.exists(model_path):
                model_path = os.path.join(ARCHIVE_DIR, model_version_with_ext)
            
            model = load_model(model_path)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model version '{model_version}' not found")
        else:
            # Get latest model path
            model_path = get_latest_model_path()
            if not model_path:
                raise HTTPException(status_code=404, detail="No model file found. Please train a model first.")
            
            # Load model
            model = load_model(model_path)
            if not model:
                raise HTTPException(status_code=500, detail="Failed to load model. Please check model integrity.")
            
            # Extract version from path
            model_version = os.path.basename(model_path).replace('.pkl', '')

        predictions_list = []
        input_data_info = {}
        prediction_stats = {}
        
        # Prioritize batch prediction from file
        if file:
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="Only CSV files are supported for batch prediction")
            
            # Read file content
            file_content = await file.read()
            temp_file_path = os.path.join(UPLOADS_DIR, f"prediction_{prediction_start_time.strftime('%Y%m%d_%H%M%S')}.csv")
            
            try:
                # Save file temporarily
                with open(temp_file_path, 'wb') as f:
                    f.write(file_content)
                
                # Load and preprocess the uploaded data
                df = load_and_preprocess_data(temp_file_path, is_training=False)
                
                logger.info(f"Loaded prediction data with shape: {df.shape}")
                
                # Prepare features for prediction
                X = df.drop(columns=LABEL_COLUMNS, errors='ignore')
                logger.info(f"Feature matrix shape: {X.shape}")
                
                # Get predictions and probabilities using the model directly
                batch_predictions = model.predict(X)
                batch_probabilities = model.predict_proba(X)
                
                logger.info(f"Generated predictions for {len(df)} rows")
                logger.info(f"Generated predictions for targets: {list(batch_predictions.keys())}")
                
                # Count predictions by target
                hit_predictions = 0
                message_predictions = 0
                
                # Format results for each row
                for i in range(len(df)):
                    result = {}
                    for target, preds in batch_predictions.items():
                        if target in model.label_encoders:
                            encoder = model.label_encoders[target]
                            predicted_class_index = preds[i]
                            
                            predicted_class = encoder.inverse_transform([predicted_class_index])[0] if predicted_class_index < len(encoder.classes_) else "Unknown"
                            
                            class_probabilities = {}
                            if target in batch_probabilities:
                                proba_dist = batch_probabilities[target][i]
                                for j, class_name in enumerate(encoder.classes_):
                                    class_probabilities[class_name] = float(proba_dist[j])
                            
                            result[target] = {
                                'predicted_class': predicted_class,
                                'probabilities': class_probabilities
                            }
                            
                            # Count predictions by level
                            if target in ['hit.review_decision', 'hit.review_comments']:
                                hit_predictions += 1
                            elif target in ['decision.last_action', 'decision.reviewer_comments']:
                                message_predictions += 1
                    
                    predictions_list.append({
                        "input_id": i,
                        "input_text": df.iloc[i].get('hit.matching_text', 'N/A'),
                        "predictions": result
                    })
                
                input_data_info = {
                    "source_type": "csv",
                    "s3_key": f"uploads/prediction_input_{prediction_start_time.strftime('%Y%m%d_%H%M%S')}.csv",
                    "record_count": len(df)
                }
                
                prediction_stats = {
                    "total_predictions": len(predictions_list),
                    "hit_predictions": hit_predictions,
                    "message_predictions": message_predictions,
                    "results_s3_key": f"predictions/results_{prediction_start_time.strftime('%Y%m%d_%H%M%S')}.json"
                }
                
                logger.info(f"Formatted {len(predictions_list)} predictions for response")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        # Handle single text prediction only if no file is provided
        elif text:
            if not model_path:
                raise HTTPException(status_code=500, detail="Model path could not be determined for single prediction.")

            result = predict_single_input(text, model_path)
            predictions_list.append({
                "input_id": 0,
                "input_text": text,
                "predictions": result
            })
            
            input_data_info = {
                "source_type": "text",
                "record_count": 1
            }
            
            prediction_stats = {
                "total_predictions": 1,
                "hit_predictions": len([k for k in result.keys() if k in ['hit.review_decision', 'hit.review_comments']]),
                "message_predictions": len([k for k in result.keys() if k in ['decision.last_action', 'decision.reviewer_comments']]),
                "results_s3_key": f"predictions/single_result_{prediction_start_time.strftime('%Y%m%d_%H%M%S')}.json"
            }
        
        prediction_end_time = datetime.now()
        processing_time = (prediction_end_time - prediction_start_time).total_seconds()
        
        # Log prediction run to MongoDB (preserve your MongoDB integration)
        try:
            # Get model info from MongoDB
            version_num = 1
            if model_version:
                if model_version.startswith('v'):
                    version_num = int(model_version.replace('v', '').replace('.pkl', ''))
                else:
                    version_num = int(model_version.replace('.pkl', ''))
            
            # Insert prediction run record (if MongoDB integration is available)
            training_id = f"predict_{prediction_start_time.strftime('%Y%m%d_%H%M%S')}"
            predictions_per_second = len(predictions_list) / processing_time if processing_time > 0 else 0
            
            logger.info(f"Prediction run completed for model {model_version} with {len(predictions_list)} predictions")
                
        except Exception as mongo_error:
            logger.warning(f"MongoDB logging failed for prediction: {str(mongo_error)}")
        
        # Return the complete prediction response with ALL predictions
        return {
            "predictions": predictions_list,
            "model_version": model_version,
            "processing_time": processing_time,
            "total_predictions": len(predictions_list),
            "input_data_info": input_data_info,
            "prediction_stats": prediction_stats,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Model management endpoints

@router.get("/download-model/{model_version}", tags=["Model Management"])
async def download_model(model_version: str):
    """Download a specific model version."""
    try:
        model_path = None
        filename = None
        
        # Clean up model version (remove .pkl if present)
        clean_version = model_version.replace('.pkl', '')
        
        # Check current models first
        current_model_path = os.path.join(MODEL_SAVE_DIR, f"{clean_version}.pkl")
        if os.path.exists(current_model_path):
            model_path = current_model_path
            filename = f"{clean_version}.pkl"
        else:
            # Check archived models - search for pattern archived_v*.pkl_timestamp
            archived_files = glob.glob(os.path.join(ARCHIVE_DIR, f"archived_{clean_version}.pkl_*"))
            if archived_files:
                # Get the most recent one if multiple exist
                model_path = max(archived_files, key=os.path.getmtime)
                filename = os.path.basename(model_path)
        
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model version {model_version} not found")
        
        return FileResponse(
            path=model_path,
            media_type='application/octet-stream',
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")

@router.get("/feature-importance/{model_version}", tags=["Model Management"])
async def get_feature_importance(model_version: str):
    """Get feature importance for a specific model version."""
    try:
        # Clean up model version (add .pkl if not present)
        clean_version = model_version.replace('.pkl', '')
        model_filename = f"{clean_version}.pkl"
        
        # Check current models first
        model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
        if not os.path.exists(model_path):
            # Check archived models
            archived_files = glob.glob(os.path.join(ARCHIVE_DIR, f"archived_{clean_version}.pkl_*"))
            if archived_files:
                model_path = max(archived_files, key=os.path.getmtime)
            else:
                raise HTTPException(status_code=404, detail=f"Model version {model_version} not found")
        
        logger.info(f"Loading model from: {model_path}")
        model = load_model(model_path)
        if not model:
            raise HTTPException(status_code=500, detail="Failed to load model")
        
        # Try the model's built-in method first
        feature_importance = {}
        if hasattr(model, 'get_feature_importance'):
            try:
                feature_importance = model.get_feature_importance()
                logger.info("Used model's built-in get_feature_importance method")
            except Exception as e:
                logger.warning(f"Model's get_feature_importance failed: {e}")
                
        # Fallback to our custom function
        if not feature_importance:
            feature_importance = calculate_feature_importance(model)
            logger.info("Used custom calculate_feature_importance function")
        
        return {
            "model_version": model_version,
            "model_path": model_path,
            "feature_importance": feature_importance,
            "method_used": "builtin" if hasattr(model, 'get_feature_importance') else "custom",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}")

@router.get("/performance-report/{model_version}", tags=["Model Management"])
async def get_performance_report(model_version: str):
    """Get performance report for a specific model version."""
    try:
        # Clean up model version (add .pkl if not present)
        clean_version = model_version.replace('.pkl', '')
        model_filename = f"{clean_version}.pkl"
        
        # Check current models first
        model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
        if not os.path.exists(model_path):
            # Check archived models
            archived_files = glob.glob(os.path.join(ARCHIVE_DIR, f"archived_{clean_version}.pkl_*"))
            if archived_files:
                model_path = max(archived_files, key=os.path.getmtime)
            else:
                raise HTTPException(status_code=404, detail=f"Model version {model_version} not found")
        
        logger.info(f"Loading model for performance report from: {model_path}")
        model = load_model(model_path)
        if not model:
            raise HTTPException(status_code=500, detail="Failed to load model")
        
        # Get performance data from model metadata
        performance_data = {}
        
        # Check different possible locations for performance data
        if hasattr(model, 'training_metadata'):
            performance_data['training_metadata'] = model.training_metadata
            logger.info("Found training_metadata in model")
        
        if hasattr(model, 'performance_summary'):
            performance_data['performance_summary'] = model.performance_summary
            logger.info("Found performance_summary in model")
            
        if hasattr(model, 'training_results'):
            performance_data['training_results'] = model.training_results
            logger.info("Found training_results in model")
            
        # Try to get model attributes for debugging
        model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
        logger.info(f"Model attributes: {model_attrs}")
        
        # If no performance data found, try to get basic model info
        if not performance_data:
            performance_data = {
                "message": "No performance data found in model metadata",
                "available_attributes": model_attrs,
                "model_info": {
                    "model_type": str(type(model)),
                    "has_models": hasattr(model, 'models'),
                    "has_label_encoders": hasattr(model, 'label_encoders')
                }
            }
        
        return {
            "model_version": model_version,
            "model_path": model_path,
            "performance_data": performance_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting performance report: {str(e)}")

@router.post("/split-dataset", tags=["Data Management"])
async def split_dataset(
    file: UploadFile = File(...),
    test_size: float = Query(0.2, description="Test set size (0.1-0.5)"),
    validation_size: float = Query(0.15, description="Validation set size (0.1-0.3)"),
    random_state: int = Query(42, description="Random state for reproducibility")
):
    """Split uploaded dataset into training, validation, and test sets."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    if not (0.1 <= test_size <= 0.5):
        raise HTTPException(status_code=400, detail="Test size must be between 0.1 and 0.5")
    
    if not (0.1 <= validation_size <= 0.3):
        raise HTTPException(status_code=400, detail="Validation size must be between 0.1 and 0.3")
    
    try:
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file_path = os.path.join(UPLOADS_DIR, f"split_input_{timestamp}.csv")
        
        with open(temp_file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Load and preprocess data
        df = load_and_preprocess_data(temp_file_path, is_training=True)
        
        # Split data
        train_df, val_df, test_df = split_data_hierarchical(
            df, 
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state
        )
        
        # Save split datasets
        train_path = os.path.join(UPLOADS_DIR, f"train_split_{timestamp}.csv")
        val_path = os.path.join(UPLOADS_DIR, f"validation_split_{timestamp}.csv")
        test_path = os.path.join(UPLOADS_DIR, f"test_split_{timestamp}.csv")
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return {
            "message": "Dataset split successfully",
            "split_info": {
                "total_samples": len(df),
                "train_samples": len(train_df),
                "validation_samples": len(val_df),
                "test_samples": len(test_df),
                "train_percentage": round(len(train_df) / len(df) * 100, 2),
                "validation_percentage": round(len(val_df) / len(df) * 100, 2),
                "test_percentage": round(len(test_df) / len(df) * 100, 2)
            },
            "download_urls": {
                "train": f"/v1/firco-xgb/download-split/train_split_{timestamp}.csv",
                "validation": f"/v1/firco-xgb/download-split/validation_split_{timestamp}.csv",
                "test": f"/v1/firco-xgb/download-split/test_split_{timestamp}.csv"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Dataset splitting failed: {str(e)}")

@router.get("/download-split/{filename}", tags=["Data Management"])
async def download_split_file(filename: str):
    """Download a split dataset file."""
    try:
        file_path = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Split file {filename} not found")
        
        return FileResponse(
            path=file_path,
            media_type='text/csv',
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Error downloading split file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading split file: {str(e)}")

# Include router
app.include_router(router)

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    print("Starting Firco XGBoost API...")
    print(f"API will be available at: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print(f"Documentation at: http://{API_CONFIG['host']}:{API_CONFIG['port']}/docs")
    
    uvicorn.run(
        app,
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        log_level="info"
    )