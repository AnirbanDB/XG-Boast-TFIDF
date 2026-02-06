"""
Utility functions for validation and response formatting.
This enables the same clean architecture as prediqai-Deep.
"""
from typing import Optional, List
from fastapi import HTTPException, UploadFile
from schemas import ModelInDB, TrainingRunInDB, ValidationRunInDB, PredictionRunInDB
from datetime import datetime
import uuid


def generate_operation_id() -> str:
    """Generate a unique ID for operations (training, validation, prediction)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + str(uuid.uuid4())[:12]


def generate_training_id(prefix: str = "train") -> str:
    """Generate a unique training ID with prefix."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"


def validate_csv_file(file: UploadFile):
    """Validate that uploaded file is a CSV"""
    if not file.filename.endswith('.csv'):
        print(f"[VALIDATE] Error: Only CSV files are allowed. Got: {file.filename}")
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")


def validate_level(level: str):
    """Validate training/prediction level"""
    if level not in ("hit", "message", "both"):
        print(f"[VALIDATE] Error: Invalid level: {level}")
        raise HTTPException(status_code=400, detail="level must be 'hit', 'message', or 'both'")


def validate_required_param(param, param_name: str):
    """Validate required parameter is provided"""
    if not param:
        raise HTTPException(status_code=400, detail=f"{param_name} is required")


def format_model_list_response(models: List[ModelInDB]):
    """Format models list response"""
    return {
        "models": [model.model_dump() for model in models],
        "total": len(models)
    }


def format_single_model_response(model: Optional[ModelInDB]):
    """Format single model response"""
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model.model_dump()


def format_training_runs_response(runs: List[TrainingRunInDB]):
    """Format training runs response"""
    return {
        "training_runs": [run.model_dump() for run in runs],
        "total": len(runs)
    }


def format_validation_runs_response(runs: List[ValidationRunInDB]):
    """Format validation runs response"""
    return {
        "validation_runs": [run.model_dump() for run in runs],
        "total": len(runs)
    }


def format_prediction_runs_response(runs: List[PredictionRunInDB]):
    """Format prediction runs response"""
    return {
        "prediction_runs": [run.model_dump() for run in runs],
        "total": len(runs)
    }


def format_single_run_response(run: Optional[TrainingRunInDB], run_type: str):
    """Format single run response"""
    if not run:
        raise HTTPException(status_code=404, detail=f"{run_type} not found")
    return run.model_dump()


async def get_training_id_from_model_id(model_id: str, model_crud, training_crud):
    """
    Get training_id from model_id for download operations.
    This enables the same download functionality as prediqai-Deep.
    """
    # Get model by model_id
    model = await model_crud.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get the training run for this model
    training_run = await training_crud.get_by_model_id(model_id)
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found for this model")
    
    return training_run.training_id


def get_download_files_and_name(level: str, training_id: str, model_id: str):
    """
    Get files to download and zip filename based on level.
    This enables the same download patterns as prediqai-Deep.
    """
    if level == "both":
        # Both hit and message models
        files_to_download = [
            (f"xgb_model_hit_{training_id}.pkl", f"xgb_model_hit_{training_id}.pkl"),
            (f"feature_encoder_hit_{training_id}.pkl", f"feature_encoder_hit_{training_id}.pkl"),
            (f"label_encoder_hit_{training_id}.pkl", f"label_encoder_hit_{training_id}.pkl"),
            (f"xgb_model_message_{training_id}.pkl", f"xgb_model_message_{training_id}.pkl"),
            (f"feature_encoder_message_{training_id}.pkl", f"feature_encoder_message_{training_id}.pkl"),
            (f"label_encoder_message_{training_id}.pkl", f"label_encoder_message_{training_id}.pkl")
        ]
        zip_filename = f"xgb_models_both_{model_id}.zip"
    else:
        # Single level model
        files_to_download = [
            (f"xgb_model_{level}_{training_id}.pkl", f"xgb_model_{level}_{training_id}.pkl"),
            (f"feature_encoder_{level}_{training_id}.pkl", f"feature_encoder_{level}_{training_id}.pkl"),
            (f"label_encoder_{level}_{training_id}.pkl", f"label_encoder_{level}_{training_id}.pkl")
        ]
        zip_filename = f"xgb_model_{level}_{model_id}.zip"
    
    return files_to_download, zip_filename
