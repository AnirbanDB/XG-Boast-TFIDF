# Async Background Task Manager for Firco XGBoost API
# Extracted for better scalability and maintainability while preserving ML logic

import os
import asyncio
import joblib
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from config import UPLOADS_DIR, MODEL_SAVE_DIR, ARCHIVE_DIR
from main_xgb_F import train_model_without_save, validate_model
from train_utils import (
    evaluate_model_performance, calculate_feature_importance,
    generate_performance_report, summarize_metrics
)
from s3_utils import upload_file_to_s3, S3_BUCKET_NAME

# Modern modular imports (PrediqAI-Deep architecture pattern)
from api_utils import (
    get_model_file_info,
    archive_previous_model,
    count_existing_models,
    get_next_model_version,
    handle_training_error,
    handle_validation_error
)
from state_manager import state_manager
from crud import async_status_crud, async_training_crud
from schemas import SystemStatusCreate

logger = logging.getLogger(__name__)

# ============================================================================
# ASYNC TASK MANAGER 
# ============================================================================

class AsyncTaskManager:
    """
    Manages async background tasks for training and validation.
    Follows PrediqAI-Deep async patterns while preserving your ML logic.
    """
    
    def __init__(self):
        self.active_tasks = {}
        
    async def start_training_task(
        self, 
        file_content: bytes, 
        config: Dict[str, Any], 
        training_id: str, 
        user_id: str = "api_user"
    ) -> Dict[str, Any]:
        """
        Start async training task while preserving your TF-IDF + XGBoost logic.
        """
        start_time = datetime.now()
        
        try:
            # Create system status entry in MongoDB (PrediqAI-Deep pattern)
            status_create = SystemStatusCreate(
                training_id=training_id,
                user_id=user_id
            )
            await async_status_crud.create_or_update(status_create)
            
            # Update initial status
            await self._update_training_status(
                training_id=training_id,
                status="starting",
                progress=5.0,
                message="Initializing training process...",
                start_time=start_time
            )
            
            # Save uploaded file (preserving your file handling)
            temp_file_path = os.path.join(UPLOADS_DIR, f"training_data_{training_id}.csv")
            with open(temp_file_path, 'wb') as f:
                f.write(file_content)
            
            # Archive existing models (preserving your logic)
            archived_count = count_existing_models()
            next_version = get_next_model_version()
            model_filename = f"v{next_version}.pkl"
            
            await self._update_training_status(
                training_id=training_id,
                status="data_processing",
                progress=15.0,
                message="Processing training data...",
                metrics={"archived_models": archived_count}
            )
            
            # Train model using your original logic (PRESERVED)
            await self._update_training_status(
                training_id=training_id,
                status="training",
                progress=30.0,
                message="Training TF-IDF + XGBoost model..."
            )
            
            # Call your original training function (FIXED PARAMETERS)
            # Extract parameters from config
            label_cols = config.get("label_cols", ["hit.review_decision", "hit.review_comments", "decision.last_action", "decision.reviewer_comments"])
            text_col = config.get("text_col", "hit.matching_text")
            
            model, results = await asyncio.to_thread(
                train_model_without_save,
                temp_file_path,
                label_cols,
                text_col
            )
            
            # Attach training results to model for later retrieval
            performance_summary = results.get("performance_summary", {})
            model.training_results = results
            model.training_metadata = {
                "training_id": training_id,
                "training_time": (datetime.now() - start_time).total_seconds(),
                "performance_summary": performance_summary,
                "model_filename": model_filename
            }
            
            # Save the trained model with the specified filename
            model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
            joblib.dump(model, model_path)
            logger.info(f"Model saved successfully to {model_path}")
            logger.info(f"Training results attached to model: {list(results.keys())}")

            # --- S3 INTEGRATION: UPLOAD MODEL TO S3 ---
            s3_object_name = f"models/{os.path.basename(model_path)}"
            upload_successful = upload_file_to_s3(model_path, s3_object_name)
            s3_path = f"s3://{S3_BUCKET_NAME}/{s3_object_name}" if upload_successful else None
            
            if upload_successful:
                logger.info(f"Successfully uploaded model to {s3_path}")
            else:
                logger.error(f"Failed to upload model to S3. Model is only available locally at {model_path}")
            # --- END S3 INTEGRATION ---

            # Update model status in database with S3 path
            training_time = (datetime.now() - start_time).total_seconds()
            await state_manager.update_model_status(
                training_id=training_id,
                status="available",
                model_path=model_path,
                s3_path=s3_path, # <-- Pass the S3 path
                performance_summary=performance_summary,
                training_time=training_time
            )
            
            await self._update_training_status(
                training_id=training_id,
                status="completing",
                progress=90.0,
                message="Finalizing model training..."
            )
            
            # Archive previous model if exists (new functionality)
            if next_version > 1:
                archive_success = archive_previous_model(next_version)
                if archive_success:
                    logger.info(f"Successfully archived previous model before saving v{next_version}")
                else:
                    logger.warning(f"Failed to archive previous model before saving v{next_version}")
            
            # Complete training
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            await self._update_training_status(
                training_id=training_id,
                status="completed",
                progress=100.0,
                message="Training completed successfully",
                end_time=end_time,
                metrics={
                    "training_duration_seconds": training_duration,
                    "model_version": model_filename,
                    "archived_models": archived_count,
                    **results.get("metrics", {})
                }
            )
            
            # Clean up temporary files
            try:
                os.remove(temp_file_path)
            except:
                pass
                
            return {
                "status": "completed",
                "training_id": training_id,
                "model_version": model_filename,
                "results": results
            }
            
        except Exception as e:
            # Handle training error (improved error management)
            await self._update_training_status(
                training_id=training_id,
                status="failed",
                progress=0.0,
                message=f"Training failed: {str(e)}",
                end_time=datetime.now(),
                metrics={"error": str(e)}
            )
            
            error_result = handle_training_error(e, training_id)
            logger.error(f"Training task failed for {training_id}: {str(e)}")
            return error_result
    
    async def start_validation_task(
        self,
        file_content: bytes,
        training_id: str,
        level: str,
        model_type: str,
        user_id: str = "api_user"
    ) -> Dict[str, Any]:
        """
        Start async validation task while preserving your validation logic.
        """
        validation_id = f"val_{training_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Save validation file (preserving your file handling)
            temp_file_path = os.path.join(UPLOADS_DIR, f"validation_data_{validation_id}.csv")
            with open(temp_file_path, 'wb') as f:
                f.write(file_content)
            
            # Map training_id to model path (preserving your model versioning)
            if training_id.startswith('v') and training_id[1:].isdigit():
                model_filename = f"{training_id}.pkl"
            else:
                model_filename = f"v{training_id}.pkl"
            
            model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
            if not os.path.exists(model_path):
                # Check archive directory
                model_path = os.path.join(ARCHIVE_DIR, model_filename)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model {model_filename} not found in saved models or archive")
            
            # Call your original validation function (FIXED PARAMETERS)
            results = await asyncio.to_thread(
                validate_model,
                temp_file_path,
                model_path
            )
            
            # Clean up temporary files
            try:
                os.remove(temp_file_path)
            except:
                pass
                
            return {
                "status": "completed",
                "validation_id": validation_id,
                "results": results
            }
            
        except Exception as e:
            error_result = handle_validation_error(e)
            logger.error(f"Validation task failed for {training_id}: {str(e)}")
            return error_result
    
    async def _update_training_status(
        self,
        training_id: str,
        status: str,
        progress: float,
        message: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Update training status in MongoDB (following PrediqAI-Deep pattern)."""
        try:
            updates = {
                "status": status,
                "progress_percentage": progress,
                "updated_at": datetime.utcnow()
            }
            
            if start_time:
                updates["start_time"] = start_time
                updates["is_training"] = True
                
            if end_time:
                updates["end_time"] = end_time
                updates["is_training"] = False
                
            if metrics:
                updates["metrics"] = metrics
            
            await async_status_crud.update_status(training_id, updates)
            
        except Exception as e:
            logger.error(f"Failed to update training status: {str(e)}")
    
    async def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """Get training status from MongoDB (PrediqAI-Deep pattern)."""
        try:
            status = await async_status_crud.get_by_training_id(training_id)
            if status:
                return status.model_dump()
            else:
                return {
                    "training_id": training_id,
                    "status": "not_found",
                    "message": "Training ID not found"
                }
        except Exception as e:
            logger.error(f"Failed to get training status: {str(e)}")
            return {
                "training_id": training_id,
                "status": "error",
                "message": f"Error retrieving status: {str(e)}"
            }

# Global instance (thread-safe)
task_manager = AsyncTaskManager()

# ============================================================================
# ASYNC HELPER FUNCTIONS (PrediqAI-Deep pattern)
# ============================================================================

async def get_training_status(training_id: Optional[str], user_id: Optional[str]) -> Dict[str, Any]:
    """
    Get training status for a specific training ID or latest training for a user.
    Follows PrediqAI-Deep async pattern.
    """
    if training_id:
        return await task_manager.get_training_status(training_id)
    elif user_id:
        # Get latest training for user
        try:
            statuses = await async_status_crud.get_by_user(user_id)
            if statuses:
                latest_status = statuses[0]  # Most recent
                return latest_status.model_dump()
            else:
                return {
                    "user_id": user_id,
                    "status": "no_training",
                    "message": "No training found for user"
                }
        except Exception as e:
            return {
                "user_id": user_id,
                "status": "error",
                "message": f"Error retrieving status: {str(e)}"
            }
    else:
        return {
            "status": "error",
            "message": "Either training_id or user_id must be provided"
        }
