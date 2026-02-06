# Database-Driven State Management for Firco XGBoost API
# Replaces global state with MongoDB for scalability (PrediqAI-Deep pattern)

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

from database import async_mongodb
from crud import (
    async_model_crud, async_training_crud, async_validation_crud,
    async_prediction_crud, async_status_crud
)
from schemas import ModelCreate, TrainingRunCreate, ModelType
from api_utils import get_latest_model_path, load_model, get_model_file_info

logger = logging.getLogger(__name__)

# ============================================================================
# STATE MANAGER (Database-driven, thread-safe)
# ============================================================================

class DatabaseStateManager:
    """
    Database-driven state management replacing global variables.
    Follows PrediqAI-Deep scalable architecture patterns.
    """
    
    def __init__(self):
        self._connection_initialized = False
    
    async def initialize(self):
        """Initialize database connection and collections."""
        if not self._connection_initialized:
            try:
                if await async_mongodb.connect():
                    logger.info("State Manager: MongoDB connected successfully")
                    await async_mongodb.create_collections_and_indexes()
                    logger.info("State Manager: Collections and indexes created")
                    self._connection_initialized = True
                    return True
                else:
                    logger.error("State Manager: Failed to connect to MongoDB")
                    return False
            except Exception as e:
                logger.error(f"State Manager initialization failed: {str(e)}")
                return False
        return True
    
    async def cleanup(self):
        """Cleanup database connections."""
        if self._connection_initialized:
            try:
                await async_mongodb.disconnect()
                logger.info("State Manager: MongoDB disconnected successfully")
                self._connection_initialized = False
            except Exception as e:
                logger.error(f"State Manager cleanup error: {str(e)}")
    
    # ========================================================================
    # MODEL STATE MANAGEMENT (Preserving your model logic)
    # ========================================================================
    
    async def get_current_model_info(self) -> Dict[str, Any]:
        """
        Get current model information from database and filesystem.
        Preserves your original model info logic.
        """
        try:
            # Get latest model from filesystem (preserving your logic)
            model_path = get_latest_model_path()
            if not model_path:
                return {
                    "status": "no_model",
                    "message": "No trained model available",
                    "model_info": None
                }
            
            # Get model file info (preserving your logic)
            model_info = get_model_file_info(model_path)
            
            # Get additional info from database
            try:
                # Try to find model in database
                models = await async_model_crud.list_user_models("api_user", 0, 1)
                if models:
                    latest_model = models[0]
                    model_info.update({
                        "model_id": latest_model.model_id,
                        "model_name": latest_model.model_name,
                        "level": latest_model.level,
                        "model_type": latest_model.model_type,
                        "status": latest_model.status
                    })
            except Exception as db_e:
                logger.warning(f"Could not get model from database: {db_e}")
            
            return {
                "status": "available",
                "message": "Model available and ready",
                "model_info": model_info
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {
                "status": "error",
                "message": f"Error retrieving model info: {str(e)}",
                "model_info": None
            }
    
    async def register_new_model(
        self,
        model_version: str,
        training_id: str,
        user_id: str = "api_user",
        level: str = "both",
        notes: Optional[str] = None
    ) -> Optional[str]:
        """
        Register a new model in the database.
        Integrates with your model versioning system.
        """
        try:
            model_create = ModelCreate(
                model_id=f"firco_xgb_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=user_id,
                model_name=f"Firco XGBoost TF-IDF {model_version}",
                model_version=int(model_version.replace('v', '').replace('.pkl', '')),
                tag="latest",
                level=level,
                model_type="firco_xgb_tfidf",
                framework="scikit-learn",
                notes=notes or f"TF-IDF + XGBoost hierarchical model {model_version}"
            )
            
            model_in_db = await async_model_crud.create(model_create)
            logger.info(f"Model {model_version} registered with ID: {model_in_db.model_id}")
            return model_in_db.model_id
            
        except Exception as e:
            logger.error(f"Failed to register model {model_version}: {str(e)}")
            return None

    async def update_model_status(
        self,
        training_id: str,
        status: str,
        model_path: str,
        s3_path: Optional[str],
        performance_summary: Dict[str, Any],
        training_time: float
    ):
        """
        Update model status, path, and performance metrics after training.
        This will create or update the model entry in the database.
        """
        try:
            training_run = await async_training_crud.get_by_training_id(training_id)
            if not training_run:
                logger.error(f"Cannot update model status: Training run {training_id} not found.")
                return

            model_version_str = Path(model_path).stem
            model_id_str = f"firco_xgb_{model_version_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Check if a model is already associated with the training_id
            model = await async_model_crud.get_by_training_id(training_id)

            if model:
                # Update the existing model's metadata
                update_data = {
                    "status": status,
                    "s3_path": s3_path,
                    "local_path": model_path,
                    "performance_summary": performance_summary,
                    "updated_at": datetime.now()
                }
                await async_model_crud.update(model.model_id, update_data)
                logger.info(f"Updated model {model.model_id} with status '{status}'.")
            else:
                # Create a new model entry if none exists
                model_create = ModelCreate(
                    model_id=model_id_str,
                    user_id=training_run.user_id,
                    model_name=f"Firco XGBoost TF-IDF {model_version_str}",
                    model_version=int(model_version_str.replace('v', '')),
                    tag="latest",
                    level=training_run.level,
                    model_type="firco_xgb_tfidf",
                    framework="scikit-learn",
                    status=status,
                    s3_path=s3_path,
                    local_path=model_path,
                    performance_summary=performance_summary,
                    training_id=training_id
                )
                await async_model_crud.create(model_create)
                logger.info(f"Registered new model {model_id_str} for training {training_id}.")

            # Update the training run itself
            await async_training_crud.update_status(training_id, status, training_time)
            logger.info(f"Updated training run {training_id} status to '{status}'.")

        except Exception as e:
            logger.error(f"Failed to update model status for training {training_id}: {str(e)}")
    
    # ========================================================================
    # TRAINING STATE MANAGEMENT
    # ========================================================================
    
    async def get_training_status(self, training_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get training status from database."""
        try:
            if training_id:
                status = await async_status_crud.get_by_training_id(training_id)
                if status:
                    return {
                        "is_training": status.is_training,
                        "status": status.status,
                        "progress": status.progress_percentage,
                        "message": f"Training {status.status}",
                        "start_time": status.start_time.isoformat() if status.start_time else None,
                        "end_time": status.end_time.isoformat() if status.end_time else None,
                        "training_id": training_id,
                        "metrics": status.metrics or {}
                    }
                else:
                    return {
                        "is_training": False,
                        "status": "not_found",
                        "message": "Training ID not found",
                        "training_id": training_id
                    }
            elif user_id:
                statuses = await async_status_crud.get_by_user(user_id)
                if statuses:
                    latest = statuses[0]  # Most recent
                    return {
                        "is_training": latest.is_training,
                        "status": latest.status,
                        "progress": latest.progress_percentage,
                        "message": f"Latest training {latest.status}",
                        "start_time": latest.start_time.isoformat() if latest.start_time else None,
                        "end_time": latest.end_time.isoformat() if latest.end_time else None,
                        "training_id": latest.training_id,
                        "metrics": latest.metrics or {}
                    }
            
            # Default idle state
            return {
                "is_training": False,
                "status": "idle",
                "message": "No training has been initiated",
                "progress": 0
            }
            
        except Exception as e:
            logger.error(f"Error getting training status: {str(e)}")
            return {
                "is_training": False,
                "status": "error",
                "message": f"Error retrieving status: {str(e)}",
                "progress": 0
            }
    
    # ========================================================================
    # HEALTH CHECK (PrediqAI-Deep pattern)
    # ========================================================================
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.
        Follows PrediqAI-Deep health check pattern.
        """
        try:
            # Check database connection
            if not self._connection_initialized:
                return {
                    "status": "unhealthy",
                    "mongodb": "not_connected",
                    "message": "Database not initialized"
                }
            
            # Test database connection
            await async_mongodb.client.admin.command('ping')
            
            # Check collections
            collections = await async_mongodb.database.list_collection_names()
            expected_collections = ["models", "training_runs", "validation_runs", "prediction_runs", "system_status"]
            missing_collections = [col for col in expected_collections if col not in collections]
            
            if missing_collections:
                return {
                    "status": "unhealthy",
                    "mongodb": "connected",
                    "collections": "incomplete",
                    "missing": missing_collections,
                    "message": f"Missing collections: {missing_collections}"
                }
            
            # Check model availability (preserving your model logic)
            model_info = await self.get_current_model_info()
            
            return {
                "status": "healthy",
                "mongodb": "connected",
                "collections": "complete",
                "model_status": model_info["status"],
                "existing_collections": collections,
                "message": "All systems operational"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "mongodb": "error",
                "error": str(e),
                "message": f"Health check failed: {str(e)}"
            }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    async def list_recent_training_runs(self, user_id: str = "api_user", limit: int = 10) -> List[Dict[str, Any]]:
        """List recent training runs for a user."""
        try:
            # Get user's models first
            models = await async_model_crud.list_user_models(user_id, 0, limit)
            
            training_runs = []
            for model in models:
                runs = await async_training_crud.list_by_model(model.model_id, 0, 5)
                for run in runs:
                    training_runs.append({
                        "training_id": run.training_id,
                        "model_id": run.model_id,
                        "status": run.status,
                        "level": run.level,
                        "created_at": run.created_at.isoformat(),
                        "updated_at": run.updated_at.isoformat()
                    })
            
            # Sort by creation time (most recent first)
            training_runs.sort(key=lambda x: x["created_at"], reverse=True)
            return training_runs[:limit]
            
        except Exception as e:
            logger.error(f"Error listing training runs: {str(e)}")
            return []

# Global state manager instance (thread-safe, database-driven)
state_manager = DatabaseStateManager()

# ============================================================================
# CONVENIENCE FUNCTIONS (Preserving your API patterns)
# ============================================================================

async def get_current_model_state() -> Dict[str, Any]:
    """Get current model state (replaces global app_state['model_info'])."""
    return await state_manager.get_current_model_info()

async def get_training_state(training_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Get training state (replaces global app_state['training_status'])."""
    return await state_manager.get_training_status(training_id, user_id)

async def get_health_state() -> Dict[str, Any]:
    """Get health state for monitoring."""
    return await state_manager.get_health_status()
