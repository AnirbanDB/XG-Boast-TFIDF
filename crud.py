"""
CRUD operations for MongoDB collections.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from database import get_async_database
from schemas import (
    ModelCreate, ModelInDB,
    TrainingRunCreate, TrainingRunInDB,
    ValidationRunCreate, ValidationRunInDB,
    PredictionRunCreate, PredictionRunInDB,
    SystemStatusCreate, SystemStatusInDB
)

# ============================================================================
# ASYNC CRUD OPERATIONS 
# ============================================================================

from motor.motor_asyncio import AsyncIOMotorCollection
from database import get_async_collection

class AsyncModelCRUD:
    """Async CRUD operations for models collection"""
    
    def __init__(self):
        self.collection_name = "models"
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Lazy-load async collection"""
        return get_async_collection(self.collection_name)
    
    async def create(self, model_data: ModelCreate) -> ModelInDB:
        """Create a new model asynchronously"""
        model_dict = model_data.model_dump()
        model_dict["created_at"] = datetime.utcnow()
        model_dict["updated_at"] = datetime.utcnow()
        model_dict["status"] = "ACTIVE"
        model_dict["deployed"] = False
        
        # Generate model_id if not provided
        if not model_dict.get("model_id"):
            model_dict["model_id"] = datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + str(uuid.uuid4())[:12]
        
        result = await self.collection.insert_one(model_dict)
        model_dict["_id"] = str(result.inserted_id)
        return ModelInDB(**model_dict)
    
    async def get_by_id(self, model_id: str) -> Optional[ModelInDB]:
        """Get model by ID asynchronously"""
        model_data = await self.collection.find_one({"model_id": model_id})
        if model_data:
            model_data["_id"] = str(model_data["_id"])
            return ModelInDB(**model_data)
        return None
    
    async def get_by_training_id(self, training_id: str) -> Optional[ModelInDB]:
        """Get model by training_id asynchronously"""
        model_data = await self.collection.find_one({"training_id": training_id})
        if model_data:
            model_data["_id"] = str(model_data["_id"])
            return ModelInDB(**model_data)
        return None
    
    async def get_by_model_id(self, model_id: str) -> Optional[ModelInDB]:
        """Get model by model_id asynchronously"""
        return await self.get_by_id(model_id)
    
    async def list_user_models(self, user_id: str, skip: int = 0, limit: int = 10) -> List[ModelInDB]:
        """List models for a user asynchronously"""
        cursor = self.collection.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit)
        models = []
        async for model_data in cursor:
            model_data["_id"] = str(model_data["_id"])
            models.append(ModelInDB(**model_data))
        return models
    
    async def update(self, model_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a model's data."""
        update_data["updated_at"] = datetime.utcnow()
        result = await self.collection.update_one(
            {"model_id": model_id},
            {"$set": update_data}
        )
        return result.modified_count > 0

    async def update_status(self, model_id: str, status: str) -> bool:
        """Update model status asynchronously"""
        result = await self.collection.update_one(
            {"model_id": model_id},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0

class AsyncTrainingRunCRUD:
    """Async CRUD operations for training runs collection"""
    
    def __init__(self):
        self.collection_name = "training_runs"
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Lazy-load async collection"""
        return get_async_collection(self.collection_name)
    
    async def create(self, training_data: TrainingRunCreate) -> TrainingRunInDB:
        """Create a new training run asynchronously"""
        training_dict = training_data.model_dump()
        training_dict["created_at"] = datetime.utcnow()
        training_dict["updated_at"] = datetime.utcnow()
        training_dict["status"] = "PENDING"
        training_dict["artifacts"] = {}
        training_dict["metrics"] = {}
        
        result = await self.collection.insert_one(training_dict)
        training_dict["_id"] = str(result.inserted_id)
        return TrainingRunInDB(**training_dict)
    
    async def get_by_training_id(self, training_id: str) -> Optional[TrainingRunInDB]:
        """Get training run by training_id asynchronously"""
        training_data = await self.collection.find_one({"training_id": training_id})
        if training_data:
            training_data["_id"] = str(training_data["_id"])
            return TrainingRunInDB(**training_data)
        return None
    
    async def list_by_model(self, model_id: str, skip: int = 0, limit: int = 10) -> List[TrainingRunInDB]:
        """List training runs for a model asynchronously"""
        cursor = self.collection.find({"model_id": model_id}).sort("created_at", -1).skip(skip).limit(limit)
        runs = []
        async for run_data in cursor:
            run_data["_id"] = str(run_data["_id"])
            runs.append(TrainingRunInDB(**run_data))
        return runs
    
    async def update_status(self, training_id: str, status: str, training_time: Optional[float] = None) -> bool:
        """Update training run status and duration asynchronously"""
        update_data = {"status": status, "updated_at": datetime.utcnow()}
        if training_time is not None:
            update_data["training_duration_seconds"] = training_time
        
        result = await self.collection.update_one(
            {"training_id": training_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    async def update_artifacts(self, training_id: str, artifacts: Dict) -> bool:
        """Update training run artifacts asynchronously"""
        result = await self.collection.update_one(
            {"training_id": training_id},
            {"$set": {"artifacts": artifacts, "updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0
    
    async def update_data_info(self, training_id: str, data_size: int, datasets: Dict) -> bool:
        """Update training run data info asynchronously"""
        result = await self.collection.update_one(
            {"training_id": training_id},
            {"$set": {
                "data_size": data_size,
                "datasets": datasets,
                "updated_at": datetime.utcnow()
            }}
        )
        return result.modified_count > 0

class AsyncValidationRunCRUD:
    """Async CRUD operations for validation runs collection"""
    
    def __init__(self):
        self.collection_name = "validation_runs"
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Lazy-load async collection"""
        return get_async_collection(self.collection_name)
    
    async def create(self, validation_data: ValidationRunCreate) -> ValidationRunInDB:
        """Create a new validation run asynchronously"""
        validation_dict = validation_data.model_dump()
        validation_dict["created_at"] = datetime.utcnow()
        validation_dict["updated_at"] = datetime.utcnow()
        validation_dict["metrics"] = {}
        validation_dict["data_size"] = {}
        
        result = await self.collection.insert_one(validation_dict)
        validation_dict["_id"] = str(result.inserted_id)
        return ValidationRunInDB(**validation_dict)
    
    async def update_results(self, validation_id: str, metrics: Dict, data_size: Dict) -> bool:
        """Update validation results asynchronously"""
        result = await self.collection.update_one(
            {"validation_id": validation_id},
            {"$set": {
                "metrics": metrics,
                "data_size": data_size,
                "updated_at": datetime.utcnow()
            }}
        )
        return result.modified_count > 0
    
    async def list_by_training(self, training_id: str) -> List[ValidationRunInDB]:
        """List validation runs for a training run asynchronously"""
        cursor = self.collection.find({"training_id": training_id}).sort("created_at", -1)
        runs = []
        async for run_data in cursor:
            run_data["_id"] = str(run_data["_id"])
            runs.append(ValidationRunInDB(**run_data))
        return runs

class AsyncPredictionRunCRUD:
    """Async CRUD operations for prediction runs collection"""
    
    def __init__(self):
        self.collection_name = "prediction_runs"
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Lazy-load async collection"""
        return get_async_collection(self.collection_name)
    
    async def create(self, prediction_data: PredictionRunCreate) -> PredictionRunInDB:
        """Create a new prediction run asynchronously"""
        prediction_dict = prediction_data.model_dump()
        prediction_dict["created_at"] = datetime.utcnow()
        prediction_dict["updated_at"] = datetime.utcnow()
        prediction_dict["results"] = {}
        prediction_dict["performance"] = {}
        
        result = await self.collection.insert_one(prediction_dict)
        prediction_dict["_id"] = str(result.inserted_id)
        return PredictionRunInDB(**prediction_dict)
    
    async def update_results(self, prediction_id: str, results: Dict, performance: Dict) -> bool:
        """Update prediction results asynchronously"""
        result = await self.collection.update_one(
            {"prediction_id": prediction_id},
            {"$set": {
                "results": results,
                "performance": performance,
                "updated_at": datetime.utcnow()
            }}
        )
        return result.modified_count > 0
    
    async def list_by_training(self, training_id: str, skip: int = 0, limit: int = 10) -> List[PredictionRunInDB]:
        """List prediction runs for a training run asynchronously"""
        cursor = self.collection.find({"training_id": training_id}).sort("created_at", -1).skip(skip).limit(limit)
        runs = []
        async for run_data in cursor:
            run_data["_id"] = str(run_data["_id"])
            runs.append(PredictionRunInDB(**run_data))
        return runs

class AsyncSystemStatusCRUD:
    """Async CRUD operations for system status collection"""
    
    def __init__(self):
        self.collection_name = "system_status"
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Lazy-load async collection"""
        return get_async_collection(self.collection_name)
    
    async def create_or_update(self, status_data: SystemStatusCreate) -> SystemStatusInDB:
        """Create or update system status asynchronously"""
        status_dict = status_data.model_dump()
        status_dict["created_at"] = datetime.utcnow()
        status_dict["updated_at"] = datetime.utcnow()
        status_dict["is_training"] = False
        status_dict["current_epoch"] = 0
        status_dict["total_epochs"] = 0
        status_dict["current_loss"] = 0.0
        status_dict["start_time"] = None
        status_dict["end_time"] = None
        status_dict["status"] = "idle"
        status_dict["progress_percentage"] = 0.0
        status_dict["metrics"] = {}
        
        # Use upsert to create or update
        result = await self.collection.find_one_and_update(
            {"training_id": status_data.training_id},
            {"$set": status_dict},
            upsert=True,
            return_document=True
        )
        
        result["_id"] = str(result["_id"])
        return SystemStatusInDB(**result)
    
    async def update_status(self, training_id: str, updates: Dict) -> bool:
        """Update system status asynchronously"""
        updates["updated_at"] = datetime.utcnow()
        result = await self.collection.update_one(
            {"training_id": training_id},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    async def get_by_training_id(self, training_id: str) -> Optional[SystemStatusInDB]:
        """Get status by training_id asynchronously"""
        status_data = await self.collection.find_one({"training_id": training_id})
        if status_data:
            status_data["_id"] = str(status_data["_id"])
            return SystemStatusInDB(**status_data)
        return None
    
    async def get_by_user(self, user_id: str) -> List[SystemStatusInDB]:
        """Get all statuses for a user asynchronously"""
        cursor = self.collection.find({"user_id": user_id}).sort("created_at", -1)
        statuses = []
        async for status_data in cursor:
            status_data["_id"] = str(status_data["_id"])
            statuses.append(SystemStatusInDB(**status_data))
        return statuses

# Global Async CRUD instances
async_model_crud = AsyncModelCRUD()
async_training_crud = AsyncTrainingRunCRUD()
async_validation_crud = AsyncValidationRunCRUD()
async_prediction_crud = AsyncPredictionRunCRUD()
async_status_crud = AsyncSystemStatusCRUD()
