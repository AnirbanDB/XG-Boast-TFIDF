# MongoDB utilities for Firco XGBoost Compliance Predictor
# Handles logging of models, training runs, validation runs, and predictions

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from pymongo import MongoClient
from bson import ObjectId
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "firco_compliance")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDBClient:
    """MongoDB client for Firco XGBoost system"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self._connect()
    
    def _connect(self):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[MONGO_DB]
            # Test connection
            self.client.admin.command('ismaster')
            logger.info(f"Connected to MongoDB database: {MONGO_DB}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.client = None
            self.db = None
    
    def _get_utc_timestamp(self) -> datetime:
        """Get current UTC timestamp"""
        return datetime.now(timezone.utc)
    
    def _serialize_for_mongo(self, data: Any) -> Any:
        """Convert data types that aren't MongoDB compatible"""
        if isinstance(data, dict):
            return {k: self._serialize_for_mongo(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_mongo(item) for item in data]
        elif hasattr(data, 'tolist'):  # numpy arrays
            return data.tolist()
        elif hasattr(data, 'item'):  # numpy scalars
            return data.item()
        else:
            return data

# Global MongoDB client instance
mongo_client = MongoDBClient()

def insert_model(user_id, model_name, model_version, tag, level, deployed, notes, model_type, framework, model_path=None, feature_list=None, performance_metrics=None, training_data_info=None):
    """
    Insert a trained model record into MongoDB
    
    Args:
        user_id (str): User identifier
        model_name (str): Name/identifier for the model
        model_version (int): Version number of the model
        tag (str): Tag for the model (e.g., 'latest', 'production')
        level (str): Model level ('hit', 'message', 'both')
        deployed (bool): Deployment status
        notes (str): Additional notes
        model_type (str): Type of model (e.g., 'xgboost', 'random_forest')
        framework (str): Framework used (e.g., 'scikit-learn', 'xgboost')
        model_path (str, optional): Path where model is saved
        feature_list (list, optional): List of features used in training
        performance_metrics (dict, optional): Performance metrics
        training_data_info (dict, optional): Information about training data
        
    Returns:
        str: MongoDB ObjectId of inserted document
    """
    global mongo_client
    if not mongo_client or mongo_client.db is None:
        logger.warning("MongoDB client not connected - returning dummy ID")
        return str(ObjectId())
    
    try:
        model_doc = {
            "user_id": user_id,
            "model_name": model_name,
            "model_version": model_version,
            "tag": tag,
            "level": level,
            "deployed": deployed,
            "notes": notes,
            "model_type": model_type,
            "framework": framework,
            "model_path": model_path,
            "feature_list": feature_list,
            "performance_metrics": mongo_client._serialize_for_mongo(performance_metrics),
            "training_data_info": training_data_info,
            "created_at": mongo_client._get_utc_timestamp(),
            "updated_at": mongo_client._get_utc_timestamp()
        }
        
        result = mongo_client.db.models.insert_one(model_doc)
        logger.info(f"Model inserted with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    except Exception as e:
        logger.error(f"Failed to insert model: {str(e)}")
        return str(ObjectId())  # Return dummy ID on failure

def insert_training_run(model_id, training_id, user_id, level, status, datasets, data_size, metrics, started_at, ended_at, duration=None):
    """
    Insert a training run record into MongoDB
    
    Args:
        model_id (str): Associated model ID
        training_id (str): Unique training run identifier
        user_id (str): User identifier
        level (str): Training level ('hit', 'message', 'both')
        status (str): Training status ('completed', 'failed', 'running')
        datasets (dict): Dataset information
        data_size (dict): Data size information
        metrics (dict): Training metrics
        started_at (datetime): Start time
        ended_at (datetime): End time
        duration (float, optional): Duration in seconds
        
    Returns:
        str: MongoDB ObjectId of inserted document
    """
    global mongo_client
    if not mongo_client or mongo_client.db is None:
        logger.warning("MongoDB client not connected - returning dummy ID")
        return str(ObjectId())
    
    try:
        training_doc = {
            "model_id": model_id,
            "training_id": training_id,
            "user_id": user_id,
            "level": level,
            "status": status,
            "datasets": datasets,
            "data_size": data_size,
            "metrics": mongo_client._serialize_for_mongo(metrics),
            "started_at": started_at,
            "ended_at": ended_at,
            "duration": duration,
            "created_at": mongo_client._get_utc_timestamp()
        }
        
        result = mongo_client.db.training_runs.insert_one(training_doc)
        logger.info(f"Training run inserted with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    except Exception as e:
        logger.error(f"Failed to insert training run: {str(e)}")
        return str(ObjectId())  # Return dummy ID on failure

def insert_validation_run(model_id, training_id, user_id, level, status, datasets, data_size, metrics, started_at, ended_at):
    """
    Insert a validation run record into MongoDB
    
    Args:
        model_id (str): Associated model ID
        training_id (str): Associated training run ID
        user_id (str): User identifier
        level (str): Validation level ('hit', 'message', 'both')
        status (str): Validation status ('completed', 'failed', 'running')
        datasets (dict): Dataset information
        data_size (dict): Data size information
        metrics (dict): Validation metrics
        started_at (datetime): Start time
        ended_at (datetime): End time
        
    Returns:
        str: MongoDB ObjectId of inserted document
    """
    global mongo_client
    if not mongo_client or mongo_client.db is None:
        logger.warning("MongoDB client not connected - returning dummy ID")
        return str(ObjectId())
    
    try:
        validation_doc = {
            "model_id": model_id,
            "training_id": training_id,
            "user_id": user_id,
            "level": level,
            "status": status,
            "datasets": datasets,
            "data_size": data_size,
            "metrics": mongo_client._serialize_for_mongo(metrics),
            "started_at": started_at,
            "ended_at": ended_at,
            "created_at": mongo_client._get_utc_timestamp()
        }
        
        result = mongo_client.db.validation_runs.insert_one(validation_doc)
        logger.info(f"Validation run inserted with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    except Exception as e:
        logger.error(f"Failed to insert validation run: {str(e)}")
        return str(ObjectId())  # Return dummy ID on failure

def get_model_by_version(model_name: str, version: int) -> Optional[Dict]:
    """
    Retrieve a model by name and version
    
    Args:
        model_name (str): Model name
        version (int): Model version
        
    Returns:
        Optional[Dict]: Model document or None
    """
    global mongo_client
    if not mongo_client or mongo_client.db is None:
        logger.warning("MongoDB client not connected")
        return None
    
    try:
        model_doc = mongo_client.db.models.find_one({
            "model_name": model_name,
            "model_version": version
        })
        return model_doc
    
    except Exception as e:
        logger.error(f"Failed to retrieve model: {str(e)}")
        return None

def generate_training_id(prefix: str = "train") -> str:
    """Generate a unique training ID with timestamp"""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ObjectId()}"

def close_connection():
    """Close MongoDB connection"""
    global mongo_client
    if mongo_client and mongo_client.client:
        mongo_client.client.close()
        logger.info("MongoDB connection closed")

if __name__ == '__main__':
    # Test MongoDB connection
    try:
        print("Testing MongoDB connection...")
        if mongo_client and mongo_client.db is not None:
            # Test database access
            collections = mongo_client.db.list_collection_names()
            print(f"✅ MongoDB connection successful")
            print(f"✅ Available collections: {collections}")
            print("✅ MongoDB utilities are working")
        else:
            print("❌ MongoDB client not properly initialized")
    except Exception as e:
        print(f"❌ MongoDB test failed: {e}")
    finally:
        close_connection()
