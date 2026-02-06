"""
MongoDB database configuration and connection management.
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from dotenv import load_dotenv
from typing import Optional
import dns.resolver

dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

# Load environment variables
load_dotenv()

# ============================================================================
# ASYNC MONGODB 
# ============================================================================

class AsyncMongoDB:
    """Async MongoDB connection manager using Motor"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.database_name = os.getenv("MONGODB_DATABASE", "firco_xgb")
        
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI environment variable is required")
    
    async def connect(self):
        """Connect to MongoDB asynchronously"""
        try:
            print(f"[ASYNC_DATABASE] Attempting to connect to MongoDB...")
            print(f"[ASYNC_DATABASE] URI format: {'mongodb+srv://' if 'mongodb+srv://' in self.mongodb_uri else 'mongodb://'}")
            print(f"[ASYNC_DATABASE] Database name: {self.database_name}")
            
            self.client = AsyncIOMotorClient(self.mongodb_uri, serverSelectionTimeoutMS=10000)
            self.database = self.client[self.database_name]
            
            # Test connection with timeout
            print("[ASYNC_DATABASE] Testing connection...")
            await self.client.admin.command('ping')
            print(f"[ASYNC_DATABASE] Successfully connected to MongoDB: {self.database_name}")
            
            # List existing databases to verify connection
            try:
                db_list = await self.client.list_database_names()
                print(f"[ASYNC_DATABASE] Available databases: {db_list}")
            except Exception as e:
                print(f"[ASYNC_DATABASE] Warning: Could not list databases: {e}")
            
            return True
            
        except Exception as e:
            print(f"[ASYNC_DATABASE] Failed to connect to MongoDB: {str(e)}")
            print(f"[ASYNC_DATABASE] Connection details:")
            print(f"[ASYNC_DATABASE] - URI format: {'mongodb+srv://' if 'mongodb+srv://' in self.mongodb_uri else 'mongodb://'}")
            print(f"[ASYNC_DATABASE] - Database: {self.database_name}")
            print(f"[ASYNC_DATABASE] - Error type: {type(e).__name__}")
            return False
    
    async def disconnect(self):
        """Disconnect from MongoDB asynchronously"""
        if self.client is not None:
            self.client.close()
            print("[ASYNC_DATABASE] Disconnected from MongoDB")
    
    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get a collection from the database"""
        if self.database is None:
            raise ValueError("Database not connected. Call await connect() first.")
        return self.database[collection_name]
    
    async def create_collections_and_indexes(self):
        """Create collections and indexes if they don't exist"""
        try:
            print("[ASYNC_DATABASE] Starting collection and index creation...")
            
            # Ensure we have a valid connection
            if self.database is None:
                if not await self.connect():
                    raise RuntimeError("Cannot connect to database")
            
            # List existing collections before creation
            existing_collections = await self.database.list_collection_names()
            print(f"[ASYNC_DATABASE] Existing collections before creation: {existing_collections}")
            
            # Models collection
            print("[ASYNC_DATABASE] Creating models collection...")
            models_collection = self.database["models"]
            await models_collection.create_index([("model_id", 1)], unique=True)
            await models_collection.create_index([("user_id", 1), ("model_name", 1)])
            await models_collection.create_index([("user_id", 1), ("tag", 1)])
            await models_collection.create_index([("status", 1)])
            await models_collection.create_index([("created_at", -1)])
            print("[ASYNC_DATABASE] Models collection and indexes created")
            
            # Training runs collection
            print("[ASYNC_DATABASE] Creating training_runs collection...")
            training_runs_collection = self.database["training_runs"]
            await training_runs_collection.create_index([("training_id", 1)], unique=True)
            await training_runs_collection.create_index([("model_id", 1)])
            await training_runs_collection.create_index([("user_id", 1)])
            await training_runs_collection.create_index([("status", 1)])
            await training_runs_collection.create_index([("created_at", -1)])
            print("[ASYNC_DATABASE] Training runs collection and indexes created")
            
            # Validation runs collection
            print("[ASYNC_DATABASE] Creating validation_runs collection...")
            validation_runs_collection = self.database["validation_runs"]
            await validation_runs_collection.create_index([("validation_id", 1)], unique=True)
            await validation_runs_collection.create_index([("training_id", 1)])
            await validation_runs_collection.create_index([("model_id", 1)])
            await validation_runs_collection.create_index([("user_id", 1)])
            await validation_runs_collection.create_index([("created_at", -1)])
            print("[ASYNC_DATABASE] Validation runs collection and indexes created")
            
            # Prediction runs collection
            print("[ASYNC_DATABASE] Creating prediction_runs collection...")
            prediction_runs_collection = self.database["prediction_runs"]
            await prediction_runs_collection.create_index([("prediction_id", 1)], unique=True)
            await prediction_runs_collection.create_index([("training_id", 1)])
            await prediction_runs_collection.create_index([("model_id", 1)])
            await prediction_runs_collection.create_index([("user_id", 1)])
            await prediction_runs_collection.create_index([("created_at", -1)])
            print("[ASYNC_DATABASE] Prediction runs collection and indexes created")
            
            # System status collection
            print("[ASYNC_DATABASE] Creating system_status collection...")
            status_collection = self.database["system_status"]
            await status_collection.create_index([("training_id", 1)], unique=True)
            await status_collection.create_index([("user_id", 1)])
            await status_collection.create_index([("status", 1)])
            await status_collection.create_index([("created_at", -1)])
            print("[ASYNC_DATABASE] System status collection and indexes created")
            
            # Verify collections were created
            final_collections = await self.database.list_collection_names()
            print(f"[ASYNC_DATABASE] Final collections after creation: {final_collections}")
            
            print("[ASYNC_DATABASE] All collections and indexes created successfully")
            
        except Exception as e:
            print(f"[ASYNC_DATABASE] Error creating collections and indexes: {str(e)}")
            print(f"[ASYNC_DATABASE] Error type: {type(e).__name__}")
            import traceback
            print(f"[ASYNC_DATABASE] Full traceback: {traceback.format_exc()}")
            raise

# Global Async MongoDB instance
async_mongodb = AsyncMongoDB()

async def get_async_database() -> AsyncIOMotorDatabase:
    """Get the async database instance"""
    if async_mongodb.database is None:
        raise ValueError("Async database not connected")
    return async_mongodb.database

def get_async_collection(collection_name: str) -> AsyncIOMotorCollection:
    """Get a specific async collection"""
    return async_mongodb.get_collection(collection_name)
