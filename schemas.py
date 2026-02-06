"""
Pydantic models for MongoDB collections and Endpoint validation.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List, Annotated
from datetime import datetime
from bson import ObjectId
from enum import Enum
import uuid

# Pydantic v2 compatible ObjectId type
def validate_object_id(v):
    if isinstance(v, ObjectId):
        return v
    if isinstance(v, str):
        if ObjectId.is_valid(v):
            return ObjectId(v)
    raise ValueError('Invalid ObjectId')

# Use Annotated for Pydantic v2
PyObjectId = Annotated[ObjectId, Field(validation_alias='_id')]

# Pydantic validation schemas for JSON input
class MT103Schema(BaseModel):
    mt_type: str = Field(..., description="MT type (103)")
    transaction_reference_number: str = Field(..., description="Transaction reference number")
    sender: str = Field(..., description="Sender name")
    beneficiary: str = Field(..., description="Beneficiary name")
    origin: str = Field(..., description="Origin country")
    country: str = Field(..., description="Country")
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State")
    organization: str = Field(..., description="Organization name")
    bic_code: str = Field(..., description="BIC code")
    transaction_type: str = Field(..., description="Transaction type")
    reference: str = Field(..., description="Reference")
    dob: str = Field(..., description="Date of birth")
    national_id: str = Field(..., description="National ID")
    passport: str = Field(..., description="Passport number")
    account_number: str = Field(..., description="Account number")
    sanction_list: str = Field(..., description="Sanction list")
    synonyms: Dict[str, List[str]] = Field(..., description="Synonyms for various fields")
    hits_count_103: str = Field(..., description="Hits count for MT103")

class MT202Schema(BaseModel):
    mt_type: str = Field(..., description="MT type (202)")
    transaction_reference_number: str = Field(..., description="Transaction reference number")
    related_reference: str = Field(..., description="Related reference")
    value_date: str = Field(..., description="Value date")
    currency: str = Field(..., description="Currency")
    amount: float = Field(..., description="Amount")
    ordering_institution: Dict[str, Any] = Field(..., description="Ordering institution details")
    beneficiary_institution: Dict[str, Any] = Field(..., description="Beneficiary institution details")
    sender_to_receiver_info: str = Field(..., description="Sender to receiver info")
    intermediary_institution: Dict[str, Any] = Field(..., description="Intermediary institution details")
    charges: str = Field(..., description="Charges")
    instruction_code: str = Field(..., description="Instruction code")
    hits_count_202: str = Field(..., description="Hits count for MT202")

class MessageSchema(BaseModel):
    MT103: MT103Schema = Field(..., description="MT103 message details")
    MT202: MT202Schema = Field(..., description="MT202 message details")

class BlockingHitSchema(BaseModel):
    hit_id: str = Field(..., description="Hit ID")
    mt_type: str = Field(..., description="MT type")
    tag: Optional[str] = Field(None, description="Tag")
    is_pep: str = Field(..., description="Is PEP")
    severity: str = Field(..., description="Severity")
    fuzzy_match_score: float = Field(..., description="Fuzzy match score")
    sender: str = Field(..., description="Sender")
    Organization: str = Field(..., description="Organization")
    beneficiary: str = Field(..., description="Beneficiary")
    bic_code: str = Field(..., description="BIC code")
    account_number: str = Field(..., description="Account number")
    country: str = Field(..., description="Country")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State")
    score: int = Field(..., description="Score")
    priority: int = Field(..., description="Priority")
    matching_type: str = Field(..., description="Matching type")
    hit_type: str = Field(..., description="Hit type")
    matching_text: str = Field(..., description="Matching text")
    watchlist_text: List[str] = Field(..., description="Watchlist text")
    review_decision: str = Field(..., description="Review decision")
    review_comments: str = Field(..., description="Review comments")

class MessageDecisionSchema(BaseModel):
    decision: str = Field(..., description="Decision")
    options: List[str] = Field(..., description="Options")
    reviewer_comments: str = Field(..., description="Reviewer comments")
    last_action: str = Field(..., description="Last action")

class FircoAlertSchema(BaseModel):
    message_id: str = Field(..., description="Message ID")
    message: MessageSchema = Field(..., description="Message details")
    total_hits: int = Field(..., description="Total hits")
    blocking_hits: List[BlockingHitSchema] = Field(..., description="Blocking hits")
    message_decision: MessageDecisionSchema = Field(..., description="Message decision")

# Supported model types enum
class ModelType(str, Enum):
    FIRCO_BERT_XGB = "firco_bert_xgb"
    FIRCO_DEBERTA_XGB = "firco_deberta_xgb"
    FIRCO_ROBERTA_XGB = "firco_roberta_xgb"
    FIRCO_LGBM = "firco_lgbm"
    FIRCO_LOGREG = "firco_logreg"
    FIRCO_ENSEMBLE = "firco_ensemble"
    FIRCO_XGB_TFIDF = "firco_xgb_tfidf"  # Your XGBoost + TF-IDF model

class PredictionRequestSchema(BaseModel):
    data: FircoAlertSchema = Field(..., description="Single FIRCO alert data")
    level: str = Field("hit", description="Prediction level (hit or message)")
    training_id: str = Field(..., description="Training ID for the model")
    model_type: ModelType = Field(ModelType.FIRCO_XGB_TFIDF, description="Type of model to use")
    user_id: Optional[str] = Field(None, description="User ID (optional)")

class TrainingConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

class ModelTypeSchema(BaseModel):
    model_type: ModelType = Field(..., description="Type of model to use")
    
    class Config:
        use_enum_values = True

class TrainingResponse(BaseModel):
    message: str
    training_id: str
    status: str
    download_url: Optional[str] = None

class ValidationResponse(BaseModel):
    message: str
    metrics: Dict[str, Any]
    accuracy: float

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]

class BatchPredictionResponse(BaseModel):
    message: str
    predictions: List[Dict[str, Any]]  # Each prediction now includes 'probability' field

# Model Collection Schema
class ModelCreate(BaseModel):
    model_id: str  # Added unique model_id field
    user_id: str
    model_name: str
    model_version: int
    tag: str = "latest"
    level: str  # "hit", "message", "both"
    model_type: str  # "xgboost", "bert", "roberta", "deberta", "lgbm", "logreg"
    framework: str  # "scikit-learn", "pytorch", "tensorflow", "xgboost"
    notes: Optional[str] = None

class ModelInDB(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[str] = Field(default=None, alias="_id")
    model_id: str  # Added unique model_id field
    user_id: str
    model_name: str
    model_version: int
    tag: str
    status: str = "active"  # "active", "archived", "deleted"
    level: str  # "hit", "message", "both"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deployed: bool = False
    notes: Optional[str] = None
    model_type: str  # "xgboost", "bert", "roberta", "deberta", "lgbm", "logreg"
    framework: str  # "scikit-learn", "pytorch", "tensorflow", "xgboost"
    s3_path: Optional[str] = None # <-- ADD THIS LINE

# Training Run Collection Schema
class DatasetInfo(BaseModel):
    train: Optional[str] = None
    validation: Optional[str] = None
    test: Optional[str] = None
    s3_key: Optional[str] = None

class DataSize(BaseModel):
    total_records: Optional[int] = None
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    validation: Optional[int] = None

class TrainingConfigSchema(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    class_weights: Optional[Dict[str, float]] = None

class ModelArtifacts(BaseModel):
    model_url: Optional[str] = None
    feature_encoder_url: Optional[str] = None
    label_encoder_url: Optional[str] = None
    s3_keys: Optional[Dict[str, str]] = None

class TrainingMetadata(BaseModel):
    training_time_sec: Optional[float] = None
    device: Optional[str] = "CPU"
    framework_version: Optional[str] = None
    python_version: Optional[str] = None

class TrainingRunCreate(BaseModel):
    model_id: str
    training_id: str
    user_id: str
    level: str  # "hit", "message", "both"
    datasets: Optional[DatasetInfo] = None
    training_config: Optional[TrainingConfigSchema] = None

class TrainingRunInDB(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[str] = Field(default=None, alias="_id")
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Added UUID
    model_id: str
    training_id: str
    user_id: str
    type: str = "training"
    level: str  # "hit", "message", "both"
    status: str = "started"  # "started", "training", "completed", "failed", "idle"
    datasets: Optional[DatasetInfo] = None
    data_size: Optional[DataSize] = None
    training_config: Optional[TrainingConfigSchema] = None
    model_artifacts: Optional[ModelArtifacts] = None
    metadata: Optional[TrainingMetadata] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Validation Run Collection Schema
class ValidationMetrics(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    support: Optional[int] = None
    classification_report: Optional[Dict[str, Any]] = None
    confusion_matrix: Optional[List[List[int]]] = None

class ValidationRunCreate(BaseModel):
    validation_id: str  # Added validation_id field
    model_id: str
    training_id: str
    user_id: str
    level: str
    datasets: Optional[DatasetInfo] = None

class ValidationRunInDB(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[str] = Field(default=None, alias="_id")
    validation_id: str  # Added validation_id field
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Added UUID
    model_id: str
    training_id: str
    user_id: str
    type: str = "validation"
    level: str  # "hit", "message", "both"
    status: str = "started"  # "started", "validating", "completed", "failed", "idle"
    datasets: Optional[DatasetInfo] = None
    data_size: Optional[DataSize] = None
    metrics: Optional[ValidationMetrics] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Prediction Run Collection Schema
class InputData(BaseModel):
    source_type: str  # "csv" or "json"
    s3_key: Optional[str] = None
    record_count: Optional[int] = None

class PredictionResults(BaseModel):
    total_predictions: Optional[int] = None
    hit_predictions: Optional[int] = None
    message_predictions: Optional[int] = None
    results_s3_key: Optional[str] = None

class Performance(BaseModel):
    processing_time_sec: Optional[float] = None
    predictions_per_second: Optional[float] = None

class PredictionRunCreate(BaseModel):
    prediction_id: str  # Added prediction_id field
    model_id: str
    training_id: str
    user_id: str
    level: str  # "hit", "message", "both"
    input_data: Optional[InputData] = None

class PredictionRunInDB(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[str] = Field(default=None, alias="_id")
    prediction_id: str  # Added prediction_id field
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Added UUID
    model_id: str
    training_id: str
    user_id: str
    type: str = "prediction"
    level: str  # "hit", "message", "both"
    status: str = "started"  # "started", "predicting", "completed", "failed", "idle"
    input_data: Optional[InputData] = None
    predictions: Optional[PredictionResults] = None
    performance: Optional[Performance] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# System Status Collection Schema
class SystemStatusCreate(BaseModel):
    training_id: str
    user_id: str

class SystemStatusInDB(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[str] = Field(default=None, alias="_id")
    training_id: str
    user_id: str
    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    status: str = "idle"  # "idle", "started", "training", "completed", "failed"
    progress_percentage: int = 0
    estimated_completion: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
