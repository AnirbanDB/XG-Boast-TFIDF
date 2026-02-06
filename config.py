# Configuration file for Firco XGBoost API
# High-performance compliance prediction with hierarchical training strategy

import os

# Project structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), "firco_alerts_final_5000_7.csv")  # Changed to use full dataset
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), "archive")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "predictions")
TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "tokenizer")

# Ensure directories exist
for directory in [MODEL_SAVE_DIR, PREDICTIONS_DIR, UPLOADS_DIR, ARCHIVE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Target columns for prediction (hierarchical training order)
# CORRECTED: All 4 target columns as specified by user
# First train hit-level targets (using decision-level as features)
HIT_LEVEL_TARGETS = ["hit.review_decision", "hit.review_comments"]
# Then train decision-level targets (using hit-level as features)
MESSAGE_LEVEL_TARGETS = ["decision.last_action", "decision.reviewer_comments"]

# All target columns combined
LABEL_COLUMNS = HIT_LEVEL_TARGETS + MESSAGE_LEVEL_TARGETS

# Primary text column (most informative)
TEXT_COLUMN = "hit.matching_text"

# Smart Feature Engineering Configuration (Enhanced for MT103/MT202 handling)
FEATURE_ENGINEERING = {
    # Primary text columns (most important)
    'text_columns': [
        'hit.matching_text',
        'hit.watchlist_text',
        'mt103.sanction_list',
        'hit.review_comments',  # Add as feature when not target
        'decision.reviewer_comments'  # Add as feature when not target
    ],
    
    # Essential categorical columns (MT103/MT202 aware)
    'categorical_columns': [
        'hit.is_pep',
        'hit.hit_type',
        'hit.matching_type',
        'hit.priority',
        'hit.severity',
        'hit.country',
        'hit.mt_type',  # Critical: distinguishes MT103 vs MT202
        'mt103.country',
        'mt103.transaction_type',
        'mt202.currency',
        'mt202.instruction_code',
        'hit.review_decision',  # Add as feature when not target
        'decision.last_action'  # Add as feature when not target
    ],
    
    # Key numerical columns (MT103/MT202 specific)
    'numerical_columns': [
        'hit.score',
        'hit.fuzzy_match_score',
        'mt202.amount',  # Only available for MT202
        'mt103.hits_count_103',  # Only available for MT103
        'mt202.hits_count_202',  # Only available for MT202
        'mt202.charges'  # Only available for MT202
    ],
    
    # Smart derived features (enhanced for compliance)
    'derived_features': {
        'risk_score_combined': ['hit.score', 'hit.fuzzy_match_score'],
        'text_complexity': ['hit.matching_text', 'hit.watchlist_text'],
        'amount_risk_level': ['mt202.amount'],
        'pep_risk_indicator': ['hit.is_pep', 'hit.score'],
        'transaction_risk': ['mt103.hits_count_103', 'mt202.hits_count_202'],
        'mt_type_specific_features': ['hit.mt_type', 'mt202.amount', 'mt103.hits_count_103'],
        'compliance_patterns': ['hit.matching_type', 'hit.priority', 'hit.severity'],
        'geographic_risk': ['hit.country', 'mt103.country'],
        'institutional_risk': ['hit.Organization', 'mt103.organization', 'mt202.beneficiary_institution.name']
    }
}

# Optimized text processing (faster)
TEXT_PROCESSING = {
    'max_features': 2000,        # Reduced from 5000
    'ngram_range': (1, 2),       # Reduced from (1, 3)
    'min_df': 2,                 # Increased from 1
    'max_df': 0.9,               # Reduced from 0.95
    'sublinear_tf': True,
    'stop_words': None,
    'norm': 'l2',
    'analyzer': 'word',
    'token_pattern': r'(?u)\b\w+\b',
    'lowercase': True,
    'strip_accents': 'unicode'
}

# Optimized feature selection (faster)
FEATURE_SELECTION = {
    'variance_threshold': 0.01,   # Increased from 0.001
    'k_best_features': 1000,      # Reduced from 5000
    'svd_components': 300,        # Reduced from 800
    'correlation_threshold': 0.95  # Remove highly correlated features
}

# Faster XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,          # Reduced from 200
    'max_depth': 4,               # Reduced from 6
    'learning_rate': 0.1,         # Increased from 0.05
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 3,        # Increased from 1
    'gamma': 0.1
}

# Hierarchical training configuration (CORRECTED for user requirements)
HIERARCHICAL_TRAINING_CONFIG = {
    'stage_1_targets': HIT_LEVEL_TARGETS,  # hit.review_decision, hit.review_comments
    'stage_1_additional_features': MESSAGE_LEVEL_TARGETS,  # Use decision.* as features
    'stage_2_targets': MESSAGE_LEVEL_TARGETS,  # decision.last_action, decision.reviewer_comments  
    'stage_2_additional_features': HIT_LEVEL_TARGETS,  # Use hit.* as features
    'cross_validation_folds': 5,
    'early_stopping_rounds': 75,
    'validation_fraction': 0.2,
    'handle_mt_type_split': True,  # Handle MT103/MT202 differences
    'feature_selection_per_stage': True  # Different feature selection per stage
}

# Model training configuration
TRAINING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.15,
    'random_state': 42,
    'cv_folds': 5,
    'early_stopping_rounds': 75,
    'verbose': 1,
    'stratify': True,
    'shuffle': True
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 3004,  # Changed port to 3004 as per new requirement
    'debug': False,
    'title': 'Firco XGBoost Compliance Predictor API',
    'description': 'High-performance XGBoost API for Firco compliance prediction with hierarchical training'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(CURRENT_DIR, 'firco_xgb_api.log')
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'target_f1_score': 0.75,      # Higher target for Firco
    'min_accuracy': 0.70,
    'max_training_time': 2400,    # 40 minutes
    'max_prediction_time': 600    # 10 minutes
}

# Class imbalance handling
CLASS_BALANCE_CONFIG = {
    'use_class_weights': True,
    'use_smote': False,           # XGBoost handles imbalance well
    'use_sample_weights': True,
    'balance_strategy': 'auto'
}

# Model versioning configuration
VERSIONING_CONFIG = {
    'model_prefix': 'firco_xgb_model',
    'version_format': 'v{version}',
    'max_archived_versions': 10,
    'auto_cleanup': True
}

# Data validation configuration
DATA_VALIDATION = {
    'required_columns': ['hit.matching_text'] + LABEL_COLUMNS,
    'min_rows': 1,  # Changed from 100 to 1 to allow small test files
    'max_missing_ratio': 0.95,
    'encoding': 'utf-8'
}

# Memory and performance optimization
OPTIMIZATION_CONFIG = {
    'use_sparse_matrices': True,
    'batch_size': 1000,
    'memory_limit': '8GB',
    'parallel_jobs': -1,
    'cache_features': True
} 