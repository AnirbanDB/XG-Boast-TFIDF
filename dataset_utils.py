# Dataset utilities for Firco XGBoost API
# Comprehensive data processing with hierarchical training support

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any, Union
import io
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Local imports
from config import (
    LABEL_COLUMNS, HIT_LEVEL_TARGETS, MESSAGE_LEVEL_TARGETS,
    FEATURE_ENGINEERING, TRAINING_CONFIG, DATA_VALIDATION,
    HIERARCHICAL_TRAINING_CONFIG
)

def load_and_preprocess_data(
    file_path_or_buffer: Union[str, io.BytesIO],
    is_training: bool = True
) -> pd.DataFrame:
    """
    Load and preprocess Firco dataset with comprehensive cleaning and validation.
    
    Args:
        file_path_or_buffer: Path to CSV file or file buffer
        is_training: Whether this is training data (affects validation)
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed dataframe
    """
    try:
        # Load the dataset
        if isinstance(file_path_or_buffer, str):
            df = pd.read_csv(file_path_or_buffer, encoding='utf-8')
        else:
            df = pd.read_csv(file_path_or_buffer, encoding='utf-8')
            
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Basic data validation
        if df.empty:
            raise ValueError("Dataset is empty")
            
        if is_training and len(df) < DATA_VALIDATION['min_rows']:
            raise ValueError(f"Dataset has {len(df)} rows, minimum required for training: {DATA_VALIDATION['min_rows']}")
        
        # Data cleaning and preprocessing
        df = _clean_and_preprocess_data(df, is_training)
        
        # Ensure consistent column order and presence
        df = _ensure_consistent_columns(df, is_training)
        
        print(f"Preprocessing complete. Final shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {str(e)}")
        raise

def _ensure_consistent_columns(df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
    """
    Ensure consistent columns between training and prediction data.
    
    Args:
        df: Input dataframe
        is_training: Whether this is training data
        
    Returns:
        pd.DataFrame: Dataframe with consistent columns
    """
    # Define expected columns based on the original dataset structure
    expected_columns = [
        'decision.decision', 'decision.last_action', 'decision.options', 
        'decision.reviewer_comments', 'hit.Organization', 'hit.account_number',
        'hit.beneficiary', 'hit.bic_code', 'hit.city', 'hit.country',
        'hit.fuzzy_match_score', 'hit.hit_id', 'hit.hit_type', 'hit.is_pep',
        'hit.matching_text', 'hit.matching_type', 'hit.mt_type', 'hit.priority',
        'hit.review_comments', 'hit.review_decision', 'hit.score', 'hit.sender',
        'hit.severity', 'hit.state', 'hit.tag', 'hit.watchlist_text',
        'message_id', 'mt103.account_number', 'mt103.beneficiary', 'mt103.bic_code',
        'mt103.city', 'mt103.country', 'mt103.dob', 'mt103.hits_count_103',
        'mt103.mt_type', 'mt103.national_id', 'mt103.organization', 'mt103.origin',
        'mt103.passport', 'mt103.reference', 'mt103.sanction_list', 'mt103.sender',
        'mt103.state', 'mt103.street', 'mt103.synonyms.city', 'mt103.synonyms.country',
        'mt103.synonyms.name', 'mt103.synonyms.organization', 'mt103.transaction_reference_number',
        'mt103.transaction_type', 'mt202.amount', 'mt202.beneficiary_institution.address.city',
        'mt202.beneficiary_institution.address.country', 'mt202.beneficiary_institution.address.street',
        'mt202.beneficiary_institution.bic', 'mt202.beneficiary_institution.name',
        'mt202.charges', 'mt202.currency', 'mt202.hits_count_202', 'mt202.instruction_code',
        'mt202.intermediary_institution.bic', 'mt202.intermediary_institution.name',
        'mt202.mt_type', 'mt202.ordering_institution.address.city',
        'mt202.ordering_institution.address.country', 'mt202.ordering_institution.address.street',
        'mt202.ordering_institution.bic', 'mt202.ordering_institution.name',
        'mt202.related_reference', 'mt202.sender_to_receiver_info',
        'mt202.transaction_reference_number', 'mt202.value_date'
    ]
    
    # Add missing columns with default values
    for col in expected_columns:
        if col not in df.columns:
            if 'amount' in col.lower() or 'count' in col.lower() or 'score' in col.lower():
                df[col] = 0
            elif 'id' in col.lower():
                df[col] = range(len(df))
            else:
                df[col] = ""
    
    # Ensure columns are in consistent order
    available_expected = [col for col in expected_columns if col in df.columns]
    other_columns = [col for col in df.columns if col not in expected_columns]
    
    # Reorder columns: expected first, then others
    df = df[available_expected + other_columns]
    
    return df

def _clean_and_preprocess_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Clean and preprocess the dataset with smart feature engineering.
    
    Args:
        df: Raw dataframe
        is_training: Whether this is training data
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed dataframe
    """
    print("Performing comprehensive data cleaning...")
    
    # 1. Basic cleaning
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Clean text columns
    text_columns = [col for col in FEATURE_ENGINEERING['text_columns'] if col in df.columns]
    for col in text_columns:
        df[col] = df[col].astype(str).fillna("")
        df[col] = df[col].str.strip()
        df[col] = df[col].replace(['nan', 'NaN', 'null', 'NULL', 'None'], "")
    
    # 2. Handle categorical columns
    categorical_columns = [col for col in FEATURE_ENGINEERING['categorical_columns'] if col in df.columns]
    for col in categorical_columns:
        df[col] = df[col].astype(str).fillna("Unknown")
        df[col] = df[col].str.strip().str.title()
        # Normalize common variations
        df[col] = df[col].replace({
            'Yes': 'YES', 'No': 'NO', 'True': 'YES', 'False': 'NO',
            'Y': 'YES', 'N': 'NO', '1': 'YES', '0': 'NO',
            'nan': 'Unknown', 'NaN': 'Unknown', 'null': 'Unknown', 'NULL': 'Unknown'
        })
    
    # 3. Handle numerical columns with proper type conversion
    numerical_columns = [col for col in FEATURE_ENGINEERING['numerical_columns'] if col in df.columns]
    for col in numerical_columns:
        # Convert to string first, then handle empty/null values
        df[col] = df[col].astype(str)
        # Replace empty strings and null values with 0
        df[col] = df[col].replace(['', 'nan', 'NaN', 'null', 'NULL', 'None'], '0')
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill any remaining NaN with 0
        df[col] = df[col].fillna(0)
        
        # Handle outliers using IQR method only if we have valid data
        if df[col].nunique() > 1:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:  # Only apply outlier removal if IQR is valid
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 4. Handle any remaining non-numeric columns that should be numeric
    for col in df.columns:
        if col not in text_columns and col not in categorical_columns:
            # Check if column contains mostly numeric data
            sample_values = df[col].dropna().astype(str).head(100)
            if len(sample_values) > 0:
                # Count how many values look numeric
                numeric_count = sum(1 for val in sample_values if val.replace('.', '').replace('-', '').isdigit())
                if numeric_count > len(sample_values) * 0.8:  # If 80% look numeric
                    df[col] = df[col].astype(str).replace(['', 'nan', 'NaN', 'null', 'NULL', 'None'], '0')
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 5. Final cleanup
    # Remove columns with all missing values
    df = df.dropna(axis=1, how='all')
    
    # Ensure required columns exist
    required_columns = LABEL_COLUMNS + FEATURE_ENGINEERING['text_columns'][:1]  # At least primary text column
    available_required = [col for col in required_columns if col in df.columns]
    
    if len(available_required) < len(LABEL_COLUMNS) + 1:
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"Warning: Missing required columns: {missing_cols}")
    
    print(f"Data cleaning completed. Final shape: {df.shape}")
    return df

def _create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced engineered features for better model performance.
    
    Args:
        df: Preprocessed dataframe
        
    Returns:
        pd.DataFrame: Dataframe with additional engineered features
    """
    print("Creating advanced engineered features...")
    
    # 1. Risk scoring features
    if 'hit.score' in df.columns and 'hit.fuzzy_match_score' in df.columns:
        df['risk_score_combined'] = (df['hit.score'] * 0.6 + df['hit.fuzzy_match_score'] * 0.4)
        df['risk_score_max'] = df[['hit.score', 'hit.fuzzy_match_score']].max(axis=1)
        df['risk_score_diff'] = abs(df['hit.score'] - df['hit.fuzzy_match_score'])
    
    # 2. Text length and complexity features
    text_columns = [col for col in FEATURE_ENGINEERING['text_columns'] if col in df.columns]
    for col in text_columns:
        df[f'{col}_length'] = df[col].str.len()
        df[f'{col}_word_count'] = df[col].str.split().str.len()
        df[f'{col}_unique_words'] = df[col].apply(lambda x: len(set(str(x).split())))
        df[f'{col}_avg_word_length'] = df[col].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0)
    
    # 3. Combined text features
    primary_text_cols = ['hit.matching_text', 'hit.watchlist_text', 'decision.reviewer_comments']
    available_text_cols = [col for col in primary_text_cols if col in df.columns]
    if available_text_cols:
        df['combined_primary_text'] = df[available_text_cols].fillna("").astype(str).agg(' | '.join, axis=1)
    
    # 4. Institution and entity features
    institution_cols = ['hit.Organization', 'mt103.organization', 'mt202.beneficiary_institution.name']
    available_inst_cols = [col for col in institution_cols if col in df.columns]
    if available_inst_cols:
        df['combined_institutions'] = df[available_inst_cols].fillna("").astype(str).agg(' | '.join, axis=1)
    
    # 5. Geographic features
    geo_cols = ['hit.country', 'hit.city', 'mt103.country', 'mt103.city']
    available_geo_cols = [col for col in geo_cols if col in df.columns]
    if available_geo_cols:
        df['combined_geography'] = df[available_geo_cols].fillna("").astype(str).agg(' | '.join, axis=1)
    
    # 6. Transaction pattern features
    if 'mt103.hits_count_103' in df.columns:
        df['high_hit_count_103'] = (df['mt103.hits_count_103'] > df['mt103.hits_count_103'].quantile(0.75)).astype(int)
    
    if 'mt202.hits_count_202' in df.columns:
        df['high_hit_count_202'] = (df['mt202.hits_count_202'] > df['mt202.hits_count_202'].quantile(0.75)).astype(int)
    
    # 7. PEP and risk indicators
    if 'hit.is_pep' in df.columns:
        df['is_pep_binary'] = (df['hit.is_pep'].str.upper() == 'YES').astype(int)
    
    # 8. Severity and priority scoring
    if 'hit.severity' in df.columns:
        severity_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        df['severity_score'] = df['hit.severity'].map(severity_map).fillna(0)
    
    if 'hit.priority' in df.columns:
        priority_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        df['priority_score'] = df['hit.priority'].map(priority_map).fillna(0)
    
    # 9. Amount-based features
    if 'mt202.amount' in df.columns:
        df['amount_log'] = np.log1p(df['mt202.amount'])
        df['high_amount'] = (df['mt202.amount'] > df['mt202.amount'].quantile(0.9)).astype(int)
    
    print("Advanced feature engineering complete.")
    return df

def create_label_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """
    Create label encoders for all target columns with special handling for high-cardinality columns.
    
    Args:
        df: Dataframe containing target columns
        
    Returns:
        Dict[str, LabelEncoder]: Dictionary of label encoders
    """
    print("Creating label encoders for target columns...")
    
    label_encoders = {}
    
    for col in LABEL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            
            # Handle high-cardinality comment columns
            if 'comment' in col.lower():
                # For comment columns, group similar comments or use top N categories
                values = df[col].fillna("No Comment").astype(str)
                value_counts = values.value_counts()
                
                # Keep only top 100 most frequent comments, group others as "Other"
                top_comments = value_counts.head(100).index.tolist()
                processed_values = values.apply(lambda x: x if x in top_comments else "Other")
                
                le.fit(processed_values)
                print(f"  {col}: {len(le.classes_)} classes (reduced from {values.nunique()} unique comments)")
            else:
                # Regular categorical encoding
                values = df[col].astype(str).fillna("Unknown")
                le.fit(values)
                print(f"  {col}: {len(le.classes_)} classes - {list(le.classes_)}")
            
            label_encoders[col] = le
        else:
            print(f"  Warning: Target column '{col}' not found in dataset")
    
    return label_encoders

def encode_labels(df: pd.DataFrame, label_encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Encode target labels using pre-fitted encoders.
    
    Args:
        df: Dataframe with target columns
        label_encoders: Dictionary of fitted label encoders
        
    Returns:
        pd.DataFrame: Dataframe with encoded labels
    """
    encoded_df = df.copy()
    
    for col, encoder in label_encoders.items():
        if col in encoded_df.columns:
            # Handle high-cardinality comment columns
            if 'comment' in col.lower():
                # Apply the same grouping logic as in create_label_encoders
                values = encoded_df[col].fillna("No Comment").astype(str)
                
                # For comments, we need to map unseen values to "Other"
                processed_values = []
                for val in values:
                    if val in encoder.classes_:
                        processed_values.append(val)
                    else:
                        processed_values.append("Other")
                
                # Transform using the processed values
                encoded_values = encoder.transform(processed_values)
            else:
                # Regular categorical encoding
                values = encoded_df[col].astype(str).fillna("Unknown")
                
                # Handle unseen labels
                encoded_values = []
                for val in values:
                    if val in encoder.classes_:
                        encoded_values.append(encoder.transform([val])[0])
                    else:
                        encoded_values.append(0)  # Map to first class
                encoded_values = np.array(encoded_values)
            
            encoded_df[col] = encoded_values
    
    return encoded_df

def split_data_hierarchical(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    validation_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        df: Input dataframe
        test_size: Test set proportion
        validation_size: Validation set proportion
        random_state: Random state for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, validation, test dataframes
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[LABEL_COLUMNS[0]] if LABEL_COLUMNS[0] in df.columns else None
    )
    
    # Second split: train vs validation
    # Adjust validation size relative to remaining data
    adjusted_val_size = validation_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val_df[LABEL_COLUMNS[0]] if LABEL_COLUMNS[0] in train_val_df.columns else None
    )
    
    return train_df, val_df, test_df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train/test split for backward compatibility.
    
    Args:
        df: Full dataset
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test dataframes
    """
    return train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[LABEL_COLUMNS[0]] if LABEL_COLUMNS[0] in df.columns else None
    )

def prepare_hierarchical_features(
    df: pd.DataFrame, 
    stage: str = "hit_level"
) -> pd.DataFrame:
    """
    Prepare features for hierarchical training.
    
    Args:
        df: Input dataframe
        stage: Either "hit_level" or "message_level"
        
    Returns:
        pd.DataFrame: Dataframe with appropriate features for the training stage
    """
    print(f"Preparing features for {stage} training...")
    
    prepared_df = df.copy()
    
    if stage == "hit_level":
        # For hit-level training, we can use message-level targets as features
        # (if they exist and are not the current targets)
        feature_cols = [col for col in MESSAGE_LEVEL_TARGETS if col in prepared_df.columns]
        if feature_cols:
            print(f"  Using message-level features: {feature_cols}")
    
    elif stage == "message_level":
        # For message-level training, we can use hit-level targets as features
        # (if they exist and are not the current targets)
        feature_cols = [col for col in HIT_LEVEL_TARGETS if col in prepared_df.columns]
        if feature_cols:
            print(f"  Using hit-level features: {feature_cols}")
    
    return prepared_df

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        df: Dataframe to validate
        
    Returns:
        Dict[str, Any]: Quality metrics and validation results
    """
    print("Validating data quality...")
    
    quality_metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_ratio': df.isnull().sum().sum() / (len(df) * len(df.columns)),
        'duplicate_rows': df.duplicated().sum(),
        'empty_text_columns': {},
        'target_distribution': {},
        'feature_coverage': {}
    }
    
    # Check text columns
    text_columns = [col for col in FEATURE_ENGINEERING['text_columns'] if col in df.columns]
    for col in text_columns:
        empty_ratio = (df[col].str.len() == 0).sum() / len(df)
        quality_metrics['empty_text_columns'][col] = empty_ratio
    
    # Check target distribution
    for col in LABEL_COLUMNS:
        if col in df.columns:
            quality_metrics['target_distribution'][col] = df[col].value_counts().to_dict()
    
    # Check feature coverage
    for feature_type, columns in FEATURE_ENGINEERING.items():
        if isinstance(columns, list):
            available = [col for col in columns if col in df.columns]
            coverage = len(available) / len(columns) if columns else 0
            quality_metrics['feature_coverage'][feature_type] = coverage
    
    print(f"Data quality validation complete. Missing ratio: {quality_metrics['missing_ratio']:.3f}")
    return quality_metrics

def get_feature_columns(df: pd.DataFrame, exclude_targets: bool = True) -> List[str]:
    """
    Get list of feature columns, optionally excluding target columns.
    
    Args:
        df: Input dataframe
        exclude_targets: Whether to exclude target columns
        
    Returns:
        List[str]: List of feature column names
    """
    all_columns = list(df.columns)
    
    if exclude_targets:
        feature_columns = [col for col in all_columns if col not in LABEL_COLUMNS]
    else:
        feature_columns = all_columns
    
    return feature_columns 