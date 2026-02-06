# s3_utils.py
# This module handles all interactions with AWS S3 for the Firco XGB project.
# It is designed to be a reusable utility for uploading/downloading models, data, and predictions.

import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import logging
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from a .env file for better security
load_dotenv()

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# !! SECURITY WARNING !!
# Storing credentials directly in code is highly discouraged and a major security risk.
# It is strongly recommended to use environment variables or AWS IAM roles.
# This code will prioritize environment variables if they are set.
#
# To use environment variables, create a file named `.env` in the `Firco/xgb` directory
# and add the following lines:
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_S3_BUCKET_NAME=point9ml
# AWS_S3_REGION=ap-south-1

# --- S3 Configuration ---
# IMPORTANT: Set these values in your .env file for security
# Never commit credentials to version control
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
S3_REGION = os.getenv("AWS_S3_REGION", "us-east-1")

# --- Project Directory Configuration ---
# Assumes this script is in `Firco/xgb/`
XGB_ROOT = os.path.dirname(__file__)


def get_s3_client():
    """
    Initializes and returns a Boto3 S3 client.

    Returns:
        boto3.client: An S3 client object, or None if credentials are not valid.
    """
    try:
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME, S3_REGION]):
            logging.error("S3 configuration is incomplete. Please check your .env file or hardcoded values.")
            return None

        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=S3_REGION
        )
        # Test credentials to ensure they are valid
        s3_client.list_buckets()
        logging.info("S3 client created successfully. Credentials are valid.")
        return s3_client
    except (NoCredentialsError, PartialCredentialsError):
        logging.error("AWS credentials not found. Please configure them.")
        return None
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == 'InvalidClientTokenId':
            logging.error("Invalid AWS credentials. Please check the Access Key ID.")
        elif error_code == 'SignatureDoesNotMatch':
            logging.error("Signature mismatch. Please check the Secret Access Key.")
        else:
            logging.error(f"An unexpected S3 client error occurred: {e}")
        return None


def upload_file_to_s3(local_file_path, s3_object_name=None):
    """
    Uploads a single file to the configured S3 bucket.

    Args:
        local_file_path (str): The path to the file on the local machine.
        s3_object_name (str, optional): The desired object name (key) in S3.
                                        If not provided, it will use a relative path
                                        from the XGB_ROOT directory.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    if not os.path.exists(local_file_path):
        logging.error(f"Local file not found: {local_file_path}")
        return False

    s3_client = get_s3_client()
    if not s3_client:
        return False

    if s3_object_name is None:
        s3_object_name = os.path.relpath(local_file_path, start=XGB_ROOT).replace("\\", "/")

    logging.info(f"Uploading '{local_file_path}' to bucket '{S3_BUCKET_NAME}' as '{s3_object_name}'...")

    try:
        s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_object_name)
        logging.info("Upload successful.")
        return True
    except ClientError as e:
        logging.error(f"Failed to upload file to S3: {e}")
        return False


def download_file_from_s3(s3_object_name, local_file_path=None):
    """
    Downloads a single file from the configured S3 bucket.

    Args:
        s3_object_name (str): The object name (key) of the file in S3.
        local_file_path (str, optional): The local path to save the file. If not provided,
                                         it saves to the corresponding project directory.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    s3_client = get_s3_client()
    if not s3_client:
        return False

    if local_file_path is None:
        local_file_path = os.path.join(XGB_ROOT, s3_object_name.replace("/", os.sep))

    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    logging.info(f"Downloading '{s3_object_name}' from bucket '{S3_BUCKET_NAME}' to '{local_file_path}'...")

    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_object_name, local_file_path)
        logging.info("Download successful.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logging.error(f"File not found in S3: {s3_object_name}")
        else:
            logging.error(f"Failed to download file from S3: {e}")
        return False


def list_files_in_s3(prefix=''):
    """
    Lists files in a specific 'folder' (prefix) in the S3 bucket.

    Args:
        prefix (str, optional): The prefix to filter by (e.g., 'saved_models/'). Defaults to ''.

    Returns:
        list: A list of object keys, or an empty list if an error occurs.
    """
    s3_client = get_s3_client()
    if not s3_client:
        return []

    logging.info(f"Listing files in bucket '{S3_BUCKET_NAME}' with prefix '{prefix}'...")
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        file_list = [obj['Key'] for page in pages for obj in page.get('Contents', [])]
        logging.info(f"Found {len(file_list)} files.")
        return file_list
    except ClientError as e:
        logging.error(f"Failed to list files in S3: {e}")
        return []


if __name__ == '__main__':
    # This block provides an example of how to use the utility functions.
    # You can run `python Firco/xgb/s3_utils.py` from the root directory to test it.
    logging.info("--- Running S3 Utility Self-Test ---")
    test_file_path = os.path.join(XGB_ROOT, 'uploads', 's3_test.txt')
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
    with open(test_file_path, 'w') as f:
        f.write('This is a test file for S3 upload.')
    
    if upload_file_to_s3(test_file_path):
        print("\n--- Listing files in 'uploads/' prefix ---")
        files = list_files_in_s3(prefix='uploads/')
        for file_key in files:
            print(f"- {file_key}")
        
        s3_object_key = os.path.relpath(test_file_path, start=XGB_ROOT).replace("\\", "/")
        download_path = os.path.join(XGB_ROOT, 'predictions', 'downloaded_s3_test.txt')
        if download_file_from_s3(s3_object_key, local_file_path=download_path):
            print(f"\n--- Verifying downloaded file at '{download_path}' ---")
            if os.path.exists(download_path):
                with open(download_path, 'r') as f:
                    print(f"File content: '{f.read()}'")
                os.remove(download_path)
    
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
    print("\n--- S3 Utility Self-Test Complete ---") 