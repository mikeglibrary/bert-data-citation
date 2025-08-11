
#Utility functions for AWS S3 operations and logging setup

import boto3
import json
import time
import logging
from botocore.exceptions import ClientError, BotoCoreError
from config import BUCKET_NAME, S3_PREFIX, REQUIRED_DATA_FILES


def setup_logging():
    #Configure logging for the application
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def initialize_aws_clients():
    #Initialize SageMaker session and AWS clients
    try:
        import sagemaker
        sagemaker_session = sagemaker.Session()
        s3_client = boto3.client('s3')
        return sagemaker_session, s3_client
    except Exception as e:
        raise Exception(f"Failed to initialize AWS services: {str(e)}")


def generate_run_id():
    #Generate a unique run ID for this training job
    return time.strftime("%Y%m%d-%H%M%S")


def validate_s3_setup(s3_client, bucket=BUCKET_NAME, prefix=S3_PREFIX, logger=None):
    #Validate S3 bucket exists and required files are present
    if not logger:
        logger = setup_logging()
    
    # Check if bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"Confirmed bucket exists: {bucket}")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == '404':
            raise Exception(f"S3 bucket does not exist: {bucket}")
        elif error_code == '403':
            raise Exception(f"No permission to access bucket: {bucket}")
        else:
            raise Exception(f"Error accessing bucket: {str(e)}")
    
    # Check if required data files exist
    missing_files = []
    for file in REQUIRED_DATA_FILES:
        try:
            s3_client.head_object(Bucket=bucket, Key=f"{prefix}/data/{file}")
            logger.info(f"Found required file: {file}")
        except ClientError:
            missing_files.append(file)
    
    if missing_files:
        raise Exception(f"Required data files missing in S3: {', '.join(missing_files)}")


def save_to_s3(s3_client, data, key, bucket=BUCKET_NAME):
    #Save JSON data to S3
    try:
        s3_client.put_object(
            Body=json.dumps(data, indent=2),
            Bucket=bucket,
            Key=key
        )
        return True
    except Exception as e:
        raise Exception(f"Failed to save to S3: {str(e)}")


def get_s3_paths(run_id, bucket=BUCKET_NAME, prefix=S3_PREFIX):
    #Generate all S3 paths for a given run
    return {
        'input_data': f"s3://{bucket}/{prefix}/data/{run_id}/",
        'model_output': f"s3://{bucket}/{prefix}/models/{run_id}/",
        'hyperparams': f"{prefix}/configs/{run_id}/hyperparams.json",
        'evaluation': f"{prefix}/evaluations/{run_id}/metrics.json",
        'model_info': f"{prefix}/configs/{run_id}/model_info.json",
        'endpoint_info': f"{prefix}/configs/{run_id}/endpoint_info.json",
        'batch_output': f"s3://{bucket}/{prefix}/batch_out/{run_id}/",
        'train_data': f"s3://{bucket}/{prefix}/data/train.jsonl",
        'val_data': f"s3://{bucket}/{prefix}/data/val.jsonl",
        'test_data': f"s3://{bucket}/{prefix}/data/test.jsonl"
    }


def monitor_training_job(sm_client, job_name, logger):
    #Monitor training job progress
    status = "InProgress"
    dots = 0
    
    while status == "InProgress":
        time.sleep(10)
        
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        
        # Show progress dots
        dots = (dots + 1) % 4
        progress_dots = "." * dots
        current_time = time.strftime("%H:%M:%S")
        print(f"\rTraining in progress{progress_dots.ljust(3)} (Status: {status}) - Last check: {current_time}", end="")
    
    print("\n")  # Add a new line after progress tracking
    return status