
#Train BERT model for citation classification using SageMaker

import json
import time
import boto3
from sagemaker.huggingface import HuggingFace

from config import (
    AWS_ROLE, MODEL_CONFIG, TRAINING_PARAMS, 
    TRAINING_INSTANCE_TYPE, BUCKET_NAME, S3_PREFIX
)
from utils import (
    setup_logging, initialize_aws_clients, generate_run_id,
    validate_s3_setup, save_to_s3, get_s3_paths, monitor_training_job
)


def train_bert_model():
    #Main function to train BERT model on SageMaker
    logger = setup_logging()
    
    try:
        # Initialize AWS services
        logger.info("Initializing SageMaker session and AWS clients")
        sagemaker_session, s3_client = initialize_aws_clients()
        
        # Generate unique run ID
        run_id = generate_run_id()
        logger.info(f"Run ID for this training job: {run_id}")
        
        # Validate S3 setup
        validate_s3_setup(s3_client, logger=logger)
        
        # Get S3 paths
        paths = get_s3_paths(run_id)
        
        # Prepare hyperparameters
        hyperparams = {
            **MODEL_CONFIG,
            **TRAINING_PARAMS,
            "run_id": run_id
        }
        
        # Save hyperparameters to S3
        save_to_s3(s3_client, hyperparams, paths['hyperparams'])
        logger.info("Saved hyperparameters to S3")
        
        # Set up HuggingFace estimator
        logger.info("Setting up HuggingFace estimator...")
        estimator = HuggingFace(
            entry_point="train.py",
            source_dir="src/",
            instance_type=TRAINING_INSTANCE_TYPE,
            instance_count=1,
            role=AWS_ROLE,
            transformers_version=MODEL_CONFIG["transformers_version"],
            pytorch_version=MODEL_CONFIG["pytorch_version"],
            py_version=MODEL_CONFIG["py_version"],
            hyperparameters=hyperparams,
            output_path=paths['model_output'],
            input_mode="File",
            sagemaker_session=sagemaker_session
        )
        logger.info("Estimator configured successfully")
        
        # Start training
        logger.info("Starting model training")
        estimator.fit({
            "train": paths['train_data'],
            "validation": paths['val_data']
        }, wait=False)
        
        # Monitor training progress
        job_name = estimator.latest_training_job.name
        logger.info(f"Training job started: {job_name}")
        
        sm_client = boto3.client('sagemaker')
        status = monitor_training_job(sm_client, job_name, logger)
        
        # Wait for final completion
        if status != "Completed":
            logger.info(f"Waiting for final job completion. Current status: {status}")
            estimator.latest_training_job.wait()
        
        logger.info("Training complete.")
        
        # Save model info
        model_info = {
            "model_data": estimator.model_data,
            "training_job_name": estimator._current_job_name,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "s3_model_path": estimator.model_data
        }
        
        save_to_s3(s3_client, model_info, paths['model_info'])
        logger.info("Model info saved to S3")
        
        # Return important information
        return {
            "run_id": run_id,
            "model_data": estimator.model_data,
            "training_job_name": estimator._current_job_name,
            "estimator": estimator
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    result = train_bert_model()
    print(f"\nTraining completed successfully")
    print(f"Run ID: {result['run_id']}")
    print(f"Model S3 location: {result['model_data']}")
    print(f"Training job: {result['training_job_name']}")