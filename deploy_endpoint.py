
#Deploy fine-tuned/trained BERT model to SageMaker endpoint

import json
import time
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from config import AWS_ROLE, MODEL_CONFIG, INFERENCE_INSTANCE_TYPE
from utils import setup_logging, initialize_aws_clients, save_to_s3, get_s3_paths


def deploy_model_endpoint(model_data_s3_path, run_id):
    #Deploy model to SageMaker endpoint
    logger = setup_logging()
    
    try:
        # Initialize AWS services
        logger.info("Initializing SageMaker session")
        sagemaker_session, s3_client = initialize_aws_clients()
        
        # Get S3 paths
        paths = get_s3_paths(run_id)
        
        # Create HuggingFace model
        logger.info("Setting up HuggingFace model")
        huggingface_model = HuggingFaceModel(
            model_data=model_data_s3_path,
            role=AWS_ROLE,
            transformers_version=MODEL_CONFIG["transformers_version"],
            pytorch_version=MODEL_CONFIG["pytorch_version"],
            py_version=MODEL_CONFIG["py_version"],
            sagemaker_session=sagemaker_session
        )
        logger.info("Model configured successfully")
        
        # Deploy to endpoint
        logger.info("Deploying model to real-time endpoint")
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type=INFERENCE_INSTANCE_TYPE,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        logger.info(f"Endpoint deployed: {predictor.endpoint_name}")
        
        # Test the endpoint
        logger.info("Running test inference...")
        example_text = "Data for this study was taken from Fisheries and Oceans Canada."
        payload = {"inputs": example_text}
        result = predictor.predict(payload)
        logger.info(f"Test result: {result}")
        
        # Save endpoint info
        endpoint_info = {
            "endpoint_name": predictor.endpoint_name,
            "model_data": model_data_s3_path,
            "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "test_result": result
        }
        
        save_to_s3(s3_client, endpoint_info, paths['endpoint_info'])
        logger.info("Endpoint info saved to S3")
        
        return {
            "endpoint_name": predictor.endpoint_name,
            "predictor": predictor,
            "run_id": run_id
        }
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise


def deploy_from_run_id(run_id):
    #Deploy model using a specific run ID (loads model info from S3)
    logger = setup_logging()
    
    try:
        # Initialize AWS services
        sagemaker_session, s3_client = initialize_aws_clients()
        paths = get_s3_paths(run_id)
        
        # Load model info from S3
        logger.info(f"Loading model info for run ID: {run_id}")
        response = s3_client.get_object(
            Bucket=BUCKET_NAME, 
            Key=paths['model_info'].replace(f"s3://{BUCKET_NAME}/", "")
        )
        model_info = json.loads(response['Body'].read())
        
        model_data = model_info.get("model_data")
        if not model_data:
            raise ValueError(f"No model_data found in model info for run {run_id}")
        
        logger.info(f"Found model at: {model_data}")
        
        # Deploy the model
        return deploy_model_endpoint(model_data, run_id)
        
    except Exception as e:
        logger.error(f"Failed to deploy from run ID: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python deploy_endpoint.py <run_id>")
        print("   or: python deploy_endpoint.py <model_s3_path> <run_id>")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Deploy from run ID (load model info from S3)
        run_id = sys.argv[1]
        result = deploy_from_run_id(run_id)
    else:
        # Deploy from specific model path
        model_path = sys.argv[1]
        run_id = sys.argv[2]
        result = deploy_model_endpoint(model_path, run_id)
    
    print(f"\nDeployment completed successfully!")
    print(f"Endpoint name: {result['endpoint_name']}")
    print(f"Run ID: {result['run_id']}")