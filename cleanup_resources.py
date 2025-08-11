#CLEAN UP AWS SAGEMAKER RESOURCES

import boto3
import json
from utils import setup_logging, initialize_aws_clients, get_s3_paths
from config import BUCKET_NAME


def delete_endpoint(endpoint_name):
    #Delete endpoint
    logger = setup_logging()
    
    try:
        sm_client = boto3.client('sagemaker')
        
        logger.info(f"🗑️ Deleting endpoint: {endpoint_name}")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"✅ Endpoint {endpoint_name} deleted successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to delete endpoint: {str(e)}")
        raise


def delete_endpoint_config(endpoint_config_name):
    #Delete endpoint config
    logger = setup_logging()
    
    try:
        sm_client = boto3.client('sagemaker')
        
        logger.info(f"🗑️ Deleting endpoint config: {endpoint_config_name}")
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        logger.info(f"✅ Endpoint config {endpoint_config_name} deleted successfully")
        
    except Exception as e:
        logger.error(f"Failed to delete endpoint config: {str(e)}")
        raise


def delete_model(model_name):
    #Delete model
    logger = setup_logging()
    
    try:
        sm_client = boto3.client('sagemaker')
        
        logger.info(f"Deleting model: {model_name}")
        sm_client.delete_model(ModelName=model_name)
        logger.info(f"Model {model_name} deleted successfully")
        
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        raise


def cleanup_run_resources(run_id):
    #Clean up resources
    logger = setup_logging()
    
    try:
        sagemaker_session, s3_client = initialize_aws_clients()
        sm_client = boto3.client('sagemaker')
        paths = get_s3_paths(run_id)
        
        # Try to get endpoint info from S3
        try:
            response = s3_client.get_object(
                Bucket=BUCKET_NAME,
                Key=paths['endpoint_info'].replace(f"s3://{BUCKET_NAME}/", "")
            )
            endpoint_info = json.loads(response['Body'].read())
            endpoint_name = endpoint_info.get("endpoint_name")
            
            if endpoint_name:
                logger.info(f"Found endpoint to clean up: {endpoint_name}")
                
                # Get endpoint details to find associated resources
                try:
                    endpoint_details = sm_client.describe_endpoint(EndpointName=endpoint_name)
                    endpoint_config_name = endpoint_details['EndpointConfigName']
                    
                    # Get endpoint config to find model name
                    config_details = sm_client.describe_endpoint_config(
                        EndpointConfigName=endpoint_config_name
                    )
                    model_name = config_details['ProductionVariants'][0]['ModelName']
                    
                    # Delete resources in order
                    delete_endpoint(endpoint_name)
                    delete_endpoint_config(endpoint_config_name)
                    delete_model(model_name)
                    
                except Exception as e:
                    logger.warning(f"Some resources may have already been deleted: {str(e)}")
            
        except Exception as e:
            logger.warning(f"Could not find endpoint info for run {run_id}: {str(e)}")
        
        logger.info(f"Cleanup completed for run {run_id}")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise


def list_all_endpoints():
   #List all sagemaker endpoints
    logger = setup_logging()
    
    try:
        sm_client = boto3.client('sagemaker')
        
        response = sm_client.list_endpoints()
        endpoints = response['Endpoints']
        
        if not endpoints:
            logger.info("No endpoints found")
            return []
        
        logger.info("Found endpoints:")
        for endpoint in endpoints:
            status = endpoint['EndpointStatus']
            name = endpoint['EndpointName']
            created = endpoint['CreationTime']
            logger.info(f"  - {name} ({status}) - Created: {created}")
        
        return endpoints
        
    except Exception as e:
        logger.error(f"Failed to list endpoints: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    
    logger = setup_logging()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  List endpoints:     python cleanup_resources.py --list")
        print("  Delete endpoint:    python cleanup_resources.py --delete-endpoint <endpoint_name>")
        print("  Cleanup run:        python cleanup_resources.py --cleanup-run <run_id>")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "--list":
        endpoints = list_all_endpoints()
        print(f"\nFound {len(endpoints)} endpoint(s)")
    
    elif action == "--delete-endpoint" and len(sys.argv) >= 3:
        endpoint_name = sys.argv[2]
        delete_endpoint(endpoint_name)
        print(f"\nEndpoint {endpoint_name} deleted successfully")
    
    elif action == "--cleanup-run" and len(sys.argv) >= 3:
        run_id = sys.argv[2]
        cleanup_run_resources(run_id)
        print(f"\nCleanup completed for run {run_id}")
    
    else:
        print("Invalid arguments.")
        sys.exit(1)