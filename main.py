
#Main orchestrator for BERT Citation Classification

import sys
import json
from train_model import train_bert_model
from deploy_endpoint import deploy_model_endpoint, deploy_from_run_id
from run_inference import single_inference, batch_inference_local
from cleanup_resources import cleanup_run_resources, list_all_endpoints
from utils import setup_logging


def run_full_pipeline(deploy_after_training=True):
    #Run the complete training and deployment pipeline
    logger = setup_logging()
    
    try:
        # Step 1: Train the model
        logger.info("Starting full pipeline: Training and then Deploy and then Test")
        training_result = train_bert_model()
        
        run_id = training_result['run_id']
        model_data = training_result['model_data']
        
        if deploy_after_training:
            # Step 2: Deploy the model
            deployment_result = deploy_model_endpoint(model_data, run_id)
            endpoint_name = deployment_result['endpoint_name']
            
            # Step 3: Test with a simple inference
            test_text = "Data for this study was taken from Fisheries and Oceans Canada."
            test_result = single_inference(endpoint_name, test_text)
            
            logger.info("Full pipeline completed successfully")
            
            return {
                'run_id': run_id,
                'model_data': model_data,
                'endpoint_name': endpoint_name,
                'test_result': test_result
            }
        else:
            logger.info("Training completed. Skipping deployment.")
            return {
                'run_id': run_id,
                'model_data': model_data
            }
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


def print_usage():
    """Print usage instructions"""
    print("BERT Citation Classification Pipeline")
    print("=" * 50)
    print("\nUsage: python main.py <command> [arguments]\n")
    print("Commands:")
    print("  train                          - Train BERT model only")
    print("  deploy <run_id>                - Deploy model from run ID")
    print("  deploy <model_s3_path> <run_id> - Deploy model from S3 path")
    print("  inference <endpoint> '<text>'   - Single text inference")
    print("  batch <endpoint> <input> <output> - Batch inference")
    print("  pipeline                       - Run full pipeline (train + deploy)")
    print("  pipeline-train-only            - Train only (no deployment)")
    print("  cleanup <run_id>               - Clean up resources for run")
    print("  list-endpoints                 - List all endpoints")
    print("\nExamples:")
    print("  python main.py train")
    print("  python main.py deploy 20250423-124743")
    print("  python main.py inference hf-endpoint-20250423-124743 'This is a citation.'")
    print("  python main.py batch hf-endpoint-20250423-124743 test.jsonl results.csv")
    print("  python main.py pipeline")
    print("  python main.py cleanup 20250423-124743")


if __name__ == "__main__":
    logger = setup_logging()
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "train":
            result = train_bert_model()
            print(f"\nTraining completed")
            print(f"Run ID: {result['run_id']}")
            print(f"Model location: {result['model_data']}")
        
        elif command == "deploy":
            if len(sys.argv) == 3:
                # Deploy from run ID
                run_id = sys.argv[2]
                result = deploy_from_run_id(run_id)
            elif len(sys.argv) == 4:
                # Deploy from model path and run ID
                model_path = sys.argv[2]
                run_id = sys.argv[3]
                result = deploy_model_endpoint(model_path, run_id)
            else:
                print("Usage: python main.py deploy <run_id> OR <model_s3_path> <run_id>")
                sys.exit(1)
            
            print(f"\nDeployment completed")
            print(f"Endpoint: {result['endpoint_name']}")
        
        elif command == "inference":
            if len(sys.argv) != 4:
                print("Usage: python main.py inference <endpoint_name> '<text>'")
                sys.exit(1)
            
            endpoint_name = sys.argv[2]
            text = sys.argv[3]
            result = single_inference(endpoint_name, text)
            print(f"\nInference completed")
            print(f"Text: {text}")
            print(f"Result: {result}")
        
        elif command == "batch":
            if len(sys.argv) != 5:
                print("Usage: python main.py batch <endpoint_name> <input_file> <output_file>")
                sys.exit(1)
            
            endpoint_name = sys.argv[2]
            input_file = sys.argv[3]
            output_file = sys.argv[4]
            results, errors = batch_inference_local(endpoint_name, input_file, output_file)
            print(f"\nBatch inference completed")
            print(f"Processed: {len(results)} examples")
            print(f"Errors: {len(errors)}")
            print(f"Results saved to: {output_file}")
        
        elif command == "pipeline":
            result = run_full_pipeline(deploy_after_training=True)
            print(f"\nFull pipeline completed")
            print(f"Run ID: {result['run_id']}")
            print(f"Endpoint: {result['endpoint_name']}")
            print(f"Test result: {result['test_result']}")
        
        elif command == "pipeline-train-only":
            result = run_full_pipeline(deploy_after_training=False)
            print(f"\nTraining pipeline completed")
            print(f"Run ID: {result['run_id']}")
            print(f"Model location: {result['model_data']}")
        
        elif command == "cleanup":
            if len(sys.argv) != 3:
                print("Usage: python main.py cleanup <run_id>")
                sys.exit(1)
            
            run_id = sys.argv[2]
            cleanup_run_resources(run_id)
            print(f"\nCleanup completed for run {run_id}")
        
        elif command == "list-endpoints":
            endpoints = list_all_endpoints()
            print(f"\nFound {len(endpoints)} endpoint(s)")
        
        else:
            print(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)