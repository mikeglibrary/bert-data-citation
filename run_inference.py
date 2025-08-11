
#Run inference on trained BERT model using SageMaker endpoint
#Supports both single predictions and batch processing

import json
import csv
import pandas as pd
import os
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from utils import setup_logging, initialize_aws_clients


def get_predictor(endpoint_name):
    #Create a predictor for the given endpoint
    sagemaker_session, _ = initialize_aws_clients()
    
    return Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )


def single_inference(endpoint_name, text):
    #Run inference on a single text example
    logger = setup_logging()
    
    try:
        predictor = get_predictor(endpoint_name)
        
        logger.info(f"Running inference on: '{text[:50]}...'")
        payload = {"inputs": text}  # or {"text": text} depending on your model
        result = predictor.predict(payload)
        
        logger.info(f"Prediction: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Single inference failed: {str(e)}")
        raise


def batch_inference_local(endpoint_name, input_file, output_file):
    
    #Process a JSONL file locally using the endpoint
    #Each line should be a JSON object with a 'text' field
    
    logger = setup_logging()
    
    try:
        predictor = get_predictor(endpoint_name)
        
        logger.info(f"Processing file: {input_file}")
        
        results = []
        errors = []
        
        # Read the file line by line
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        logger.info(f"Found {total_lines} examples to process")
        
        for i, line in enumerate(lines):
            try:
                # Parse the JSON object
                data = json.loads(line.strip())
                
                # Extract the text field
                text = data.get("text", "")
                if not text:
                    logger.warning(f"Example {i+1}/{total_lines}: Empty text field, skipping")
                    continue
                
                # Create payload and predict
                payload = {"inputs": text}  # Adjust based on your model's expected format
                prediction = predictor.predict(payload)
                
                # Store the result
                result = {
                    "input_text": text,
                    "prediction": prediction,
                    "line_number": i + 1
                }
                results.append(result)
                
                # Log progress
                if (i+1) % 10 == 0 or (i+1) == total_lines:
                    logger.info(f"✅ Processed {i+1}/{total_lines} examples")
                
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing error at line {i+1}: {str(e)}"
                logger.error(f"Error: {error_msg}")
                errors.append({"line": i+1, "error": "JSON parsing error", "details": str(e)})
            except Exception as e:
                error_msg = f"Error processing example {i+1}: {str(e)}"
                logger.error(f"Error: {error_msg}")
                errors.append({"line": i+1, "error": "Processing error", "details": str(e)})
        
        # Save results to CSV
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        # Save errors to a separate file if any
        if errors:
            error_df = pd.DataFrame(errors)
            error_path = f"{os.path.splitext(output_file)[0]}_errors.csv"
            error_df.to_csv(error_path, index=False)
            logger.info(f"{len(errors)} errors encountered. Details saved to {error_path}")
        
        return results, errors
        
    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}")
        raise


def save_single_result_to_csv(text, result, filename="single_inference_result.csv"):
    """Save a single inference result to CSV"""
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["input_text", "prediction"])
        writer.writeheader()
        writer.writerow({
            "input_text": text,
            "prediction": json.dumps(result)
        })


if __name__ == "__main__":
    import sys
    
    logger = setup_logging()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single inference: python run_inference.py <endpoint_name> --single '<text>'")
        print("  Batch inference:  python run_inference.py <endpoint_name> --batch <input_file> <output_file>")
        sys.exit(1)
    
    endpoint_name = sys.argv[1]
    
    if len(sys.argv) >= 4 and sys.argv[2] == "--single":
        # Single inference
        text = sys.argv[3]
        result = single_inference(endpoint_name, text)
        
        # Save result
        save_single_result_to_csv(text, result)
        logger.info(f"Result saved to single_inference_result.csv")
        
        print(f"\nSingle inference completed")
        print(f"Text: {text}")
        print(f"Result: {result}")
    
    elif len(sys.argv) >= 5 and sys.argv[2] == "--batch":
        # Batch inference
        input_file = sys.argv[3]
        output_file = sys.argv[4]
        
        results, errors = batch_inference_local(endpoint_name, input_file, output_file)
        
        print(f"\nBatch inference completed")
        print(f"Processed {len(results)} examples successfully")
        print(f"Encountered {len(errors)} errors")
        print(f"Results saved to {output_file}")
    
    else:
        print("Invalid arguments. Use --single or --batch mode.")
        sys.exit(1)