# BERT Fine-Tuned for Government of Canada Data Citation Classification

## Project Structure
```
├── config.py              # Configuration settings and hyperparameters
├── utils.py               # Utility functions for AWS operations
├── train_model.py         # Model training script
├── deploy_endpoint.py     # Model deployment script
├── run_inference.py       # Inference script (single and batch)
├── cleanup_resources.py   # Resource cleanup utilities
├── main.py               # Main orchestration script
├── requirements.txt      # Python dependencies
└── src/
    └── train.py          # Training script for SageMaker 
```

## Quick Start

### Prerequisites
1. AWS account with SageMaker access
2. IAM role with appropriate permissions
3. S3 bucket with training data
4. Python 3.9+

### Installation
```bash
pip install -r requirements.txt
```

### Setup
1. Update `config.py` with your AWS settings:
   - AWS IAM role ARN
   - S3 bucket name
   - S3 prefix

2. Ensure your training data is in S3 in JSONL format:
   - `s3://your-bucket/prefix/data/train.jsonl`
   - `s3://your-bucket/prefix/data/val.jsonl`
   - `s3://your-bucket/prefix/data/test.jsonl`

## Usage

### Option 1: Run Complete Pipeline
```bash
# Train and deploy in one go
python main.py pipeline

# Train only (no deployment)
python main.py pipeline-train-only
```

### Option 2: Step-by-Step
```bash
# 1. Train the model
python main.py train

# 2. Deploy the trained model (using run_id from training)
python main.py deploy 20250423-124743

# 3. Run inference
python main.py inference hf-endpoint-20250423-124743 "Your text here"

# 4. Batch inference
python main.py batch hf-endpoint-20250423-124743 input.jsonl results.csv
```

### Individual Scripts

#### Training
```bash
python train_model.py
```

#### Deployment
```bash
# Deploy from run ID
python deploy_endpoint.py 20250423-124743

# Deploy from specific S3 model path
python deploy_endpoint.py s3://bucket/path/model.tar.gz 20250423-124743
```

#### Inference
```bash
# Single inference
python run_inference.py hf-endpoint-20250423-124743 --single "Your text here"

# Batch inference
python run_inference.py hf-endpoint-20250423-124743 --batch input.jsonl output.csv
```

#### Cleanup
```bash
# List all endpoints
python main.py list-endpoints

# Clean up specific run resources
python main.py cleanup 20250423-124743

# Delete specific endpoint
python cleanup_resources.py --delete-endpoint endpoint-name
```

## Configuration

### Key Configuration Files
**config.py** - Main configuration:
- AWS settings (role, bucket, regions)
- Model hyperparameters
- Instance types
- Training parameters

**Environment Variables** (optional):
```bash
export BUCKET_NAME=your-bucket-name
export S3_PREFIX=your-prefix
```

### Training Data Format
Your JSONL files should contain one JSON object per line:

```json
{"text": "This is a citation example.", "label": 1}
{"text": "This is not a citation.", "label": 0}
```

## Outputs

### Training Outputs
- **Model artifacts**: Saved to S3 at `s3://bucket/prefix/models/{run_id}/`
- **Training logs**: Available in SageMaker console
- **Hyperparameters**: Saved to `s3://bucket/prefix/configs/{run_id}/hyperparams.json`
- **Model info**: Saved to `s3://bucket/prefix/configs/{run_id}/model_info.json`

### Inference Outputs
- **Single inference**: Saved to `single_inference_result.csv`
- **Batch inference**: Results saved to specified CSV file
- **Errors**: Saved to `{output_file}_errors.csv` if any occur

## Monitoring and Debugging

### Check Training Progress
```bash
# Monitor in SageMaker console or check CloudWatch logs
# Training job name format: huggingface-pytorch-training-{timestamp}
```

### List Active Resources
```bash
python main.py list-endpoints
```

### View Run Information
Run IDs follow format: `YYYYMMDD-HHMMSS`

All run information is stored in S3 under:
- `s3://bucket/prefix/configs/{run_id}/`

## Cost Management

### Estimated Costs
- **Training**: ~$1-5 per hour (ml.p3.2xlarge)
- **Inference endpoint**: ~$0.10-0.50 per hour (ml.m5.large)
- **S3 storage**: Minimal for model artifacts

### Best Practices
1. Always clean up endpoints after use:
   ```bash
   python main.py cleanup {run_id}
   ```

2. Use batch inference for multiple predictions instead of keeping endpoints running

3. Monitor your AWS costs regularly

## Troubleshooting

### Common Issues
**Training fails with permission errors:**
- Check IAM role has SageMaker, S3 access
- Verify bucket permissions

**Endpoint deployment fails:**
- Check instance limits in your AWS account
- Verify model artifacts exist in S3

**Inference fails:**
- Check payload format matches model expectations
- Verify endpoint is in 'InService' status

**Batch processing errors:**
- Check JSONL file format
- Review error CSV for specific line issues

### Getting Help
1. Check CloudWatch logs for detailed error messages
2. Verify all S3 paths and permissions
3. Ensure required data files exist in correct format

## Example Workflow

```bash
# 1. Complete pipeline
python main.py pipeline

# Output will show:
# Run ID: 20250423-124743
# Endpoint: hf-endpoint-20250423-124743

# 2. Test with your data
python main.py batch hf-endpoint-20250423-124743 my_test_data.jsonl results.csv

# 3. Clean up when done
python main.py cleanup 20250423-124743
```

## Security Notes
- Never commit AWS credentials to code
- Use IAM roles with minimal required permissions
- Regularly rotate access keys
- Monitor CloudTrail for API usage
