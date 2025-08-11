
#Configuration settings for BERT Citation Classification

import os

# AWS Configuration
AWS_ROLE = "arn:aws:iam::058264459012:role/sagemaker-access"
BUCKET_NAME = os.environ.get("BUCKET_NAME", "bert-citation-bkt")
S3_PREFIX = os.environ.get("S3_PREFIX", "bert_citation")

# Model Configuration
MODEL_CONFIG = {
    "model_name_or_path": "bert-base-uncased",
    "task_name": "binary_classification",
    "transformers_version": "4.26",
    "pytorch_version": "1.13",
    "py_version": "py39"
}

# Training Hyperparameters
TRAINING_PARAMS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "learning_rate": 2e-5,
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "save_steps": 500,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1"
}

# Instance Configuration
TRAINING_INSTANCE_TYPE = "ml.p3.2xlarge"
INFERENCE_INSTANCE_TYPE = "ml.m5.large"
BATCH_INSTANCE_TYPE = "ml.m5.large"

# File paths
REQUIRED_DATA_FILES = ["train.jsonl", "val.jsonl", "test.jsonl"]