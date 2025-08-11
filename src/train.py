import os
import logging
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load training and validation datasets from JSONL files."""
    try:
        train_path = "/opt/ml/input/data/train/train.jsonl"
        validation_path = "/opt/ml/input/data/validation/val.jsonl"
        
        # Check if files exist
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not os.path.exists(validation_path):
            raise FileNotFoundError(f"Validation file not found: {validation_path}")
        
        # Load JSONL datasets
        logger.info(f"Loading training dataset from {train_path}")
        train_dataset = load_dataset('json', data_files=train_path, split='train')
        logger.info(f"Loading validation dataset from {validation_path}")
        validation_dataset = load_dataset('json', data_files=validation_path, split='train')
        
        # Verify datasets are not empty
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        if len(validation_dataset) == 0:
            raise ValueError("Validation dataset is empty")
        
        logger.info(f"Loaded {len(train_dataset)} training samples and {len(validation_dataset)} validation samples")
        
        # Log label distribution
        train_labels = train_dataset['label']
        val_labels = validation_dataset['label']
        logger.info(f"Training label distribution: {np.bincount(train_labels)}")
        logger.info(f"Validation label distribution: {np.bincount(val_labels)}")
        
        return train_dataset, validation_dataset
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def tokenize_fn(examples, tokenizer):
    """Tokenize text data."""
    try:
        # Use a more conservative max_length that matches your training
        max_length = int(os.environ.get("max_length", 128))
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=max_length,
            return_tensors=None  # Let datasets handle tensor conversion
        )
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise

def compute_metrics(eval_pred):
    """Compute comprehensive metrics."""
    try:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate multiple metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    except Exception as e:
        logger.error(f"Metrics computation failed: {str(e)}")
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

def main():
    try:
        # Load model and tokenizer
        model_name = os.environ.get("model_name_or_path", "bert-base-uncased")
        logger.info(f"Loading model and tokenizer: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Add special tokens if needed (shouldn't be necessary for BERT, but good practice)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.resize_token_embeddings(len(tokenizer))
        
        # Load datasets
        train_dataset, eval_dataset = load_data()
        
        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        train_dataset = train_dataset.map(
            lambda x: tokenize_fn(x, tokenizer), 
            batched=True,
            remove_columns=['text']  # Remove original text to save memory
        )
        eval_dataset = eval_dataset.map(
            lambda x: tokenize_fn(x, tokenizer), 
            batched=True,
            remove_columns=['text']
        )
        
        # Set format for PyTorch
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # Setup training arguments
        output_dir = os.path.join(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), os.environ.get("run_id", ""))
        
        # Log training configuration
        num_epochs = int(os.environ.get("num_train_epochs", 3))
        batch_size = int(os.environ.get("per_device_train_batch_size", 16))
        learning_rate = float(os.environ.get("learning_rate", 2e-5))
        
        logger.info(f"Training configuration:")
        logger.info(f"  - Epochs: {num_epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Output directory: {output_dir}")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=int(os.environ.get("per_device_eval_batch_size", 32)),
            evaluation_strategy=os.environ.get("evaluation_strategy", "steps"),
            eval_steps=int(os.environ.get("eval_steps", 500)),
            save_steps=int(os.environ.get("save_steps", 500)),
            save_total_limit=3,
            learning_rate=learning_rate,
            weight_decay=float(os.environ.get("weight_decay", 0.01)),
            load_best_model_at_end=True,
            metric_for_best_model=os.environ.get("metric_for_best_model", "f1"),
            greater_is_better=True,  # Explicitly state that higher F1 is better
            logging_strategy="steps",
            logging_steps=100,
            report_to=None,  # Disable wandb/tensorboard logging
            dataloader_pin_memory=False,  # Can help with memory issues
        )
        
        # Initialize Trainer
        logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,  # Add tokenizer to trainer
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        
        # Train model
        logger.info("Starting training...")
        train_result = trainer.train()
        logger.info("Training completed.")
        
        # Log final training metrics
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Evaluate on validation set
        logger.info("Running final evaluation...")
        eval_result = trainer.evaluate()
        logger.info(f"Final evaluation metrics: {eval_result}")
        
        # Save model and tokenizer
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training summary
        summary = {
            "training_loss": train_result.training_loss,
            "eval_metrics": eval_result,
            "model_name": model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_samples": len(train_dataset),
            "validation_samples": len(eval_dataset)
        }
        
        summary_path = os.path.join(output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_path}")
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()