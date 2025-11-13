"""
Dataset loaders for Alpaca and DailyDialog datasets.
"""

import json
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset, DatasetDict
import logging

logger = logging.getLogger(__name__)


class AlpacaDatasetLoader:
    """Loader for Alpaca instruction-following dataset."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load(self) -> DatasetDict:
        """Load and process Alpaca dataset."""
        try:
            # Load from Hugging Face
            dataset = load_dataset("tatsu-lab/alpaca", split="train")
            logger.info(f"Loaded Alpaca dataset with {len(dataset)} examples")
        except Exception as e:
            logger.warning(f"Failed to load from HF, creating sample dataset: {e}")
            # Create a sample dataset if loading fails
            dataset = self._create_sample_dataset()
        
        # Split into train/validation
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Process the datasets
        train_dataset = split_dataset["train"].map(
            self._process_alpaca_example,
            remove_columns=dataset.column_names,
            desc="Processing Alpaca training data"
        )
        
        val_dataset = split_dataset["test"].map(
            self._process_alpaca_example,
            remove_columns=dataset.column_names,
            desc="Processing Alpaca validation data"
        )
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    
    def _process_alpaca_example(self, example: Dict) -> Dict:
        """Process a single Alpaca example into model inputs."""
        # Format the instruction
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        full_text = prompt + output
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def _create_sample_dataset(self) -> Dataset:
        """Create a small sample dataset for testing."""
        samples = [
            {
                "instruction": "Write a greeting message.",
                "input": "",
                "output": "Hello! How can I assist you today?"
            },
            {
                "instruction": "Translate the following to French.",
                "input": "Hello, how are you?",
                "output": "Bonjour, comment allez-vous?"
            },
            {
                "instruction": "Summarize the text.",
                "input": "The quick brown fox jumps over the lazy dog.",
                "output": "A fox jumps over a dog."
            }
        ] * 100  # Repeat for more samples
        
        return Dataset.from_list(samples)


class DailyDialogDatasetLoader:
    """Loader for DailyDialog conversation dataset."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load(self) -> DatasetDict:
        """Load and process DailyDialog dataset."""
        try:
            # Load from Hugging Face
            dataset = load_dataset("daily_dialog", split="train")
            logger.info(f"Loaded DailyDialog dataset with {len(dataset)} examples")
        except Exception as e:
            logger.warning(f"Failed to load from HF, creating sample dataset: {e}")
            # Create a sample dataset if loading fails
            dataset = self._create_sample_dataset()
        
        # Split into train/validation
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Process the datasets
        train_dataset = split_dataset["train"].map(
            self._process_dialog_example,
            remove_columns=dataset.column_names,
            desc="Processing DailyDialog training data"
        )
        
        val_dataset = split_dataset["test"].map(
            self._process_dialog_example,
            remove_columns=dataset.column_names,
            desc="Processing DailyDialog validation data"
        )
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    
    def _process_dialog_example(self, example: Dict) -> Dict:
        """Process a single dialog example into model inputs."""
        # Get dialog turns
        dialog = example.get("dialog", [])
        
        if not dialog:
            dialog = ["Hello", "Hi there!"]
        
        # Create conversation format
        conversation = ""
        for i, turn in enumerate(dialog):
            speaker = "User" if i % 2 == 0 else "Assistant"
            conversation += f"{speaker}: {turn}\n"
        
        # Tokenize
        tokenized = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def _create_sample_dataset(self) -> Dataset:
        """Create a small sample dataset for testing."""
        samples = [
            {
                "dialog": ["Hello, how are you?", "I'm doing well, thank you!", "That's great to hear."]
            },
            {
                "dialog": ["What's the weather like?", "It's sunny today.", "Perfect for a walk!"]
            },
            {
                "dialog": ["Can you help me?", "Of course! What do you need?", "I need directions."]
            }
        ] * 100  # Repeat for more samples
        
        return Dataset.from_list(samples)


def load_datasets(dataset_name: str, tokenizer, max_length: int = 512) -> DatasetDict:
    """
    Load and combine datasets based on configuration.
    
    Args:
        dataset_name: Name of dataset(s) to load ("alpaca", "dailydialog", or "both")
        tokenizer: Tokenizer to use for processing
        max_length: Maximum sequence length
    
    Returns:
        DatasetDict with train and validation splits
    """
    if dataset_name == "alpaca":
        loader = AlpacaDatasetLoader(tokenizer, max_length)
        return loader.load()
    
    elif dataset_name == "dailydialog":
        loader = DailyDialogDatasetLoader(tokenizer, max_length)
        return loader.load()
    
    elif dataset_name == "both":
        # Load both datasets and concatenate
        alpaca_loader = AlpacaDatasetLoader(tokenizer, max_length)
        dialog_loader = DailyDialogDatasetLoader(tokenizer, max_length)
        
        alpaca_data = alpaca_loader.load()
        dialog_data = dialog_loader.load()
        
        from datasets import concatenate_datasets
        
        combined_train = concatenate_datasets([
            alpaca_data["train"],
            dialog_data["train"]
        ])
        
        combined_val = concatenate_datasets([
            alpaca_data["validation"],
            dialog_data["validation"]
        ])
        
        return DatasetDict({
            "train": combined_train,
            "validation": combined_val
        })
    
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
