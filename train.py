"""
Training script for LLAMA fine-tuning with LoRA and 4-bit quantization.
"""

import os
import logging
from typing import Optional
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader

from config import (
    ModelConfig,
    LoRAConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig
)
from model_utils import prepare_model_and_tokenizer
from data_loader import load_datasets
from evaluation import EvaluationFramework

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLAMATrainer:
    """Trainer for LLAMA models with LoRA and 4-bit quantization."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
        eval_config: EvaluationConfig
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.data_config = data_config
        self.training_config = training_config
        self.eval_config = eval_config
        
        # Initialize model and tokenizer
        logger.info("Initializing model and tokenizer...")
        self.model, self.tokenizer = prepare_model_and_tokenizer(
            model_config, lora_config
        )
        
        # Load datasets
        logger.info(f"Loading {data_config.dataset_name} dataset...")
        self.datasets = load_datasets(
            data_config.dataset_name,
            self.tokenizer,
            data_config.max_length
        )
        
        logger.info(f"Train dataset size: {len(self.datasets['train'])}")
        logger.info(f"Validation dataset size: {len(self.datasets['validation'])}")
        
        # Initialize evaluation framework
        self.evaluator = EvaluationFramework(
            self.model,
            self.tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def train(self):
        """Train the model."""
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.data_config.batch_size,
            per_device_eval_batch_size=self.data_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            fp16=self.training_config.fp16,
            optim=self.training_config.optim,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            max_grad_norm=self.training_config.max_grad_norm,
            save_total_limit=self.training_config.save_total_limit,
            report_to="wandb" if self.training_config.use_wandb else "none",
            run_name=f"{self.model_config.model_name}-lora" if self.training_config.use_wandb else None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["validation"],
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.training_config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config.output_dir)
        
        return trainer
    
    def evaluate(self, test_prompts: Optional[list] = None, test_references: Optional[list] = None):
        """
        Evaluate the model.
        
        Args:
            test_prompts: Optional list of test prompts for generation
            test_references: Optional list of reference texts
        """
        logger.info("Starting evaluation...")
        
        # Create validation dataloader
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        val_dataloader = DataLoader(
            self.datasets["validation"],
            batch_size=self.data_config.batch_size,
            collate_fn=data_collator
        )
        
        # Run evaluation
        generation_config = {
            'max_length': self.eval_config.generation_max_length,
            'num_beams': self.eval_config.num_beams,
            'temperature': self.eval_config.temperature,
            'top_p': self.eval_config.top_p
        }
        
        results = self.evaluator.evaluate(
            dataloader=val_dataloader,
            test_prompts=test_prompts,
            test_references=test_references,
            compute_perplexity=self.eval_config.compute_perplexity,
            compute_bleu=self.eval_config.compute_bleu,
            compute_rouge=self.eval_config.compute_rouge,
            generation_config=generation_config
        )
        
        logger.info("Evaluation results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results


def main():
    """Main training function."""
    # Initialize configurations
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    data_config = DataConfig()
    training_config = TrainingConfig()
    eval_config = EvaluationConfig()
    
    # Create trainer
    trainer = LLAMATrainer(
        model_config=model_config,
        lora_config=lora_config,
        data_config=data_config,
        training_config=training_config,
        eval_config=eval_config
    )
    
    # Train
    trainer.train()
    
    # Evaluate with sample prompts
    test_prompts = [
        "### Instruction:\nWrite a short story about a robot.\n\n### Response:\n",
        "### Instruction:\nExplain machine learning in simple terms.\n\n### Response:\n",
    ]
    test_references = [
        "Once upon a time, there was a friendly robot named Bob who helped people.",
        "Machine learning is a way for computers to learn from data and make predictions."
    ]
    
    trainer.evaluate(test_prompts, test_references)


if __name__ == "__main__":
    main()
