"""
Example script demonstrating LLAMA fine-tuning with different configurations.
"""

import logging
from config import ModelConfig, LoRAConfig, DataConfig, TrainingConfig, EvaluationConfig
from train import LLAMATrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_alpaca_training():
    """Example: Fine-tune LLAMA-7B on Alpaca dataset."""
    print("\n" + "="*60)
    print("Example 1: Fine-tuning LLAMA-7B on Alpaca Dataset")
    print("="*60 + "\n")
    
    model_config = ModelConfig(
        model_name="huggyllama/llama-7b",
        use_4bit=True,
        bnb_4bit_quant_type="nf4"
    )
    
    lora_config = LoRAConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05
    )
    
    data_config = DataConfig(
        dataset_name="alpaca",
        max_length=512,
        batch_size=4
    )
    
    training_config = TrainingConfig(
        output_dir="./outputs/alpaca-llama-7b",
        num_epochs=3,
        learning_rate=2e-4,
        gradient_accumulation_steps=4
    )
    
    eval_config = EvaluationConfig(
        compute_perplexity=True,
        compute_bleu=True,
        compute_rouge=True
    )
    
    trainer = LLAMATrainer(
        model_config=model_config,
        lora_config=lora_config,
        data_config=data_config,
        training_config=training_config,
        eval_config=eval_config
    )
    
    trainer.train()
    
    # Evaluate
    test_prompts = [
        "### Instruction:\nWrite a greeting message.\n\n### Response:\n",
        "### Instruction:\nExplain what is machine learning.\n\n### Response:\n",
    ]
    test_references = [
        "Hello! How can I assist you today?",
        "Machine learning is a subset of AI that enables computers to learn from data."
    ]
    
    trainer.evaluate(test_prompts, test_references)


def example_dailydialog_training():
    """Example: Fine-tune LLAMA-7B on DailyDialog dataset."""
    print("\n" + "="*60)
    print("Example 2: Fine-tuning LLAMA-7B on DailyDialog Dataset")
    print("="*60 + "\n")
    
    model_config = ModelConfig(
        model_name="huggyllama/llama-7b",
        use_4bit=True
    )
    
    lora_config = LoRAConfig(
        r=16,  # Higher rank for dialogue
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    data_config = DataConfig(
        dataset_name="dailydialog",
        max_length=512,
        batch_size=4
    )
    
    training_config = TrainingConfig(
        output_dir="./outputs/dailydialog-llama-7b",
        num_epochs=5,
        learning_rate=3e-4
    )
    
    eval_config = EvaluationConfig(
        compute_perplexity=True,
        generation_max_length=128
    )
    
    trainer = LLAMATrainer(
        model_config=model_config,
        lora_config=lora_config,
        data_config=data_config,
        training_config=training_config,
        eval_config=eval_config
    )
    
    trainer.train()
    trainer.evaluate()


def example_combined_datasets():
    """Example: Fine-tune LLAMA-13B on combined datasets."""
    print("\n" + "="*60)
    print("Example 3: Fine-tuning LLAMA-13B on Combined Datasets")
    print("="*60 + "\n")
    
    model_config = ModelConfig(
        model_name="huggyllama/llama-13b",
        use_4bit=True,
        bnb_4bit_quant_type="nf4",
        use_nested_quant=True  # Additional optimization for 13B
    )
    
    lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1
    )
    
    data_config = DataConfig(
        dataset_name="both",  # Combined Alpaca and DailyDialog
        max_length=512,
        batch_size=2  # Smaller batch for 13B
    )
    
    training_config = TrainingConfig(
        output_dir="./outputs/combined-llama-13b",
        num_epochs=3,
        learning_rate=1e-4,
        gradient_accumulation_steps=8,
        use_wandb=True,
        wandb_project="llama-13b-finetuning"
    )
    
    eval_config = EvaluationConfig(
        compute_perplexity=True,
        compute_bleu=True,
        compute_rouge=True
    )
    
    trainer = LLAMATrainer(
        model_config=model_config,
        lora_config=lora_config,
        data_config=data_config,
        training_config=training_config,
        eval_config=eval_config
    )
    
    trainer.train()
    trainer.evaluate()


def example_inference():
    """Example: Inference with a fine-tuned model."""
    print("\n" + "="*60)
    print("Example 4: Inference with Fine-tuned Model")
    print("="*60 + "\n")
    
    from transformers import AutoTokenizer
    from peft import AutoPeftModelForCausalLM
    from evaluation import QualitativeEvaluator
    
    # Load fine-tuned model
    model_path = "./outputs/alpaca-llama-7b"
    
    print(f"Loading model from {model_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create evaluator
    evaluator = QualitativeEvaluator(model, tokenizer)
    
    # Test prompts
    prompts = [
        "### Instruction:\nWrite a short poem about autumn.\n\n### Response:\n",
        "### Instruction:\nExplain why the sky is blue.\n\n### Response:\n",
        "### Instruction:\nWrite a product description for a smartwatch.\n\n### Response:\n"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt:\n{prompt}")
        response = evaluator.generate_response(
            prompt,
            max_length=256,
            temperature=0.7,
            num_beams=4
        )
        print(f"\nResponse:\n{response}")
        print("-" * 60)


if __name__ == "__main__":
    import sys
    
    examples = {
        "1": example_alpaca_training,
        "2": example_dailydialog_training,
        "3": example_combined_datasets,
        "4": example_inference
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("\nAvailable examples:")
        print("1. Fine-tune LLAMA-7B on Alpaca dataset")
        print("2. Fine-tune LLAMA-7B on DailyDialog dataset")
        print("3. Fine-tune LLAMA-13B on combined datasets")
        print("4. Inference with fine-tuned model")
        print("\nUsage: python examples.py [1-4]")
        print("Running default example (1)...\n")
        example_alpaca_training()
