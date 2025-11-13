"""
Configuration for LLAMA fine-tuning with LoRA and 4-bit quantization.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for model architecture and loading."""
    model_name: str = "huggyllama/llama-7b"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    cache_dir: Optional[str] = None


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """Configuration for dataset loading and processing."""
    dataset_name: str = "alpaca"  # Options: "alpaca", "dailydialog", "both"
    max_length: int = 512
    train_split: float = 0.9
    validation_split: float = 0.1
    batch_size: int = 4
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    output_dir: str = "./outputs"
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    fp16: bool = True
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    save_total_limit: int = 3
    use_wandb: bool = False
    wandb_project: str = "llama-finetuning"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    compute_perplexity: bool = True
    compute_bleu: bool = True
    compute_rouge: bool = True
    generation_max_length: int = 256
    num_beams: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
