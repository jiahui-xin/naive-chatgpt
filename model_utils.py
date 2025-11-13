"""
Model utilities for loading LLAMA with 4-bit quantization and LoRA.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def load_model_with_quantization(
    model_name: str,
    use_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    use_nested_quant: bool = False,
    cache_dir: str = None
) -> AutoModelForCausalLM:
    """
    Load LLAMA model with 4-bit quantization.
    
    Args:
        model_name: Name or path of the model
        use_4bit: Whether to use 4-bit quantization
        bnb_4bit_compute_dtype: Compute dtype for 4-bit base models
        bnb_4bit_quant_type: Quantization type (fp4 or nf4)
        use_nested_quant: Whether to use nested quantization
        cache_dir: Directory to cache downloaded models
    
    Returns:
        Loaded model
    """
    if use_4bit:
        # Configure 4-bit quantization
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )
        
        logger.info(f"Loading model {model_name} with 4-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    else:
        logger.info(f"Loading model {model_name} without quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    
    return model


def add_lora_adapters(
    model: AutoModelForCausalLM,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: list = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> AutoModelForCausalLM:
    """
    Add LoRA adapters to the model.
    
    Args:
        model: Base model to add LoRA to
        r: LoRA attention dimension
        lora_alpha: Alpha parameter for LoRA scaling
        target_modules: Names of modules to apply LoRA to
        lora_dropout: Dropout probability for LoRA layers
        bias: Bias type for LoRA
        task_type: Type of task
    
    Returns:
        Model with LoRA adapters
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else task_type
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params
    
    logger.info(f"Trainable params: {trainable_params:,} || "
                f"All params: {all_params:,} || "
                f"Trainable%: {trainable_percent:.2f}%")
    
    return model


def load_tokenizer(model_name: str, cache_dir: str = None) -> AutoTokenizer:
    """
    Load tokenizer for the model.
    
    Args:
        model_name: Name or path of the model
        cache_dir: Directory to cache downloaded tokenizer
    
    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def prepare_model_and_tokenizer(
    model_config,
    lora_config
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Prepare model and tokenizer with quantization and LoRA.
    
    Args:
        model_config: Model configuration
        lora_config: LoRA configuration
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = load_tokenizer(
        model_config.model_name,
        cache_dir=model_config.cache_dir
    )
    
    # Load model with quantization
    model = load_model_with_quantization(
        model_name=model_config.model_name,
        use_4bit=model_config.use_4bit,
        bnb_4bit_compute_dtype=model_config.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        use_nested_quant=model_config.use_nested_quant,
        cache_dir=model_config.cache_dir
    )
    
    # Add LoRA adapters
    model = add_lora_adapters(
        model=model,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type
    )
    
    return model, tokenizer
