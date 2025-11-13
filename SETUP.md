# Setup Guide for LLAMA Fine-tuning Framework

## Prerequisites

Before you begin, ensure you have the following:

1. **Hardware Requirements**
   - GPU with at least 16GB VRAM (for LLAMA-7B with 4-bit quantization)
   - 32GB+ system RAM recommended
   - CUDA-compatible GPU (NVIDIA)

2. **Software Requirements**
   - Python 3.8 or higher
   - CUDA 11.7 or higher
   - Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/jiahui-xin/naive-chatgpt.git
cd naive-chatgpt
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n llama-finetune python=3.10
conda activate llama-finetune
```

### 3. Install PyTorch with CUDA

First, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Basic Training

Run the default training script:

```bash
python train.py
```

### Using Examples

Run specific examples:

```bash
# Example 1: Alpaca fine-tuning
python examples.py 1

# Example 2: DailyDialog fine-tuning
python examples.py 2

# Example 3: Combined datasets
python examples.py 3

# Example 4: Inference
python examples.py 4
```

### Run Tests

```bash
python test_framework.py
```

## Configuration

### Edit Configuration Files

Modify `config.py` to customize:

- Model architecture (7B vs 13B)
- LoRA parameters (rank, alpha, target modules)
- Training hyperparameters
- Dataset selection
- Evaluation metrics

### Example Custom Configuration

```python
from config import ModelConfig, LoRAConfig

# Use LLAMA-13B instead of 7B
model_config = ModelConfig(
    model_name="huggyllama/llama-13b",
    use_4bit=True
)

# Increase LoRA rank for better performance
lora_config = LoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution:**
- Reduce batch size in `DataConfig`
- Increase `gradient_accumulation_steps`
- Use LLAMA-7B instead of 13B
- Enable gradient checkpointing

```python
data_config = DataConfig(
    batch_size=2  # Reduced from 4
)

training_config = TrainingConfig(
    gradient_accumulation_steps=8  # Increased from 4
)
```

#### 2. BitsAndBytes Installation Error

**Solution:**
```bash
pip uninstall bitsandbytes
pip install bitsandbytes --no-cache-dir
```

#### 3. Dataset Download Fails

**Solution:**
The framework includes fallback sample datasets. If you want to use full datasets:

```bash
# Pre-download datasets
python -c "from datasets import load_dataset; load_dataset('tatsu-lab/alpaca')"
python -c "from datasets import load_dataset; load_dataset('daily_dialog')"
```

#### 4. Import Errors

**Solution:**
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Directory Structure

After setup, your directory should look like:

```
naive-chatgpt/
├── config.py              # Configuration classes
├── model_utils.py         # Model loading utilities
├── data_loader.py         # Dataset loaders
├── evaluation.py          # Evaluation framework
├── train.py              # Training script
├── examples.py           # Example usage scripts
├── test_framework.py     # Unit tests
├── requirements.txt      # Dependencies
├── README.md            # Documentation
├── SETUP.md            # This file
└── .gitignore          # Git ignore rules
```

## Next Steps

1. **Read the README.md** for detailed documentation
2. **Run the examples** to familiarize yourself with the framework
3. **Customize configurations** for your specific use case
4. **Train your model** and evaluate results
5. **Share your results** with the community

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review the examples in `examples.py`
3. Run tests with `test_framework.py` to verify installation
4. Check GitHub Issues for similar problems
5. Create a new issue with detailed error messages

## Performance Tips

1. **Use mixed precision training** (enabled by default with `fp16=True`)
2. **Enable gradient checkpointing** for larger models
3. **Use appropriate batch sizes** for your GPU
4. **Monitor GPU usage** with `nvidia-smi`
5. **Use WandB** for experiment tracking (set `use_wandb=True`)

## Advanced Configuration

### Using WandB for Experiment Tracking

```python
training_config = TrainingConfig(
    use_wandb=True,
    wandb_project="my-llama-experiments"
)
```

### Multi-GPU Training

The framework automatically supports multi-GPU training via Hugging Face Accelerate:

```bash
# Automatic multi-GPU
python train.py

# Or with accelerate
accelerate config
accelerate launch train.py
```

### Custom Datasets

Create your own dataset loader:

```python
from data_loader import AlpacaDatasetLoader

class CustomDatasetLoader(AlpacaDatasetLoader):
    def load(self):
        # Your custom loading logic
        pass
```

## License

MIT License - see LICENSE file for details.
