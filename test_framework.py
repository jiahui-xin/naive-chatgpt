"""
Unit tests for LLAMA fine-tuning framework.
"""

import unittest
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, LoRAConfig, DataConfig, TrainingConfig, EvaluationConfig


class TestConfigurations(unittest.TestCase):
    """Test configuration classes."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        self.assertEqual(config.model_name, "huggyllama/llama-7b")
        self.assertTrue(config.use_4bit)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")
    
    def test_lora_config_defaults(self):
        """Test LoRAConfig default values."""
        config = LoRAConfig()
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertIn("q_proj", config.target_modules)
        self.assertIn("v_proj", config.target_modules)
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        self.assertEqual(config.dataset_name, "alpaca")
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.batch_size, 4)
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.learning_rate, 2e-4)
        self.assertTrue(config.fp16)
    
    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig default values."""
        config = EvaluationConfig()
        self.assertTrue(config.compute_perplexity)
        self.assertTrue(config.compute_bleu)
        self.assertTrue(config.compute_rouge)


class TestDataLoader(unittest.TestCase):
    """Test data loading functionality."""
    
    @patch('data_loader.load_dataset')
    def test_alpaca_loader_creation(self, mock_load):
        """Test AlpacaDatasetLoader creation."""
        from data_loader import AlpacaDatasetLoader
        
        mock_tokenizer = Mock()
        loader = AlpacaDatasetLoader(mock_tokenizer, max_length=512)
        
        self.assertEqual(loader.max_length, 512)
        self.assertEqual(loader.tokenizer, mock_tokenizer)
    
    @patch('data_loader.load_dataset')
    def test_dailydialog_loader_creation(self, mock_load):
        """Test DailyDialogDatasetLoader creation."""
        from data_loader import DailyDialogDatasetLoader
        
        mock_tokenizer = Mock()
        loader = DailyDialogDatasetLoader(mock_tokenizer, max_length=512)
        
        self.assertEqual(loader.max_length, 512)
        self.assertEqual(loader.tokenizer, mock_tokenizer)
    
    def test_process_alpaca_example(self):
        """Test processing of Alpaca example."""
        from data_loader import AlpacaDatasetLoader
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1]
        }
        
        loader = AlpacaDatasetLoader(mock_tokenizer, max_length=512)
        
        example = {
            'instruction': 'Test instruction',
            'input': 'Test input',
            'output': 'Test output'
        }
        
        result = loader._process_alpaca_example(example)
        
        self.assertIn('input_ids', result)
        self.assertIn('labels', result)
        self.assertEqual(result['labels'], result['input_ids'])


class TestModelUtils(unittest.TestCase):
    """Test model utility functions."""
    
    def test_load_tokenizer_import(self):
        """Test that load_tokenizer can be imported."""
        from model_utils import load_tokenizer
        self.assertIsNotNone(load_tokenizer)
    
    def test_prepare_model_and_tokenizer_import(self):
        """Test that prepare_model_and_tokenizer can be imported."""
        from model_utils import prepare_model_and_tokenizer
        self.assertIsNotNone(prepare_model_and_tokenizer)


class TestEvaluation(unittest.TestCase):
    """Test evaluation framework."""
    
    def test_perplexity_evaluator_creation(self):
        """Test PerplexityEvaluator creation."""
        from evaluation import PerplexityEvaluator
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        evaluator = PerplexityEvaluator(mock_model, mock_tokenizer, device="cpu")
        
        self.assertEqual(evaluator.model, mock_model)
        self.assertEqual(evaluator.tokenizer, mock_tokenizer)
        self.assertEqual(evaluator.device, "cpu")
    
    def test_qualitative_evaluator_creation(self):
        """Test QualitativeEvaluator creation."""
        from evaluation import QualitativeEvaluator
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        evaluator = QualitativeEvaluator(mock_model, mock_tokenizer, device="cpu")
        
        self.assertEqual(evaluator.model, mock_model)
        self.assertEqual(evaluator.tokenizer, mock_tokenizer)
        self.assertEqual(evaluator.device, "cpu")
    
    def test_evaluation_framework_creation(self):
        """Test EvaluationFramework creation."""
        from evaluation import EvaluationFramework
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        framework = EvaluationFramework(mock_model, mock_tokenizer, device="cpu")
        
        self.assertIsNotNone(framework.perplexity_evaluator)
        self.assertIsNotNone(framework.qualitative_evaluator)


class TestTrainer(unittest.TestCase):
    """Test trainer functionality."""
    
    def test_trainer_import(self):
        """Test that LLAMATrainer can be imported."""
        from train import LLAMATrainer
        self.assertIsNotNone(LLAMATrainer)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_config_creation_flow(self):
        """Test creating all configs in sequence."""
        model_config = ModelConfig(model_name="test-model")
        lora_config = LoRAConfig(r=4)
        data_config = DataConfig(dataset_name="alpaca")
        training_config = TrainingConfig(num_epochs=1)
        eval_config = EvaluationConfig(compute_perplexity=True)
        
        self.assertEqual(model_config.model_name, "test-model")
        self.assertEqual(lora_config.r, 4)
        self.assertEqual(data_config.dataset_name, "alpaca")
        self.assertEqual(training_config.num_epochs, 1)
        self.assertTrue(eval_config.compute_perplexity)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
