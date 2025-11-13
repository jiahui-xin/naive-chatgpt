"""
Evaluation framework with perplexity and qualitative metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """Evaluator for computing perplexity on validation data."""
    
    def __init__(self, model, tokenizer: AutoTokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def compute_perplexity(self, dataloader: DataLoader) -> float:
        """
        Compute perplexity on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
        
        Returns:
            Perplexity score
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing perplexity"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Count non-padding tokens
                num_tokens = (labels != -100).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity


class QualitativeEvaluator:
    """Evaluator for qualitative metrics like BLEU and ROUGE."""
    
    def __init__(self, model, tokenizer: AutoTokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_response(
        self,
        prompt: str,
        max_length: int = 256,
        num_beams: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text
        """
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        
        return response
    
    def compute_bleu(self, references: List[str], hypotheses: List[str]) -> float:
        """
        Compute BLEU score.
        
        Args:
            references: Reference texts
            hypotheses: Generated texts
        
        Returns:
            BLEU score
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            smoothie = SmoothingFunction().method4
            scores = []
            
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()
                score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
                scores.append(score)
            
            avg_bleu = np.mean(scores)
            logger.info(f"BLEU Score: {avg_bleu:.4f}")
            return avg_bleu
        
        except ImportError:
            logger.warning("NLTK not available, skipping BLEU computation")
            return 0.0
    
    def compute_rouge(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            references: Reference texts
            hypotheses: Generated texts
        
        Returns:
            Dictionary of ROUGE scores
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for ref, hyp in zip(references, hypotheses):
                scores = scorer.score(ref, hyp)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            results = {
                'rouge1': np.mean(rouge1_scores),
                'rouge2': np.mean(rouge2_scores),
                'rougeL': np.mean(rougeL_scores)
            }
            
            logger.info(f"ROUGE-1: {results['rouge1']:.4f}, "
                       f"ROUGE-2: {results['rouge2']:.4f}, "
                       f"ROUGE-L: {results['rougeL']:.4f}")
            
            return results
        
        except ImportError:
            logger.warning("rouge-score not available, skipping ROUGE computation")
            return {}


class EvaluationFramework:
    """Complete evaluation framework combining quantitative and qualitative metrics."""
    
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: str = "cuda"
    ):
        self.perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device)
        self.qualitative_evaluator = QualitativeEvaluator(model, tokenizer, device)
    
    def evaluate(
        self,
        dataloader: DataLoader,
        test_prompts: Optional[List[str]] = None,
        test_references: Optional[List[str]] = None,
        compute_perplexity: bool = True,
        compute_bleu: bool = True,
        compute_rouge: bool = True,
        generation_config: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Run complete evaluation.
        
        Args:
            dataloader: DataLoader for perplexity computation
            test_prompts: Prompts for generation evaluation
            test_references: Reference texts for generated outputs
            compute_perplexity: Whether to compute perplexity
            compute_bleu: Whether to compute BLEU
            compute_rouge: Whether to compute ROUGE
            generation_config: Configuration for text generation
        
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Compute perplexity
        if compute_perplexity:
            perplexity = self.perplexity_evaluator.compute_perplexity(dataloader)
            results['perplexity'] = perplexity
        
        # Compute qualitative metrics
        if test_prompts and test_references:
            if generation_config is None:
                generation_config = {}
            
            # Generate responses
            hypotheses = []
            for prompt in tqdm(test_prompts, desc="Generating responses"):
                response = self.qualitative_evaluator.generate_response(
                    prompt, **generation_config
                )
                hypotheses.append(response)
            
            # Compute BLEU
            if compute_bleu:
                bleu = self.qualitative_evaluator.compute_bleu(test_references, hypotheses)
                results['bleu'] = bleu
            
            # Compute ROUGE
            if compute_rouge:
                rouge_scores = self.qualitative_evaluator.compute_rouge(test_references, hypotheses)
                results.update(rouge_scores)
        
        return results
