import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import wandb
from tqdm import tqdm
import math
from dataclasses import dataclass
import json
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import Counter, defaultdict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration settings for model evaluation"""
    max_length: int = 100
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beam_size: int = 5
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    loss: float
    perplexity: float
    accuracy: float
    edit_distance: float
    attention_entropy: float
    prediction_confidence: float
    sequence_length_ratio: float
    beam_diversity: float
    repetition_score: float
    character_error_rate: float
    word_error_rate: float
    attention_alignment_score: float

class SequenceAnalyzer:
    """Advanced sequence analysis tools"""
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def compute_repetition_score(self, sequence: List[int]) -> float:
        """Compute repetition score for a sequence"""
        if not sequence:
            return 0.0
        
        # Count n-gram repetitions
        n_grams = defaultdict(int)
        total_ngrams = 0
        
        for n in range(2, 5):  # Check 2-4 grams
            for i in range(len(sequence) - n + 1):
                n_gram = tuple(sequence[i:i+n])
                n_grams[n_gram] += 1
                total_ngrams += 1
        
        # Compute repetition ratio
        if total_ngrams == 0:
            return 0.0
        
        repetition_ratio = sum(count - 1 for count in n_grams.values()) / total_ngrams
        return repetition_ratio
    
    def compute_sequence_similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """Compute similarity between two sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        # Convert to sets for Jaccard similarity
        set1 = set(seq1)
        set2 = set(seq2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_attention_patterns(self, 
                                 attention_weights: torch.Tensor,
                                 source_length: int,
                                 target_length: int) -> Dict[str, float]:
        """Analyze attention patterns"""
        # Convert to numpy for analysis
        weights = attention_weights.detach().cpu().numpy()
        
        # Compute attention statistics
        mean_attention = np.mean(weights, axis=0)
        max_attention = np.max(weights, axis=0)
        attention_entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1)
        
        # Compute alignment score
        alignment_score = np.mean(np.max(weights, axis=-1))
        
        return {
            'mean_attention': float(np.mean(mean_attention)),
            'max_attention': float(np.mean(max_attention)),
            'attention_entropy': float(np.mean(attention_entropy)),
            'alignment_score': float(alignment_score)
        }

class ModelEvaluator:
    """Advanced model evaluation with comprehensive metrics"""
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 source_mapper: Any,
                 target_mapper: Any):
        self.model = model
        self.device = device
        self.source_mapper = source_mapper
        self.target_mapper = target_mapper
        self.sequence_analyzer = SequenceAnalyzer()
    
    def compute_edit_distance(self, pred: str, target: str) -> int:
        """Compute Levenshtein distance between sequences"""
        m, n = len(pred), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # deletion
                        dp[i][j-1] + 1,    # insertion
                        dp[i-1][j-1] + 1   # substitution
                    )
        
        return dp[m][n]
    
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute attention entropy"""
        # Convert to numpy
        weights = attention_weights.detach().cpu().numpy()
        
        # Compute entropy
        entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1)
        return float(np.mean(entropy))
    
    def compute_prediction_confidence(self, logits: torch.Tensor) -> float:
        """Compute prediction confidence"""
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1)[0]
        return float(confidence.mean().item())
    
    def evaluate(self, 
                dataloader: DataLoader,
                max_samples: Optional[int] = None) -> EvaluationMetrics:
        """Evaluate model with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        correct_tokens = 0
        total_edit_distance = 0
        total_attention_entropy = 0
        total_confidence = 0
        total_length_ratio = 0
        total_beam_diversity = 0
        total_repetition_score = 0
        total_cer = 0
        total_wer = 0
        total_alignment_score = 0
        
        with torch.no_grad():
            for i, (source, target) in enumerate(tqdm(dataloader, desc="Evaluating")):
                if max_samples and i >= max_samples:
                    break
                
                # Move data to device
                source = source.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                outputs, attention_weights = self.model(source, target)
                
                # Compute loss
                loss = nn.CrossEntropyLoss(ignore_index=0)(
                    outputs.view(-1, outputs.size(-1)),
                    target.view(-1)
                )
                
                # Update metrics
                total_loss += loss.item()
                total_tokens += (target != 0).sum().item()
                correct_tokens += ((outputs.argmax(dim=-1) == target) & (target != 0)).sum().item()
                
                # Compute edit distance
                predictions = outputs.argmax(dim=-1)
                for pred, tgt in zip(predictions, target):
                    pred_str = self.target_mapper.decode(pred.cpu().numpy())
                    tgt_str = self.target_mapper.decode(tgt.cpu().numpy())
                    total_edit_distance += self.compute_edit_distance(pred_str, tgt_str)
                
                # Compute attention metrics
                total_attention_entropy += self.compute_attention_entropy(attention_weights)
                
                # Compute prediction confidence
                total_confidence += self.compute_prediction_confidence(outputs)
                
                # Compute sequence metrics
                for pred, tgt in zip(predictions, target):
                    pred_len = (pred != 0).sum().item()
                    tgt_len = (tgt != 0).sum().item()
                    total_length_ratio += pred_len / tgt_len if tgt_len > 0 else 0
                    
                    # Compute beam diversity
                    beam_outputs = self.model.generate(source, max_length=target.size(1))
                    beam_diversity = self.sequence_analyzer.compute_sequence_similarity(
                        beam_outputs[0].cpu().numpy(),
                        beam_outputs[1].cpu().numpy()
                    )
                    total_beam_diversity += beam_diversity
                    
                    # Compute repetition score
                    total_repetition_score += self.sequence_analyzer.compute_repetition_score(
                        pred.cpu().numpy()
                    )
                
                # Compute error rates
                for pred, tgt in zip(predictions, target):
                    pred_str = self.target_mapper.decode(pred.cpu().numpy())
                    tgt_str = self.target_mapper.decode(tgt.cpu().numpy())
                    
                    # Character Error Rate
                    cer = self.compute_edit_distance(pred_str, tgt_str) / len(tgt_str)
                    total_cer += cer
                    
                    # Word Error Rate (using character-level for simplicity)
                    pred_words = pred_str.split()
                    tgt_words = tgt_str.split()
                    wer = self.compute_edit_distance(' '.join(pred_words), ' '.join(tgt_words)) / len(tgt_words)
                    total_wer += wer
                
                # Compute attention alignment score
                alignment_score = self.sequence_analyzer.analyze_attention_patterns(
                    attention_weights,
                    source.size(1),
                    target.size(1)
                )['alignment_score']
                total_alignment_score += alignment_score
        
        # Compute final metrics
        num_samples = len(dataloader) if not max_samples else max_samples
        metrics = EvaluationMetrics(
            loss=total_loss / num_samples,
            perplexity=math.exp(total_loss / num_samples),
            accuracy=correct_tokens / total_tokens,
            edit_distance=total_edit_distance / num_samples,
            attention_entropy=total_attention_entropy / num_samples,
            prediction_confidence=total_confidence / num_samples,
            sequence_length_ratio=total_length_ratio / num_samples,
            beam_diversity=total_beam_diversity / num_samples,
            repetition_score=total_repetition_score / num_samples,
            character_error_rate=total_cer / num_samples,
            word_error_rate=total_wer / num_samples,
            attention_alignment_score=total_alignment_score / num_samples
        )
        
        return metrics
    
    def visualize_attention(self,
                           source: str,
                           target: str,
                           output_path: str):
        """Visualize attention weights with improved visualization"""
        self.model.eval()
        
        # Convert input to tensor
        source_tensor = torch.tensor(
            [self.source_mapper.encode(source)],
            device=self.device
        )
        target_tensor = torch.tensor(
            [self.target_mapper.encode(target)],
            device=self.device
        )
        
        # Get attention weights
        with torch.no_grad():
            _, attention_weights = self.model(source_tensor, target_tensor)
        
        # Convert to numpy
        weights = attention_weights[0].cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            weights,
            cmap='viridis',
            xticklabels=list(target),
            yticklabels=list(source)
        )
        plt.title('Attention Weights')
        plt.xlabel('Target')
        plt.ylabel('Source')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path)
        plt.close()
    
    def plot_metrics(self,
                    metrics: Dict[str, List[float]],
                    output_path: str):
        """Plot training and evaluation metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot loss and accuracy
        plt.subplot(2, 2, 1)
        plt.plot(metrics['train_loss'], label='Train Loss')
        plt.plot(metrics['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(metrics['train_acc'], label='Train Acc')
        plt.plot(metrics['val_acc'], label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot attention metrics
        plt.subplot(2, 2, 3)
        plt.plot(metrics['attention_entropy'], label='Attention Entropy')
        plt.plot(metrics['alignment_score'], label='Alignment Score')
        plt.title('Attention Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        # Plot error rates
        plt.subplot(2, 2, 4)
        plt.plot(metrics['cer'], label='CER')
        plt.plot(metrics['wer'], label='WER')
        plt.title('Error Rates')
        plt.xlabel('Epoch')
        plt.ylabel('Rate')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def save_metrics(self,
                    metrics: EvaluationMetrics,
                    output_path: str):
        """Save evaluation metrics to file"""
        metrics_dict = {
            'loss': metrics.loss,
            'perplexity': metrics.perplexity,
            'accuracy': metrics.accuracy,
            'edit_distance': metrics.edit_distance,
            'attention_entropy': metrics.attention_entropy,
            'prediction_confidence': metrics.prediction_confidence,
            'sequence_length_ratio': metrics.sequence_length_ratio,
            'beam_diversity': metrics.beam_diversity,
            'repetition_score': metrics.repetition_score,
            'character_error_rate': metrics.character_error_rate,
            'word_error_rate': metrics.word_error_rate,
            'attention_alignment_score': metrics.attention_alignment_score
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

# Legacy functions for backward compatibility
def evaluate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
            device: torch.device) -> float:
    """Legacy evaluation function"""
    config = EvaluationConfig(device=device)
    evaluator = ModelEvaluator(model, device)
    return evaluator.evaluate(val_loader, None).loss

def calculate_accuracy(model: nn.Module, data_loader: DataLoader,
                     src_vocab: Any, tgt_vocab: Any,
                     device: torch.device) -> float:
    """Legacy accuracy calculation function"""
    config = EvaluationConfig(device=device)
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(data_loader, None)
    return metrics.accuracy