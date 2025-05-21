import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import wandb
from tqdm.notebook import tqdm
import math
from dataclasses import dataclass
import json
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import Counter

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
    """Comprehensive evaluation metrics with detailed tracking"""
    loss: float
    perplexity: float
    accuracy: float
    edit_distance: float
    attention_entropy: float
    prediction_confidence: float
    sequence_length_ratio: float
    beam_diversity: float = 0.0
    repetition_score: float = 0.0

class SequenceAnalyzer:
    """Analyzes sequence properties and patterns"""
    @staticmethod
    def compute_sequence_stats(sequence: str) -> Dict[str, float]:
        """Compute various statistics about a sequence"""
        return {
            'length': len(sequence),
            'unique_chars': len(set(sequence)),
            'char_freq': dict(Counter(sequence)),
            'repetition_score': SequenceAnalyzer._compute_repetition_score(sequence)
        }
    
    @staticmethod
    def _compute_repetition_score(text: str) -> float:
        """Compute a score indicating repetition in the text"""
        if not text:
            return 0.0
        
        # Count repeated substrings
        repeats = 0
        for i in range(len(text)):
            for j in range(i + 1, len(text)):
                if text[i:j] in text[j:]:
                    repeats += 1
        
        return repeats / (len(text) * (len(text) - 1) / 2)

class ModelEvaluator:
    """Advanced model evaluator with comprehensive analysis"""
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 wandb_project: Optional[str] = None):
        self.model = model
        self.device = device
        self.wandb_project = wandb_project
        self.sequence_analyzer = SequenceAnalyzer()
        
        # Initialize metrics tracking
        self.metrics_history: Dict[str, List[float]] = {
            'loss': [], 'perplexity': [], 'accuracy': [],
            'edit_distance': [], 'attention_entropy': [],
            'prediction_confidence': [], 'sequence_length_ratio': [],
            'beam_diversity': [], 'repetition_score': []
        }
    
    def _compute_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Compute similarity between two sequences using multiple metrics"""
        # Normalized edit distance
        edit_dist = self._compute_edit_distance(seq1, seq2)
        
        # Character overlap
        overlap = len(set(seq1) & set(seq2)) / len(set(seq1) | set(seq2))
        
        # Length ratio
        length_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
        
        # Combine metrics
        return (edit_dist + overlap + length_ratio) / 3
    
    def _compute_edit_distance(self, pred: str, target: str) -> float:
        """Compute normalized edit distance with character-level operations"""
        m, n = len(pred), len(target)
        dp = np.zeros((m + 1, n + 1))
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,  # deletion
                        dp[i][j-1] + 1,  # insertion
                        dp[i-1][j-1] + 1  # substitution
                    )
        
        return 1 - (dp[m][n] / max(m, n))
    
    def _analyze_attention_patterns(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Analyze attention patterns and compute various metrics"""
        weights = attention_weights.detach().cpu().numpy()
        
        # Compute entropy
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        
        # Compute sparsity
        sparsity = np.mean(weights < np.mean(weights))
        
        # Compute concentration
        concentration = np.max(weights, axis=-1).mean()
        
        return {
            'entropy': entropy,
            'sparsity': sparsity,
            'concentration': concentration
        }
    
    def evaluate(self,
                test_loader: DataLoader,
                source_processor: Any,
                target_processor: Any) -> EvaluationMetrics:
        """Evaluate model with comprehensive metrics"""
        self.model.eval()
        metrics = defaultdict(float)
        total_steps = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        with torch.no_grad():
            for source, target in tqdm(test_loader, desc="Evaluating"):
                # Move data to device
                source = source.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output, attention_weights = self.model(source, target[:, :-1])
                
                # Compute basic metrics
                loss = criterion(output.reshape(-1, output.size(-1)), target[:, 1:].reshape(-1))
                predictions = output.argmax(dim=-1)
                
                # Update metrics
                metrics['loss'] += loss.item()
                metrics['accuracy'] += (predictions == target[:, 1:]).float().mean().item()
                
                # Compute sequence-level metrics
                for pred, tgt in zip(predictions, target[:, 1:]):
                    pred_text = target_processor.decode(pred.cpu().numpy())
                    tgt_text = target_processor.decode(tgt.cpu().numpy())
                    
                    # Edit distance
                    metrics['edit_distance'] += self._compute_edit_distance(pred_text, tgt_text)
                    
                    # Sequence analysis
                    pred_stats = self.sequence_analyzer.compute_sequence_stats(pred_text)
                    metrics['repetition_score'] += pred_stats['repetition_score']
                
                # Attention analysis
                attention_metrics = self._analyze_attention_patterns(attention_weights)
                metrics['attention_entropy'] += attention_metrics['entropy']
                
                # Prediction confidence
                probs = torch.softmax(output, dim=-1)
                metrics['prediction_confidence'] += torch.max(probs, dim=-1)[0].mean().item()
                
                # Length ratio
                pred_lengths = (predictions != 0).sum(dim=1).float()
                tgt_lengths = (target[:, 1:] != 0).sum(dim=1).float()
                metrics['sequence_length_ratio'] += (pred_lengths / tgt_lengths).mean().item()
                
                total_steps += 1
        
        # Compute averages
        avg_metrics = {k: v / total_steps for k, v in metrics.items()}
        avg_metrics['perplexity'] = math.exp(avg_metrics['loss'])
        
        # Create metrics object
        evaluation_metrics = EvaluationMetrics(**avg_metrics)
        
        # Update history
        for field in evaluation_metrics.__dataclass_fields__:
            self.metrics_history[field].append(getattr(evaluation_metrics, field))
        
        # Log to wandb
        if self.wandb_project:
            wandb.log({f'test_{k}': v for k, v in avg_metrics.items()})
        
        return evaluation_metrics
    
    def visualize_attention(self,
                          source_text: str,
                          target_text: str,
                          source_processor: Any,
                          target_processor: Any,
                          threshold: float = 0.1) -> None:
        """Create interactive attention visualization"""
        # Prepare input
        source_tensor = torch.tensor(
            source_processor.encode(source_text)
        ).unsqueeze(0).to(self.device)
        
        target_tensor = torch.tensor(
            target_processor.encode(target_text)
        ).unsqueeze(0).to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            _, attention_weights = self.model(source_tensor, target_tensor[:, :-1])
        
        # Create visualization
        self._create_attention_plot(
            attention_weights.squeeze().cpu().numpy(),
            list(source_text),
            list(target_text),
            threshold
        )
    
    def _create_attention_plot(self,
                             attention_weights: np.ndarray,
                             source_chars: List[str],
                             target_chars: List[str],
                             threshold: float) -> None:
        """Create attention visualization using matplotlib"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights, 
                   xticklabels=source_chars,
                   yticklabels=target_chars,
                   cmap='viridis')
        plt.title('Attention Weights Visualization')
        plt.xlabel('Source Characters')
        plt.ylabel('Target Characters')
        plt.tight_layout()
        plt.show()
        
        # Log to wandb
        if self.wandb_project:
            wandb.log({'attention_plot': wandb.Image(plt.gcf())})
    
    def plot_metrics(self) -> None:
        """Plot comprehensive evaluation metrics"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot basic metrics
        axes[0, 0].plot(self.metrics_history['loss'], label='Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        
        axes[0, 1].plot(self.metrics_history['perplexity'], label='Perplexity')
        axes[0, 1].set_title('Perplexity')
        axes[0, 1].legend()
        
        # Plot sequence metrics
        axes[1, 0].plot(self.metrics_history['accuracy'], label='Accuracy')
        axes[1, 0].set_title('Accuracy')
        axes[1, 0].legend()
        
        axes[1, 1].plot(self.metrics_history['edit_distance'], label='Edit Distance')
        axes[1, 1].set_title('Edit Distance')
        axes[1, 1].legend()
        
        # Plot advanced metrics
        axes[2, 0].plot(self.metrics_history['attention_entropy'], label='Attention Entropy')
        axes[2, 0].set_title('Attention Entropy')
        axes[2, 0].legend()
        
        axes[2, 1].plot(self.metrics_history['repetition_score'], label='Repetition Score')
        axes[2, 1].set_title('Repetition Score')
        axes[2, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Log to wandb
        if self.wandb_project:
            wandb.log({'metrics_plot': wandb.Image(fig)})

# Legacy functions for backward compatibility
def evaluate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
            device: torch.device) -> float:
    """Legacy evaluation function"""
    config = EvaluationConfig(device=device)
    evaluator = ModelEvaluator(model, device)
    return evaluator.evaluate(val_loader, None, None).loss

def calculate_accuracy(model: nn.Module, data_loader: DataLoader,
                     src_vocab: Any, tgt_vocab: Any,
                     device: torch.device) -> float:
    """Legacy accuracy calculation function"""
    config = EvaluationConfig(device=device)
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(data_loader, src_vocab, tgt_vocab)
    return metrics.accuracy