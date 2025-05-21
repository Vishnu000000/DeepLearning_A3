import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, List, Tuple
import wandb
from tqdm.notebook import tqdm
import math
from dataclasses import dataclass
import random

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    label_smoothing: float = 0.1
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    patience: int = 5
    min_delta: float = 1e-4

class DynamicLearningRateScheduler:
    """Dynamic learning rate scheduler with warmup and decay"""
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 config: TrainingConfig,
                 num_training_steps: int):
        self.optimizer = optimizer
        self.config = config
        self.num_training_steps = num_training_steps
        self.current_step = 0
        
        # Initialize learning rate
        self.base_lr = config.learning_rate
        self.min_lr = self.base_lr * 0.1
        
        # Warmup and decay parameters
        self.warmup_steps = config.warmup_steps
        self.decay_steps = num_training_steps - config.warmup_steps
        
    def step(self):
        """Update learning rate based on current step"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / self.decay_steps
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

class ModelTrainer:
    """Advanced model trainer with dynamic optimization"""
    def __init__(self,
                 model: nn.Module,
                 config: TrainingConfig,
                 device: torch.device,
                 wandb_project: Optional[str] = None):
        self.model = model
        self.config = config
        self.device = device
        self.wandb_project = wandb_project
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            ignore_index=0
        )
        
        # Initialize learning rate scheduler
        self.scheduler = None  # Will be set in train()
        
        # Initialize wandb if project is specified
        if wandb_project:
            wandb.init(project=wandb_project)
            wandb.watch(model)
    
    def _compute_loss(self,
                     output: torch.Tensor,
                     target: torch.Tensor,
                     attention_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with additional metrics"""
        # Main loss
        loss = self.criterion(
            output.reshape(-1, output.size(-1)),
            target[:, 1:].reshape(-1)
        )
        
        # Additional metrics
        metrics = {
            'loss': loss.item(),
            'perplexity': math.exp(loss.item())
        }
        
        # Attention regularization if available
        if attention_weights is not None:
            # Encourage sparse attention
            attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-10),
                dim=-1
            ).mean()
            metrics['attention_entropy'] = attention_entropy.item()
            
            # Add entropy regularization
            loss = loss + 0.01 * attention_entropy
        
        return loss, metrics
    
    def _train_epoch(self,
                    train_loader: DataLoader,
                    epoch: int) -> Dict[str, float]:
        """Train for one epoch with dynamic optimization"""
        self.model.train()
        total_loss = 0
        total_steps = 0
        metrics_history = []
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, (source, target) in enumerate(tqdm(train_loader)):
            # Move data to device
            source = source.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            output, attention_weights = self.model(source, target[:, :-1])
            
            # Compute loss and metrics
            loss, metrics = self._compute_loss(output, target, attention_weights)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation
            loss.backward()
            
            # Update weights if gradient accumulation is complete
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation
            total_steps += 1
            metrics_history.append(metrics)
            
            # Log to wandb
            if self.wandb_project:
                wandb.log({
                    'epoch': epoch,
                    'batch': batch_idx,
                    **metrics,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Compute average metrics
        avg_metrics = {
            k: np.mean([m[k] for m in metrics_history])
            for k in metrics_history[0].keys()
        }
        
        return avg_metrics
    
    def _validate(self,
                 val_loader: DataLoader) -> Dict[str, float]:
        """Validate model with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        metrics_history = []
        
        with torch.no_grad():
            for source, target in val_loader:
                # Move data to device
                source = source.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output, attention_weights = self.model(source, target[:, :-1])
                
                # Compute loss and metrics
                loss, metrics = self._compute_loss(output, target, attention_weights)
                
                # Update metrics
                total_loss += loss.item()
                total_steps += 1
                metrics_history.append(metrics)
        
        # Compute average metrics
        avg_metrics = {
            k: np.mean([m[k] for m in metrics_history])
            for k in metrics_history[0].keys()
        }
        
        return avg_metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int) -> Dict[str, List[float]]:
        """Train model with early stopping and dynamic optimization"""
        # Initialize learning rate scheduler
        self.scheduler = DynamicLearningRateScheduler(
            self.optimizer,
            self.config,
            len(train_loader) * num_epochs
        )
        
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': []
        }
        
        # Initialize early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_perplexity'].append(train_metrics['perplexity'])
            history['val_perplexity'].append(val_metrics['perplexity'])
            
            # Log to wandb
            if self.wandb_project:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_perplexity': train_metrics['perplexity'],
                    'val_perplexity': val_metrics['perplexity']
                })
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss - self.config.min_delta:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
        
        return history

class ModelEvaluator:
    def __init__(self, model: nn.Module, device: torch.device, vocab: Any):
        self.model = model
        self.device = device
        self.vocab = vocab
    
    def calculate_accuracy(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Calculate character-level accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for src, tgt in tqdm(dataloader, desc="Calculating Accuracy"):
                src = src.to(self.device)
                
                # Get model predictions
                output = self.model.inference(src, max_len=tgt.shape[1])
                predictions = output.argmax(dim=-1)
                
                # Compare predictions with targets
                correct += (predictions == tgt.to(self.device)).sum().item()
                total += tgt.numel()
        
        return correct / total
    
    def predict(self, text: str) -> str:
        """Generate prediction for a single input text."""
        self.model.eval()
        
        # Convert text to tensor
        tokens = self.vocab.encode(text)
        src = torch.tensor([tokens], device=self.device)
        
        # Get model prediction
        output = self.model.inference(src, max_len=len(text) + 10)
        predictions = output.argmax(dim=-1)
        
        # Convert prediction to text
        predicted_text = self.vocab.decode(predictions[0].cpu().numpy())
        
        return predicted_text

