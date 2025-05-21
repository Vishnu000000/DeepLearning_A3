import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Any
import logging
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import math
import random
import wandb
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Advanced training configuration"""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    label_smoothing: float = 0.1
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    patience: int = 5
    min_delta: float = 1e-4
    use_amp: bool = True
    use_wandb: bool = False
    checkpoint_dir: str = "checkpoints"
    early_stopping: bool = True
    use_scheduler: bool = True
    use_mixed_precision: bool = True

class DynamicLearningRateScheduler:
    """Dynamic learning rate scheduler with warmup and decay"""
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 config: TrainingConfig,
                 total_steps: int):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        
        # Initialize learning rate
        self.base_lr = config.learning_rate
        self.min_lr = self.base_lr * 0.1
        
        # Warmup and decay parameters
        self.warmup_steps = config.warmup_steps
        self.decay_steps = total_steps - config.warmup_steps
    
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
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]

class ModelTrainer:
    """Advanced model trainer with dynamic optimization"""
    def __init__(self,
                 model: nn.Module,
                 config: TrainingConfig,
                 device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            ignore_index=0  # Padding token
        )
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Initialize early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _compute_loss(self, 
                     outputs: torch.Tensor,
                     targets: torch.Tensor,
                     attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss with attention regularization"""
        # Reshape outputs and targets
        batch_size, seq_len, vocab_size = outputs.size()
        outputs = outputs.view(-1, vocab_size)
        targets = targets.view(-1)
        
        # Compute cross entropy loss
        loss = self.criterion(outputs, targets)
        
        # Add attention regularization if available
        if attention_weights is not None:
            # Encourage sparse attention
            attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-10),
                dim=-1
            ).mean()
            loss = loss + 0.1 * attention_entropy
        
        return loss
    
    def _train_epoch(self, 
                    train_loader: DataLoader,
                    scheduler: Optional[DynamicLearningRateScheduler] = None) -> Dict[str, float]:
        """Train for one epoch with dynamic optimization"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        correct_tokens = 0
        
        # Initialize progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (source, target) in enumerate(pbar):
            # Move data to device
            source = source.to(self.device)
            target = target.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast() if self.config.use_mixed_precision else torch.no_grad():
                outputs, attention_weights = self.model(source, target)
                loss = self._compute_loss(outputs, target, attention_weights)
                loss = loss / self.config.gradient_accumulation
            
            # Backward pass with gradient scaling
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights with gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate
                if scheduler is not None:
                    scheduler.step()
            
            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation
            total_tokens += (target != 0).sum().item()
            correct_tokens += ((outputs.argmax(dim=-1) == target) & (target != 0)).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': correct_tokens / total_tokens
            })
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct_tokens / total_tokens
        
        return {
            'loss': epoch_loss,
            'acc': epoch_acc
        }
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        correct_tokens = 0
        
        with torch.no_grad():
            for source, target in tqdm(val_loader, desc="Validation"):
                # Move data to device
                source = source.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                outputs, attention_weights = self.model(source, target)
                loss = self._compute_loss(outputs, target, attention_weights)
                
                # Update metrics
                total_loss += loss.item()
                total_tokens += (target != 0).sum().item()
                correct_tokens += ((outputs.argmax(dim=-1) == target) & (target != 0)).sum().item()
        
        # Compute validation metrics
        val_loss = total_loss / len(val_loader)
        val_acc = correct_tokens / total_tokens
        
        return {
            'loss': val_loss,
            'acc': val_acc
        }
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              scheduler: Optional[DynamicLearningRateScheduler] = None):
        """Train model with comprehensive monitoring"""
        # Initialize wandb if enabled
        if self.config.use_wandb:
            wandb.init(project="neural-transliteration")
            wandb.watch(self.model)
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            train_metrics = self._train_epoch(train_loader, scheduler)
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            # Update metrics
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['train_acc'].append(train_metrics['acc'])
            self.metrics['val_acc'].append(val_metrics['acc'])
            if scheduler is not None:
                self.metrics['learning_rates'].append(scheduler.get_last_lr()[0])
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Train Acc: {train_metrics['acc']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Acc: {val_metrics['acc']:.4f}")
            
            if self.config.use_wandb:
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['acc'],
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['acc'],
                    'learning_rate': scheduler.get_last_lr()[0] if scheduler else self.config.learning_rate
                })
            
            # Save checkpoint
            if val_metrics['loss'] < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save model
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['acc']
                }
                torch.save(
                    checkpoint,
                    f"{self.config.checkpoint_dir}/best_model.pt"
                )
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.config.early_stopping and self.patience_counter >= self.config.patience:
                logger.info("Early stopping triggered")
                break
        
        # Close wandb
        if self.config.use_wandb:
            wandb.finish()
        
        return self.metrics

def load_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   checkpoint_path: str) -> Tuple[int, float, float]:
    """Load model checkpoint with error handling"""
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return (
            checkpoint['epoch'],
            checkpoint['val_loss'],
            checkpoint['val_acc']
        )
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0, float('inf'), 0.0

