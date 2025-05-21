import torch
import torch.nn as nn
import wandb
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import random
import numpy as np

from models import HybridSeq2Seq
from data_utils import (
    TextPreprocessor,
    DataAugmenter,
    create_dataloaders
)
from train_utils import (
    TrainingConfig,
    ModelTrainer,
    ModelEvaluator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a hybrid CNN-LSTM model for transliteration')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_file', type=str, required=True, help='Path to validation data')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test data')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_blocks', type=int, default=3, help='Number of CNN blocks')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum token frequency')
    parser.add_argument('--max_vocab_size', type=int, default=None, help='Maximum vocabulary size')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='Teacher forcing ratio')
    
    # Data augmentation arguments
    parser.add_argument('--noise_prob', type=float, default=0.1, help='Probability of adding noise')
    parser.add_argument('--swap_prob', type=float, default=0.1, help='Probability of swapping characters')
    parser.add_argument('--delete_prob', type=float, default=0.1, help='Probability of deleting characters')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if project name is provided
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args)
        )
    
    # Initialize preprocessor and augmenter
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        normalize_unicode=True
    )
    
    augmenter = DataAugmenter(
        noise_prob=args.noise_prob,
        swap_prob=args.swap_prob,
        delete_prob=args.delete_prob
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader, source_vocab, target_vocab = create_dataloaders(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        num_workers=args.num_workers,
        preprocessor=preprocessor,
        augmenter=augmenter
    )
    
    # Initialize model
    model = HybridSeq2Seq(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        device=device
    ).to(device)
    
    # Initialize training config
    train_config = TrainingConfig(
        max_lr=args.max_lr,
        gradient_clip_val=args.gradient_clip_val,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        label_smoothing=args.label_smoothing
    )
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        config=train_config
    )
    
    # Train model
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_path=save_dir / 'best_model.pt',
        wandb_config=wandb.run.config if wandb.run else None
    )
    
    # Evaluate on test set
    evaluator = ModelEvaluator(model, device, target_vocab)
    test_accuracy = evaluator.calculate_accuracy(test_loader)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    if wandb.run:
        wandb.log({'test_accuracy': test_accuracy})
        wandb.finish()

if __name__ == '__main__':
    main() 