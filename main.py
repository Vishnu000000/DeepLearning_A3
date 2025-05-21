import torch
import torch.nn as nn
import wandb
import argparse
import logging
from pathlib import Path

from models import CustomEncoder, CustomDecoder, CustomSeq2Seq
from data_utils import create_dataloaders
from train_utils import ModelTrainer, ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a sequence-to-sequence model for transliteration')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data file')
    parser.add_argument('--val_file', type=str, required=True, help='Path to validation data file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test data file')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='Type of RNN')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=1, help='Minimum character frequency')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='transliteration', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name')
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.wandb_entity:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    # Create dataloaders
    train_loader, val_loader, test_loader, source_vocab, target_vocab = create_dataloaders(
        args.train_file,
        args.val_file,
        args.test_file,
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_freq=args.min_freq
    )
    
    # Create model
    encoder = CustomEncoder(
        vocab_size=len(source_vocab.char2idx),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout
    )
    
    decoder = CustomDecoder(
        vocab_size=len(target_vocab.char2idx),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout
    )
    
    model = CustomSeq2Seq(encoder, decoder, device).to(device)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_path=save_dir / 'best_model.pt',
        wandb_config=wandb.config if args.wandb_entity else None
    )
    
    # Evaluate model
    evaluator = ModelEvaluator(model, device, target_vocab)
    test_accuracy = evaluator.calculate_accuracy(test_loader)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    
    if args.wandb_entity:
        wandb.log({"test_accuracy": test_accuracy})
        wandb.finish()

if __name__ == '__main__':
    main()