import torch
import logging
import argparse
from pathlib import Path
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Import our refactored modules
from models import ModelConfig, NeuralTranslator
from data_utils import BilingualPairDataset, CharacterMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('neural_translator')

class ExperimentConfig:
    """Configuration class for experiment parameters"""
    def __init__(self, args):
        self.data_config = {
            'train_path': args.train_path,
            'val_path': args.val_path,
            'test_path': args.test_path,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'min_freq': args.min_freq
        }
        
        self.model_config = {
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'dropout': args.dropout
        }
        
        self.training_config = {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'warmup_steps': args.warmup_steps,
            'gradient_clip': args.gradient_clip
        }
        
        self.experiment_config = {
            'use_wandb': args.use_wandb,
            'save_dir': args.save_dir,
            'seed': args.seed,
            'early_stopping_patience': args.early_stopping_patience
        }

def parse_arguments():
    """Parse command line arguments for model configuration"""
    parser = argparse.ArgumentParser(description='Neural Translation Model')
    
    # Data paths
    parser.add_argument('--train_path', type=str, required=True,
                      help='Path to training data')
    parser.add_argument('--val_path', type=str, required=True,
                      help='Path to validation data')
    parser.add_argument('--test_path', type=str, required=True,
                      help='Path to test data')
    
    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=256,
                      help='Dimension of token embeddings')
    parser.add_argument('--hidden_dim', type=int, default=512,
                      help='Dimension of hidden layers')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay for regularization')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                      help='Number of warmup steps')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                      help='Gradient clipping value')
    parser.add_argument('--max_length', type=int, default=100,
                      help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=1,
                      help='Minimum token frequency')
    
    # Experiment tracking
    parser.add_argument('--use_wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                      help='Patience for early stopping')
    
    return parser.parse_args()

class ExperimentManager:
    """Manages the experiment setup and execution"""
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.save_path = self._setup_save_directory()
        
    def _setup_device(self):
        """Configure and return the device for training"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_save_directory(self):
        """Create and return the save directory path"""
        save_path = Path(self.config.experiment_config['save_dir'])
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path
    
    def setup_experiment(self):
        """Initialize experiment tracking"""
        if self.config.experiment_config['use_wandb']:
            wandb.init(
                project="neural-translator",
                config={
                    **self.config.data_config,
                    **self.config.model_config,
                    **self.config.training_config
                }
            )

class DataManager:
    """Manages data loading and preprocessing"""
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.source_mapper = CharacterMapper()
        self.target_mapper = CharacterMapper()
    
    def prepare_data(self):
        """Prepare and return data loaders"""
        # Create datasets
        train_dataset = BilingualPairDataset(
            self.config.data_config['train_path'],
            self.source_mapper,
            self.target_mapper
        )
        val_dataset = BilingualPairDataset(
            self.config.data_config['val_path'],
            self.source_mapper,
            self.target_mapper
        )
        test_dataset = BilingualPairDataset(
            self.config.data_config['test_path'],
            self.source_mapper,
            self.target_mapper
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.data_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.data_config['batch_size'],
            num_workers=4,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.data_config['batch_size'],
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

class ModelManager:
    """Manages model creation and training"""
    def __init__(self, config, device):
        self.config = config
        self.device = device
    
    def create_model(self, source_vocab_size, target_vocab_size):
        """Create and return the model"""
        model_config = ModelConfig(
            input_size=source_vocab_size,
            output_size=target_vocab_size,
            **self.config.model_config
        )
        
        model = NeuralTranslator(model_config).to(self.device)
        return model
    
    def create_optimizer(self, model):
        """Create and return the optimizer and scheduler"""
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.training_config['learning_rate'],
            weight_decay=self.config.training_config['weight_decay']
        )
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.training_config['warmup_steps'],
            T_mult=2
        )
        
        return optimizer, scheduler

def train_and_evaluate(args):
    """Main training and evaluation pipeline"""
    # Initialize configuration
    config = ExperimentConfig(args)
    
    # Setup experiment
    experiment_manager = ExperimentManager(config)
    experiment_manager.setup_experiment()
    
    # Prepare data
    data_manager = DataManager(config, experiment_manager.device)
    train_loader, val_loader, test_loader = data_manager.prepare_data()
    
    # Create model
    model_manager = ModelManager(config, experiment_manager.device)
    model = model_manager.create_model(
        data_manager.source_mapper.vocab_size,
        data_manager.target_mapper.vocab_size
    )
    
    # Setup training
    optimizer, scheduler = model_manager.create_optimizer(model)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training_config['epochs']):
        logger.info(f"Epoch {epoch+1}/{config.training_config['epochs']}")
        
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (source_batch, target_batch) in enumerate(train_loader):
            source_batch = source_batch.to(experiment_manager.device)
            target_batch = target_batch.to(experiment_manager.device)
            
            optimizer.zero_grad()
            predictions, _ = model(source_batch, target_batch)
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = predictions.shape
            predictions = predictions[:, 1:].contiguous().view(-1, vocab_size)
            targets = target_batch[:, 1:].contiguous().view(-1)
            
            loss = criterion(predictions, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training_config['gradient_clip']
            )
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for source_batch, target_batch in val_loader:
                source_batch = source_batch.to(experiment_manager.device)
                target_batch = target_batch.to(experiment_manager.device)
                
                predictions, _ = model(source_batch, target_batch)
                batch_size, seq_len, vocab_size = predictions.shape
                predictions = predictions[:, 1:].contiguous().view(-1, vocab_size)
                targets = target_batch[:, 1:].contiguous().view(-1)
                
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        
        if config.experiment_config['use_wandb']:
            wandb.log(metrics)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), experiment_manager.save_path / 'best_model.pt')
            logger.info("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= config.experiment_config['early_stopping_patience']:
                logger.info("Early stopping triggered!")
                break
    
    # Final evaluation
    model.load_state_dict(torch.load(experiment_manager.save_path / 'best_model.pt'))
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for source_batch, target_batch in test_loader:
            source_batch = source_batch.to(experiment_manager.device)
            target_batch = target_batch.to(experiment_manager.device)
            
            predictions, _ = model(source_batch, target_batch)
            batch_size, seq_len, vocab_size = predictions.shape
            predictions = predictions[:, 1:].contiguous().view(-1, vocab_size)
            targets = target_batch[:, 1:].contiguous().view(-1)
            
            loss = criterion(predictions, targets)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Final Test Loss: {avg_test_loss:.4f}")
    
    if config.experiment_config['use_wandb']:
        wandb.log({"test_loss": avg_test_loss})
        wandb.finish()

if __name__ == '__main__':
    args = parse_arguments()
    train_and_evaluate(args)