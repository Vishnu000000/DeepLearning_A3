import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

from models import ModelConfig, NeuralTranslator
from data_utils import BilingualPairDataset, CharacterMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('neural_translator')

class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(self, **kwargs):
        self.embed_dim = kwargs.get('embed_dim', 256)
        self.hidden_dim = kwargs.get('hidden_dim', 512)
        self.num_layers = kwargs.get('num_layers', 2)
        self.dropout = kwargs.get('dropout', 0.3)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs = kwargs.get('epochs', 20)
        self.max_length = kwargs.get('max_length', 100)
        self.min_freq = kwargs.get('min_freq', 1)
        self.use_wandb = kwargs.get('use_wandb', False)
        self.save_dir = kwargs.get('save_dir', 'checkpoints')
        self.seed = kwargs.get('seed', 42)

class TrainingManager:
    """Manages the training process"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_seeds()
        self.setup_directories()
        
    def setup_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
    
    def setup_directories(self):
        """Create necessary directories"""
        self.save_path = Path(self.config.save_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def setup_wandb(self):
        """Initialize Weights & Biases"""
        if self.config.use_wandb:
            wandb.init(
                project="neural-translator",
                config=vars(self.config)
            )

class DataManager:
    """Manages data loading and preprocessing"""
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.source_mapper = CharacterMapper()
        self.target_mapper = CharacterMapper()
    
    def load_data(self, train_path, val_path, test_path):
        """Load and prepare datasets"""
        # Create datasets
        train_dataset = BilingualPairDataset(
            train_path, self.source_mapper, self.target_mapper
        )
        val_dataset = BilingualPairDataset(
            val_path, self.source_mapper, self.target_mapper
        )
        test_dataset = BilingualPairDataset(
            test_path, self.source_mapper, self.target_mapper
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
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
        """Create and initialize the model"""
        model_config = ModelConfig(
            input_size=source_vocab_size,
            output_size=target_vocab_size,
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
        
        model = NeuralTranslator(model_config).to(self.device)
        return model
    
    def create_optimizer(self, model):
        """Create optimizer and learning rate scheduler"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        return optimizer, scheduler

class Trainer:
    """Handles the training process"""
    def __init__(self, model, optimizer, scheduler, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for source_batch, target_batch in tqdm(train_loader, desc="Training"):
            source_batch = source_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions, _ = self.model(source_batch, target_batch)
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = predictions.shape
            predictions = predictions[:, 1:].contiguous().view(-1, vocab_size)
            targets = target_batch[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(predictions, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for source_batch, target_batch in tqdm(data_loader, desc="Evaluating"):
                source_batch = source_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                predictions, _ = self.model(source_batch, target_batch)
                batch_size, seq_len, vocab_size = predictions.shape
                predictions = predictions[:, 1:].contiguous().view(-1, vocab_size)
                targets = target_batch[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)

def train_model(config, train_path, val_path, test_path):
    """Main training function"""
    # Initialize managers
    training_manager = TrainingManager(config)
    training_manager.setup_wandb()
    
    data_manager = DataManager(config, training_manager.device)
    train_loader, val_loader, test_loader = data_manager.load_data(
        train_path, val_path, test_path
    )
    
    model_manager = ModelManager(config, training_manager.device)
    model = model_manager.create_model(
        data_manager.source_mapper.vocab_size,
        data_manager.target_mapper.vocab_size
    )
    
    optimizer, scheduler = model_manager.create_optimizer(model)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    trainer = Trainer(model, optimizer, scheduler, criterion, training_manager.device)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch+1}/{config.epochs}")
        
        # Training phase
        train_loss = trainer.train_epoch(train_loader)
        
        # Validation phase
        val_loss = trainer.evaluate(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        if config.use_wandb:
            wandb.log(metrics)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), training_manager.save_path / 'best_model.pt')
            logger.info("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered!")
                break
    
    # Final evaluation
    model.load_state_dict(torch.load(training_manager.save_path / 'best_model.pt'))
    test_loss = trainer.evaluate(test_loader)
    logger.info(f"Final Test Loss: {test_loss:.4f}")
    
    if config.use_wandb:
        wandb.log({"test_loss": test_loss})
        wandb.finish()

if __name__ == '__main__':
    # Example configuration
    config = TrainingConfig(
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=20,
        max_length=100,
        min_freq=1,
        use_wandb=True,
        save_dir='checkpoints',
        seed=42
    )
    
    # Example paths
    train_path = 'data/train.txt'
    val_path = 'data/val.txt'
    test_path = 'data/test.txt'
    
    train_model(config, train_path, val_path, test_path) 