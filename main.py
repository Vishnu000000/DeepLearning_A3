import torch
import logging
import argparse
from pathlib import Path

# Renamed module imports
from data_processing import prepare_datasets
from model_components import TextEncoder, TextDecoder, SequenceConverter
from training_helpers import ModelRunner, AccuracyCalculator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('train')

def setup_arguments():
    parser = argparse.ArgumentParser(prog='Seq2Seq Trainer')
    
    # Data configuration
    parser.add_argument('-train', required=True, help='Training data path')
    parser.add_argument('-val', required=True, help='Validation data path')
    parser.add_argument('-test', required=True, help='Testing data path')
    
    # Model configuration
    parser.add_argument('-emb', type=int, default=256, help='Embedding size')
    parser.add_argument('-hid', type=int, default=512, help='Hidden layer size')
    parser.add_argument('-layers', type=int, default=2, help='RNN depth')
    parser.add_argument('-cell', choices=['lstm','gru','rnn'], default='lstm')
    parser.add_argument('-drop', type=float, default=0.3, help='Regularization')
    
    # Training configuration
    parser.add_argument('-bsize', type=int, default=32, help='Samples per batch')
    parser.add_argument('-epochs', type=int, default=20, help='Training cycles')
    parser.add_argument('-lr', type=float, default=0.001, help='Step size')
    parser.add_argument('-maxlen', type=int, default=50, help='Sequence limit')
    parser.add_argument('-mincount', type=int, default=1, help='Char frequency')
    
    # Logging and saving
    parser.add_argument('-wandb', action='store_true', help='Enable monitoring')
    parser.add_argument('-save', default='models', help='Checkpoint directory')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def configure_environment(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_model(args, src_vocab, tgt_vocab, device):
    encoder = TextEncoder(
        vocab_size=len(src_vocab.char2idx),
        emb_size=args.emb,
        hidden_size=args.hid,
        num_layers=args.layers,
        cell_type=args.cell,
        dropout=args.drop
    )
    
    decoder = TextDecoder(
        vocab_size=len(tgt_vocab.char2idx),
        emb_size=args.emb,
        hidden_size=args.hid,
        num_layers=args.layers,
        cell_type=args.cell,
        dropout=args.drop
    )
    
    return SequenceConverter(encoder, decoder, device).to(device)

def execute_training():
    params = setup_arguments()
    device = configure_environment(params.seed)
    Path(params.save).mkdir(exist_ok=True)
    
    if params.wandb:
        import wandb
        wandb.init(project='seq2seq-transliterate', config=vars(params))
    
    # Data preparation
    train_iter, val_iter, test_iter, src_vocab, tgt_vocab = prepare_datasets(
        params.train, params.val, params.test,
        batch_size=params.bsize,
        max_length=params.maxlen,
        min_count=params.mincount
    )
    
    # Model initialization
    seq_model = initialize_model(params, src_vocab, tgt_vocab, device)
    
    # Training execution
    trainer = ModelRunner(
        model=seq_model,
        device=device,
        optimizer=torch.optim.Adam(seq_model.parameters(), lr=params.lr)
    )
    
    best_model_path = Path(params.save)/'model.pt'
    trainer.fit(
        train_iter, val_iter,
        num_epochs=params.epochs,
        checkpoint_path=best_model_path,
        use_wandb=params.wandb
    )
    
    # Final evaluation
    evaluator = AccuracyCalculator(seq_model, device, tgt_vocab)
    final_acc = evaluator.measure(test_iter)
    log.info(f"Final Test Score: {final_acc:.2%}")
    
    if params.wandb:
        wandb.log({"final_accuracy": final_acc})
        wandb.finish()

if __name__ == '__main__':
    execute_training()