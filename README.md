# Sequence-to-Sequence Transliteration Model

This repository contains a PyTorch implementation of a sequence-to-sequence model for transliteration tasks. The model uses a custom encoder-decoder architecture with various RNN types (LSTM, GRU, or vanilla RNN) and supports features like teacher forcing and gradient clipping.

## Features

- Custom encoder-decoder architecture
- Support for different RNN types (LSTM, GRU, RNN)
- Character-level vocabulary management
- Teacher forcing during training
- Gradient clipping
- Wandb integration for experiment tracking
- Comprehensive logging
- Model checkpointing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

The input data should be in CSV format with two columns:
- `source`: Source text (e.g., English words)
- `target`: Target text (e.g., transliterated words)

Example:
```csv
source,target
hello,हैलो
world,वर्ल्ड
```

## Usage

### Training

To train the model, use the `train.py` script:

```bash
python train.py \
    --train_file path/to/train.csv \
    --val_file path/to/val.csv \
    --test_file path/to/test.csv \
    --embed_dim 256 \
    --hidden_dim 512 \
    --num_layers 2 \
    --rnn_type LSTM \
    --dropout 0.3 \
    --batch_size 32 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --max_len 50 \
    --min_freq 1 \
    --wandb_project transliteration \
    --wandb_entity your-username \
    --save_dir checkpoints
```

### Model Architecture

The model consists of three main components:

1. **Encoder**: Processes the source sequence using an RNN (LSTM, GRU, or vanilla RNN)
2. **Decoder**: Generates the target sequence using the same RNN type
3. **Seq2Seq**: Combines the encoder and decoder with teacher forcing

### Training Process

The training process includes:
- Character-level tokenization
- Teacher forcing with configurable ratio
- Gradient clipping
- Model checkpointing
- Validation monitoring
- Wandb integration for experiment tracking

## Project Structure

```
.
├── models.py           # Model architecture definitions
├── data_utils.py      # Data loading and preprocessing
├── train_utils.py     # Training and evaluation utilities
├── train.py           # Main training script
├── requirements.txt   # Project dependencies
└── README.md          # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.