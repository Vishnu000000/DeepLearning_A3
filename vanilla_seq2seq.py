import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SequenceEncoder(nn.Module):
    """Sequence encoder for vanilla seq2seq"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, rnn_type='lstm', dropout=0.5, bidirectional=False):
        super().__init__()
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.position_encoding = PositionalEncoding(embed_dim)
        
        # RNN layer
        rnn_class = getattr(nn, rnn_type.upper())
        self.rnn = rnn_class(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output processing
        self.output_proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Embed and add positional encoding
        embedded = self.embedding(x)
        embedded = self.position_encoding(embedded)
        
        # Process through RNN
        outputs, hidden = self.rnn(embedded)
        
        # Project and normalize
        outputs = self.output_proj(outputs)
        outputs = self.layer_norm(outputs)
        
        return outputs, hidden

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""
    def __init__(self, dim, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class SequenceDecoder(nn.Module):
    """Sequence decoder for vanilla seq2seq"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, rnn_type='lstm', dropout=0.5):
        super().__init__()
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.position_encoding = PositionalEncoding(embed_dim)
        
        # RNN layer
        rnn_class = getattr(nn, rnn_type.upper())
        self.rnn = rnn_class(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output processing
        self.output_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def decode_step(self, x, hidden):
        # Embed and add positional encoding
        embedded = self.embedding(x)
        embedded = self.position_encoding(embedded)
        
        # Process through RNN
        output, new_hidden = self.rnn(embedded, hidden)
        
        # Generate output
        output = self.output_processor(output)
        
        return output, new_hidden

class NeuralTranslator(nn.Module):
    """Neural sequence-to-sequence translator"""
    def __init__(self, config):
        super().__init__()
        self.encoder = SequenceEncoder(
            vocab_size=config.input_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            rnn_type=config.rnn_type,
            dropout=config.dropout,
            bidirectional=config.bidirectional
        )
        
        self.decoder = SequenceDecoder(
            vocab_size=config.output_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            rnn_type=config.rnn_type,
            dropout=config.dropout
        )
        
        self.config = config
        
    def forward(self, source, target, teacher_ratio=0.5):
        batch_size = target.shape[0]
        max_len = target.shape[1]
        vocab_size = self.config.output_size
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, max_len, vocab_size, device=source.device)
        
        # Encode source sequence
        _, encoder_hidden = self.encoder(source)
        
        # Initialize decoder input
        decoder_input = target[:, 0]
        
        # Decode sequence
        for t in range(1, max_len):
            # Decode step
            logits, encoder_hidden = self.decoder.decode_step(
                decoder_input.unsqueeze(1), encoder_hidden
            )
            
            # Store predictions
            outputs[:, t] = logits.squeeze(1)
            
            # Prepare next input
            if torch.rand(1).item() < teacher_ratio:
                decoder_input = target[:, t]
            else:
                decoder_input = logits.argmax(-1).squeeze(1)
        
        return outputs
    
    def predict(self, source, max_length, device):
        batch_size = source.shape[0]
        vocab_size = self.config.output_size
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, max_length, vocab_size, device=device)
        
        # Encode source sequence
        _, encoder_hidden = self.encoder(source)
        
        # Initialize decoder input
        decoder_input = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Generate sequence
        for t in range(max_length):
            # Decode step
            logits, encoder_hidden = self.decoder.decode_step(
                decoder_input.unsqueeze(1), encoder_hidden
            )
            
            # Store predictions
            outputs[:, t] = logits.squeeze(1)
            
            # Get next token
            decoder_input = logits.argmax(-1).squeeze(1)
            
            # Check for end token
            if (decoder_input == 2).all():
                break
        
        return outputs